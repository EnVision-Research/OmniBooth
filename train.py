import argparse
import functools
import gc
import cv2
import logging
import math
import os
import pickle
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from models.controlnet1x1 import ControlNetModel1x1 as ControlNetModel
from models.pipeline_controlnet_sd_xl import (
    StableDiffusionXLControlNetPipeline as StableDiffusionXLControlNetPipeline,
)

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from utils.datasets import CocoNutImgDataset as OmniboothDataset

from models.dino_model import FrozenDinoV2Encoder
from torchvision import utils
from torchvision.ops import masks_to_boxes



if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.26.0.dev0")

logger = get_logger(__name__)


def tokenize_captions(examples, tokenizer, is_train=True):
    captions = []
    for caption in examples:
        captions.append(caption)

    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids



def tokenize_captions_sdxl(args, prompt_batch, tokenizer, is_train=True):
    tokenizer, text_encoders = tokenizer

    original_size = (args.width, args.height)
    target_size = (args.width, args.height)
    crops_coords_top_left = (0, 0)

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        prompt_batch,
        text_encoders,
        tokenizer,
        args.proportion_empty_prompts,
        is_train,
    )
    add_text_embeds = pooled_prompt_embeds

    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids]).to(prompt_embeds)

    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)

    return {
        "prompt_ids": prompt_embeds,
        "unet_added_conditions": {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        },
    }

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(
    prompt_batch,
    text_encoders,
    tokenizers,
    proportion_empty_prompts,
    is_train=True,
):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

@torch.no_grad()
def log_validation(
    vae, unet, controlnet, args, accelerator, weight_dtype, step, val_dataset
):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    for img_idx in [
        0,
        1,
        2,
    ]:
        # for img_idx in [0, 122, 1179]:
        # for img_idx in range(2):
        batch = val_dataset.__getitem__(img_idx)
        mtv_condition = batch["ctrl_img"]  # [None]
        validation_prompts = batch["prompts"]

        curr_h, curr_w = batch["pixel_values"].shape[-2:]

        prompt_fea = torch.zeros((*batch["ctrl_img"].shape, args.ctrl_channel)).to(
            accelerator.device, dtype=weight_dtype
        )

        for curr_b, curr_ins_prompt in enumerate(batch["input_ids_ins"]):
            if len(curr_ins_prompt) > 0:
                curr_ins_prompt = ["anything"] + curr_ins_prompt
                input_ids = tokenize_captions(curr_ins_prompt, tokenizer_two).cuda()
                with torch.cuda.amp.autocast():
                    text_features = text_encoder_infer(input_ids, return_dict=True)[
                        "text_embeds"
                        # "pooler_output"
                    ]
                    text_features = controlnet.text_adapter(text_features).to(
                        prompt_fea
                    )

                for curr_ins_id in range(len(curr_ins_prompt)):
                    prompt_fea[curr_b][batch["ctrl_img"][curr_b] == curr_ins_id] = (
                        text_features[curr_ins_id]
                    )

        for curr_b, curr_ins_img in enumerate(batch["patches"]):
            curr_ins_id, curr_ins_patch = curr_ins_img[0], curr_ins_img[1].to(
                accelerator.device, dtype=weight_dtype
            )
            if curr_ins_id.shape[0] > 0:

                with torch.cuda.amp.autocast():
                    image_features = dino_encoder(curr_ins_patch)
                    image_features = controlnet.dino_adapter(image_features).to(
                        prompt_fea
                    )

                for id_ins, curr_ins in enumerate(curr_ins_id.tolist()):
                    all_vector = image_features[id_ins]
                    global_vector = all_vector[0:1]

                    down_s = args.patch_size // 14

                    patch_vector = (
                        all_vector[1 : 1 + down_s * down_s]
                        .view(1, down_s, down_s, -1)
                        .permute(0, 3, 1, 2)
                    )
                    curr_mask = batch["ctrl_img"][curr_b] == curr_ins

                    if curr_mask.max() < 1:
                        continue

                    curr_box = masks_to_boxes(curr_mask[None])[0].int().tolist()
                    height, width = (
                        curr_box[3] - curr_box[1],
                        curr_box[2] - curr_box[0],
                    )

                    x = torch.linspace(-1, 1, height)
                    y = torch.linspace(-1, 1, width)

                    xx, yy = torch.meshgrid(x, y)
                    grid = torch.stack((xx, yy), dim=2).to(patch_vector)[None]

                    warp_fea = F.grid_sample(
                        patch_vector,
                        grid,
                        mode="bilinear",
                        padding_mode="reflection",
                        align_corners=True,
                    )[0].permute(1, 2, 0)

                    small_mask = curr_mask[
                        curr_box[1] : curr_box[3], curr_box[0] : curr_box[2]
                    ]

                    curr_pix_num = small_mask.sum().item()
                    all_ins = np.arange(0, curr_pix_num)
                    sel_ins = np.random.choice(
                        all_ins, size=(curr_pix_num // 10,), replace=True
                    )
                    # import ipdb; ipdb.set_trace()
                    warp_fea[small_mask][sel_ins] = global_vector

                    prompt_fea[curr_b][
                        curr_box[1] : curr_box[3], curr_box[0] : curr_box[2]
                    ][small_mask] = warp_fea[small_mask]

        mtv_condition = prompt_fea.permute(0, 3, 1, 2)

        images_tensor = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt=validation_prompts,
                    image=mtv_condition,
                    num_inference_steps=30,
                    generator=generator,
                    height=curr_h,
                    width=curr_w,
                    controlnet_conditioning_scale=1.0,
                    guidance_scale=args.cfg_scale,
                ).images  # [0]
            image = torch.cat([torch.tensor(np.array(ii)) for ii in image], 1)

            images_tensor.append(image)

        raw_img = (
            batch["pixel_values"].permute(2, 0, 3, 1).reshape(images_tensor[0].shape)
            * 255
        )
        gen_img = torch.cat(images_tensor, 0)
        gen_img = torch.cat([raw_img, gen_img], 0)

        out_path = os.path.join(
            args.output_dir,
            f"step_{step:06d}_{img_idx:04d}.jpg",
        )

        cv2.imwrite(out_path, cv2.cvtColor(gen_img.cpu().numpy(), cv2.COLOR_RGB2BGR))

    del controlnet
    del pipeline

    return None

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(
                os.path.join(repo_folder, f"images_{i}.png")
            )
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: openrail++
base_model: {base_model}
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    global tokenizer_two, text_encoder_infer

    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_infer = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    args.pretrained_vae_model_name_or_path = vae_path
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    # import ipdb; ipdb.set_trace()

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(
            unet, conditioning_channels=args.ctrl_channel, text_adapter_channel=1280
        )
        # controlnet = ControlNetModel.from_unet(unet)

    # Resuming unet
    if args.resume_from_checkpoint == "latest":
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming unet from checkpoint {path}")
            # import ipdb; ipdb.set_trace()
            # unet = unet.from_pretrained(
            #     os.path.join(args.output_dir, path), subfolder="unet"
            # )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(
                    input_dir, subfolder="controlnet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_infer.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(controlnet.parameters())

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    global dino_encoder
    dino_encoder = FrozenDinoV2Encoder()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)

    # text_encoder = text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_infer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)


    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]


    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = [tokenizers, text_encoders]
    train_dataset = OmniboothDataset(args, tokenizer, "train")
    val_dataset = OmniboothDataset(args, tokenizer, "val")


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):

                batch["pixel_values"] = batch["pixel_values"][0]
                batch["ctrl_img"] = batch["ctrl_img"][0]
                batch["prompts"] =  [x[0] for x in batch["prompts"]]

                prompt_info = tokenize_captions_sdxl(args, batch["prompts"], tokenizer)
                batch.update(prompt_info) 

                prompt_fea = torch.zeros(
                    (*batch["ctrl_img"].shape, args.ctrl_channel)
                ).to(accelerator.device, dtype=weight_dtype)

                for curr_b, curr_ins_prompt in enumerate(batch["input_ids_ins"]):
                    if len(curr_ins_prompt) > 0:
                        curr_ins_prompt = ["anything"] + [x[0] for x in curr_ins_prompt]
                        input_ids = tokenize_captions(curr_ins_prompt, tokenizer_two).cuda()
                        with torch.cuda.amp.autocast():
                            text_features = text_encoder_infer(input_ids, return_dict=True)[
                                "text_embeds"
                                # "pooler_output"
                            ]
                            text_features = controlnet.module.text_adapter(
                                text_features
                            ).to(prompt_fea)

                        for curr_ins_id in range(len(curr_ins_prompt)):
                            prompt_fea[curr_b][
                                batch["ctrl_img"][curr_b] == curr_ins_id
                            ] = text_features[curr_ins_id]

                for curr_b, curr_ins_img in enumerate(batch["patches"]):
                    curr_ins_id, curr_ins_patch = curr_ins_img[0], curr_ins_img[1].to(
                        weight_dtype
                    )

                    if curr_ins_id.shape[1] > 0:
                        with torch.cuda.amp.autocast():
                            image_features = dino_encoder(
                                curr_ins_patch.reshape((-1, *curr_ins_patch.shape[2:]))
                            )
                            image_features = controlnet.module.dino_adapter(
                                image_features
                            ).to(prompt_fea)

                        for id_ins, curr_ins in enumerate(curr_ins_id[0].tolist()):
                            all_vector = image_features[id_ins]
                            global_vector = all_vector[0:1]

                            down_s = args.patch_size // 14

                            patch_vector = (
                                all_vector[1 : 1 + down_s * down_s]
                                .view(1, down_s, down_s, -1)
                                .permute(0, 3, 1, 2)
                            )

                            curr_mask = batch["ctrl_img"][curr_b] == curr_ins

                            if curr_mask.max() < 1:
                                continue

                            curr_box = masks_to_boxes(curr_mask[None])[0].int().tolist()
                            height, width = (
                                curr_box[3] - curr_box[1],
                                curr_box[2] - curr_box[0],
                            )

                            x = torch.linspace(-1, 1, height)
                            y = torch.linspace(-1, 1, width)

                            xx, yy = torch.meshgrid(x, y)
                            grid = torch.stack((xx, yy), dim=2).to(patch_vector)[None]

                            warp_fea = F.grid_sample(
                                patch_vector,
                                grid,
                                mode="bilinear",
                                padding_mode="reflection",
                                align_corners=True,
                            )[0].permute(1, 2, 0)

                            # import ipdb; ipdb.set_trace()

                            small_mask = curr_mask[
                                curr_box[1] : curr_box[3], curr_box[0] : curr_box[2]
                            ]

                            curr_pix_num = small_mask.sum().item()
                            all_ins = np.arange(0, curr_pix_num)
                            sel_ins = np.random.choice(
                                all_ins, size=(curr_pix_num // 10,), replace=True
                            )
                            warp_fea[small_mask][sel_ins] = global_vector

                            prompt_fea[curr_b][
                                curr_box[1] : curr_box[3], curr_box[0] : curr_box[2]
                            ][small_mask] = warp_fea[small_mask]

                batch["conditioning_pixel_values"] = prompt_fea.permute(0, 3, 1, 2)

                # Convert images to latent space
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(
                    dtype=weight_dtype
                )

                # ControlNet conditioning.
                controlnet_image = batch["conditioning_pixel_values"].to(
                    dtype=weight_dtype
                )
                # import ipdb; ipdb.set_trace()
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_ids"],
                    added_cond_kwargs=batch["unet_added_conditions"],
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_ids"].to(dtype=weight_dtype),
                    added_cond_kwargs=batch["unet_added_conditions"],
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype)
                        for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(
                        dtype=weight_dtype
                    ),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")


                def weighted_mse_loss(input, target, weight):
                    return torch.mean(weight * (input - target) ** 2)

                fore_value = (
                    -0.5 * (1 + np.cos(np.pi * global_step / args.max_train_steps))
                    + 2.0
                )
                
                edge_w = torch.cat([x[2] for x in batch["patches"]], 0)
                weight_mask = torch.ones_like(edge_w).to(weight_dtype)

                weight_mask[edge_w != 0] *= fore_value
                # import ipdb; ipdb.set_trace()

                loss = weighted_mse_loss(
                    model_pred.float(),
                    target.float(),
                    weight_mask.unsqueeze(1).repeat(1, target.shape[1], 1, 1),
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)

                        # # save unet
                        # accelerator.unwrap_model(unet).save_pretrained(
                        #     os.path.join(save_path, "unet")
                        # )

                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            val_dataset,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)
        unwrap_model(unet).save_pretrained(
            os.path.join(args.output_dir, f"checkpoint-{global_step}", "unet")
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    from args_file import parse_args

    args = parse_args()
    main(args)
