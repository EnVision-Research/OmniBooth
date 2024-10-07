import torch
from diffusers import UniPCMultistepScheduler

import cv2
import os
import tqdm
import pickle
import numpy as np
from torchvision.io import read_image
import torch.nn.functional as F
from torchvision import utils

import torchvision.transforms.functional as tf

import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.ops import masks_to_boxes

# torch.set_default_device('cuda')

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
)

from models.controlnet1x1 import ControlNetModel1x1 as ControlNetModel

from models.pipeline_controlnet_sd_xl import (
    StableDiffusionXLControlNetPipeline as StableDiffusionXLControlNetPipeline,
)
# from models.pipeline_controlnet_1x1_4dunet import (
#     StableDiffusionControlNetPipeline1x1 as StableDiffusionControlNetPipeline,
# )
from models.dino_model import FrozenDinoV2Encoder

# from models.unet_2d_condition_multiview import UNet2DConditionModelMultiview

from args_file import parse_args
from transformers import AutoTokenizer
from utils.dataset_nusmtv import CocoNutImg as NuScenesDataset

from transformers import CLIPTextModel, CLIPTextModelWithProjection

from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig

config = CompilationConfig.Default()
# xformers and Triton are suggested for achieving best performance.
try:
    import xformers

    config.enable_xformers = True
except ImportError:
    print("xformers not installed, skip")

try:
    import triton

    config.enable_triton = True
except ImportError:
    print("Triton not installed, skip")

args = parse_args()


ckp_path = "./exp/out_sd21_cbgs_loss/"

if args.model_path_infer is not None:
    ckp_path = args.model_path_infer

if "checkpoint" not in ckp_path:
    dirs = os.listdir(ckp_path)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    ckp_path = os.path.join(ckp_path, dirs[-1]) if len(dirs) > 0 else ckp_path

height = 448
width = 800

generator = torch.manual_seed(666)
# generator = torch.manual_seed(0)


tokenizer = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer_2",
    revision=args.revision,
    use_fast=False,
)

# text_encoder = CLIPTextModel.from_pretrained(
text_encoder = CLIPTextModelWithProjection.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="text_encoder_2",
    revision=args.revision,
    variant=args.variant,
)

val_dataset = NuScenesDataset(args, tokenizer, args.gen_train_or_val)

save_path = "vis_dir/out_sd21_cbgs_loss2_40/samples"

if args.save_img_path is not None:
    save_path = args.save_img_path

os.makedirs(save_path, exist_ok=True)

print(ckp_path, save_path)


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


def run_inference(rank, world_size, pred_results, input_datas, pipe, args):
    # if rank is not None:
    #     # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # else:
    #     rank = 0
    print(rank)
    # torch.set_default_device(rank)

    pipe.to("cuda")
    dino_encoder.to("cuda")
    text_encoder.to("cuda")
    # pipe.to(rank)
    # dino_encoder.to(rank)
    weight_dtype = torch.float16
    all_list = input_datas[rank]

    config.enable_cuda_graph = False
    # config.enable_cuda_graph = True
    # pipe.controlnet = compile(pipe.controlnet, config)
    # pipe = compile(pipe, config)

    with torch.no_grad():

        # for img_idx in [
        #     0,
        #     122,
        #     555,
        # ]:

        for img_idx in tqdm.tqdm(all_list):
            batch = val_dataset.__getitem__(img_idx)
            mtv_condition = batch["ctrl_img"]  # [None]
            validation_prompts = batch["prompts"]
            # validation_prompts = ['A photorealistic image.' + x for x in validation_prompts]

            curr_h, curr_w = batch["pixel_values"].shape[-2:]
            # import ipdb; ipdb.set_trace()
            # print(curr_h, curr_w)

            # utils.save_image(batch['patches'][0][1][0], 'output.jpg') # C H W  0~1

            prompt_fea = torch.zeros((*batch["ctrl_img"].shape, args.ctrl_channel)).to(
                "cuda", dtype=weight_dtype
            )

            for curr_b, curr_ins_prompt in enumerate(batch["input_ids_ins"]):
                curr_ins_prompt = ["anything"] + curr_ins_prompt
                input_ids = tokenize_captions(curr_ins_prompt, tokenizer).cuda()
                with torch.cuda.amp.autocast():
                    text_features = text_encoder(input_ids, return_dict=True)[
                        "text_embeds"
                        # "pooler_output"
                    ]
                    text_features = controlnet.text_adapter(text_features).to(
                        prompt_fea
                    )
                # import ipdb; ipdb.set_trace()

                for curr_ins_id in range(len(curr_ins_prompt)):
                    prompt_fea[curr_b][batch["ctrl_img"][curr_b] == curr_ins_id] = (
                        text_features[curr_ins_id]
                    )

            if 0:
            # for curr_b, curr_ins_img in enumerate(batch["patches"]):
                curr_ins_id, curr_ins_patch = curr_ins_img[0], curr_ins_img[1].to(
                    prompt_fea
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

                        # import ipdb; ipdb.set_trace()
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
                        # import ipdb; ipdb.set_trace()
                        warp_fea[small_mask][sel_ins] = global_vector

                        prompt_fea[curr_b][
                            curr_box[1] : curr_box[3], curr_box[0] : curr_box[2]
                        ][small_mask] = warp_fea[small_mask]

            mtv_condition = prompt_fea.permute(0, 3, 1, 2)

            # validation_prompts = ['show a photorealistic street view image']
            images_tensor = []

            for _ in range(args.num_validation_images):
                with torch.autocast("cuda"):
                    # import ipdb; ipdb.set_trace()
                    image = pipe(
                        prompt=validation_prompts,
                        image=mtv_condition,
                        num_inference_steps=30,
                        # num_inference_steps=20,
                        generator=generator,
                        height=curr_h,
                        width=curr_w,
                        controlnet_conditioning_scale=1.0,
                        guidance_scale=args.cfg_scale,
                    ).images  # [0]
                image = torch.cat([torch.tensor(np.array(ii)) for ii in image], 1)

                images_tensor.append(image)

            # import ipdb; ipdb.set_trace()

            # raw_img = batch['pixel_values'].permute(1,2,0) * 255
            # gen_img = torch.cat(images_tensor, 1)
            # gen_img = torch.cat([raw_img, gen_img], 1)

            raw_img = (
                batch["pixel_values"]
                .permute(2, 0, 3, 1)
                .reshape(images_tensor[0].shape)
                * 255
            )
            gen_img = torch.cat(images_tensor, 1)

            raw_w, raw_h = batch['patches'][0][4]

            gen_img = tf.resize(gen_img.permute(2,0,1), (raw_h, raw_w)).permute(1,2,0)

            # gen_img = torch.cat([raw_img, gen_img], 1)

            out_path = os.path.join(
                save_path,
                batch['patches'][0][3],
                # f"val_{img_idx:06d}.jpg",
            )

            # import ipdb; ipdb.set_trace()

            cv2.imwrite(
                out_path, cv2.cvtColor(gen_img.cpu().numpy(), cv2.COLOR_RGB2BGR)
            )


if __name__ == "__main__":
    os.system("export NCCL_SOCKET_IFNAME=eth1")

    from torch.multiprocessing import Manager

    # world_size = 8
    world_size = 4

    all_len = len(val_dataset)
    # all_len = 500

    all_list = np.arange(0, all_len, 1)
    # all_list = np.arange(0, all_len, 8)

    all_len_sel = all_list.shape[0]
    val_len = all_len_sel // world_size * world_size
    # import ipdb; ipdb.set_trace()

    all_list_filter = all_list[:val_len]

    all_list_filter = np.split(all_list_filter, world_size)

    input_datas = {}
    for i in range(world_size):
        input_datas[i] = list(all_list_filter[i])
        print(len(input_datas[i]))

    input_datas[0] += list(all_list[val_len:])
    # input_datas[0] += list(np.all_list(val_len, all_len_sel))

    global dino_encoder

    dino_encoder = FrozenDinoV2Encoder()

    controlnet = ControlNetModel.from_pretrained(
        ckp_path, subfolder="controlnet", torch_dtype=torch.float16
    )
    # unet = UNet2DConditionModel.from_pretrained(
    #     ckp_path, subfolder="unet", torch_dtype=torch.float16
    # )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    vae_path = "/hpc2hdd/home/lli181/long_video/animate-anything/download/AI-ModelScope/sdxl-vae-fp16-fix"

    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder=None,
        revision=args.revision,
        variant=args.variant,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path, vae=vae, unet=unet, controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.set_progress_bar_config(disable=True)
    from diffusers.models.attention_processor import AttnProcessor2_0

    pipe.unet.set_attn_processor(AttnProcessor2_0())

    run_inference(args.curr_gpu, 1, None, input_datas, pipe, args)

    # with Manager() as manager:
    #     pred_results = manager.list()
    #     mp.spawn(run_inference, nprocs=world_size, args=(world_size,pred_results,input_datas,pipe,args,), join=True)
