import torch
import pandas as pd
import os
import cv2
import copy
import json
import glob
import torch
import random
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as tf
from torchvision.io import read_image
from scipy import sparse
from panopticapi.utils import rgb2id
from pycocotools.coco import COCO

import albumentations as A
from albumentations.pytorch import ToTensorV2


class CocoNutImgDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, trainorval):
        self.args = copy.deepcopy(args)
        self.trainorval = trainorval

        dataroot = args.dataroot_path
        self.dataroot = dataroot

        if trainorval == "val":
            data_file = os.path.join(
                dataroot, "annotations/relabeled_coco_val.json"
            )
            ann_file_captions = f"{dataroot}/coco/annotations/captions_val2017.json"
            self.img_dir = f"{dataroot}/coco/val2017/"
            self.instance_prompt = f"{dataroot}/annotations/my-val.json"
            self.mask_dir = f"{dataroot}/relabeled_coco_val/relabeled_coco_val/"
        elif trainorval == "train":
            data_file = os.path.join(dataroot, "annotations/coconut_s.json")
            ann_file_captions = f"{dataroot}/coco/annotations/captions_train2017.json"
            self.img_dir = f"{dataroot}/coco/train2017/"
            self.instance_prompt = f"{dataroot}/annotations/my-train.json"
            self.mask_dir = f"{dataroot}/coconut_s/coconut_s/"

        with open(self.instance_prompt) as file:  # hw
            self.instance_prompt = json.load(file)
        self.instance_prompt = self.instance_prompt[trainorval]

        self.coco_caption = COCO(ann_file_captions)

        self.dataset = []

        tmp_dict = {}

        for keys, values in self.instance_prompt.items():
            curr_dict = {}
            curr_dict["height"] = values[-1][0]
            curr_dict["width"] = values[-1][1]
            curr_dict["file_name"] = keys.replace("png", "jpg")
            curr_dict["id"] = int(keys[:-4])
            self.dataset.append(curr_dict)

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        transforms_list = [
            # transforms.Resize((self.args.height, self.args.width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]

        if trainorval == "train":
            transforms_list.append(transforms.Normalize([0.5], [0.5]))

        if self.args.num_validation_images != 1 and trainorval == "val":
            self.args.mulscale_batch_size = 3

        self.image_transforms = transforms.Compose(transforms_list)

        mask_transforms_list = [
            transforms.Resize(
                (self.args.height // 8, self.args.width // 8),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ]

        self.mask_transforms = transforms.Compose(mask_transforms_list)

        self.dino_transforms = A.Compose(
            [
                A.Resize(self.args.patch_size, self.args.patch_size),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                ToTensorV2(),
            ]
        )
        self.dino_transforms_noflip = A.Compose(
            [
                A.Resize(self.args.patch_size, self.args.patch_size),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                ToTensorV2(),
            ]
        )

        if self.args.use_sdxl:
            self.tokenizer, self.text_encoders = tokenizer
        else:
            self.tokenizer = tokenizer

        self.weight_dtype = torch.float16
        self.make_ratio_dict()

    def make_ratio_dict(self):
        number = 30

        ratio_dict = {}
        ratio_list = np.linspace(0.5, 2.0, number)
        for i in range(number):
            ratio_dict[i] = []

        for img_idx in range(len(self.dataset)):
            curr_anno = self.dataset[img_idx]

            width = curr_anno["width"]
            height = curr_anno["height"]

            curr_ratio = width / height

            curr_idx = np.argmin(np.abs(ratio_list - curr_ratio))

            ratio_dict[curr_idx].append(img_idx)

        self.ratio_list = ratio_list
        self.ratio_dict = ratio_dict

    def tokenize_captions(self, examples, is_train=True):
        captions = []
        for caption in examples:
            if random.random() < self.args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def tokenize_captions_sdxl(self, prompt_batch, is_train=True):

        original_size = (self.args.width, self.args.height)
        target_size = (self.args.width, self.args.height)
        crops_coords_top_left = (0, 0)

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt_batch,
            self.text_encoders,
            self.tokenizer,
            self.args.proportion_empty_prompts,
            is_train,
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

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
        self,
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

    def __len__(self):
        "Denotes the total number of samples"
        if self.args.use_cbgs and self.trainorval == "train":
            return len(self.sample_indices)
        return self.length

    def __getitem__(self, idx):
        if self.args.use_cbgs and self.trainorval == "train":
            idx = self.sample_indices[idx]
        return self.get_data_dict(idx)

    @torch.no_grad()
    def get_data_dict(self, index):

        curr_anno = self.dataset[index]

        sample_list = [curr_anno]

        width = curr_anno["width"]
        height = curr_anno["height"]

        curr_ratio = width / height
        curr_ratio_idx = np.argmin(np.abs(self.ratio_list - curr_ratio))

        curr_ratio_list = self.ratio_dict[curr_ratio_idx]

        remanding_size = self.args.mulscale_batch_size - 1
        if remanding_size > 0:
            other_anno = random.sample(curr_ratio_list, remanding_size)
            sample_list = sample_list + [self.dataset[x] for x in other_anno]
        else:
            sample_list = sample_list

        mul_pixel_values = []
        mul_ctrl_img = []
        mul_input_ids = []
        mul_input_ids_ins = []
        mul_prompts = []
        mul_patches = []

        curr_height = self.args.height
        curr_width = curr_height * self.ratio_list.tolist()[curr_ratio_idx]
        curr_width = round(curr_width / 8) * 8

        for curr_anno in sample_list:

            file_name = curr_anno["file_name"]

            raw_loca = self.img_dir + file_name.replace("png", "jpg")
            mask_loca = self.mask_dir + file_name.replace("jpg", "png")

            try:
                instance_captions = self.instance_prompt[
                    file_name.replace("jpg", "png")
                ]

                # load panoptic mask
                img_mask = np.asarray(Image.open(mask_loca), dtype=np.uint32)
                img_mask = rgb2id(img_mask)
                ins_num = img_mask.max()
                img_mask = Image.fromarray(img_mask.astype("uint32"))

            except:
                # print('no prompt')
                # print(file_name)
                continue

            img = Image.open(raw_loca).convert("RGB")
            img = img.resize((curr_width, curr_height))
            img_mask = img_mask.resize((curr_width, curr_height), Image.NEAREST)

            use_patch = 1
            if use_patch:
                patches = []
                sel_ins = []
                img_np_raw = np.array(img, dtype=np.uint8)
                mask_np_raw = np.array(img_mask, dtype=np.uint8)
                ins_num = min(30, ins_num)
                all_ins = np.arange(1, ins_num + 1).tolist()

                for id_ins, curr_ins in enumerate(all_ins):
                    # continue
                    if self.args.text_or_img == "mix" or self.args.text_or_img == "img":
                        # keep at least 2 image sample
                        if np.random.randint(0, 100) > 50 and id_ins >= 2:
                            continue

                    mask_np = copy.deepcopy(mask_np_raw)
                    img_np = copy.deepcopy(img_np_raw)
                    img_np[mask_np != curr_ins] = 255
                    mask_np[mask_np != curr_ins] = 0

                    mask_pil = Image.fromarray(mask_np.astype("uint8"))
                    box = mask_pil.getbbox()
                    if (
                        box is None or (box[2] - box[0]) * (box[3] - box[1]) < 256
                    ) and len(patches) != 0:
                        continue

                    img_pil = Image.fromarray(img_np.astype("uint8"))

                    cropped_img = img_pil.crop(box)
                    if "sign" in instance_captions[0][id_ins]:
                        cropped_img = self.dino_transforms_noflip(
                            image=np.array(cropped_img)
                        )
                    else:
                        cropped_img = self.dino_transforms(image=np.array(cropped_img))
                    cropped_img = cropped_img["image"] / 255
                    # cropped_img = self.args.img_preprocess(cropped_img)

                    patches.append(cropped_img[None])
                    sel_ins.append(curr_ins)

                if len(patches) > 0:
                    patches = torch.cat(patches, dim=0)
                else:
                    patches = torch.zeros(
                        (0, 3, self.args.patch_size, self.args.patch_size)
                    )

                edges = (
                    cv2.Canny(
                        np.array(img.resize((curr_width // 8, curr_height // 8))),
                        100,
                        200,
                    )
                    / 255
                )

                patches = [
                    torch.tensor(sel_ins),
                    patches,
                    edges,
                    file_name,
                    (width, height),
                ]

            img_mask = img_mask.resize(
                (curr_width // 8, curr_height // 8), Image.NEAREST
            )

            img = self.image_transforms(img)  # [None]

            fea_mask = torch.tensor(np.array(img_mask))

            img_id = curr_anno["id"]
            # img_id = curr_anno['image_id']
            ann_ids_captions = self.coco_caption.getAnnIds(
                imgIds=[img_id], iscrowd=None
            )
            anns_caption = self.coco_caption.loadAnns(ann_ids_captions)[0]["caption"]
            anns_caption = [anns_caption]

            if self.args.use_sdxl:
                pass
            else:
                input_ids = self.tokenize_captions(anns_caption)
                mul_input_ids.append(input_ids)

            mul_pixel_values.append(img[None])
            mul_ctrl_img.append(fea_mask[None])
            mul_input_ids_ins.append(instance_captions[0])
            mul_prompts.append(anns_caption[0])
            mul_patches.append(patches)

        mul_pixel_values = torch.cat(mul_pixel_values, 0)
        mul_ctrl_img = torch.cat(mul_ctrl_img, 0)

        data_dict = {
            "pixel_values": mul_pixel_values,
            "ctrl_img": mul_ctrl_img,
            # "input_ids": mul_input_ids,
            "input_ids_ins": mul_input_ids_ins,
            "prompts": mul_prompts,
            "patches": mul_patches,
        }

        if self.args.use_sdxl:
            pass
        else:
            mul_input_ids = torch.cat(mul_input_ids, 0)
            data_dict["input_ids"] = mul_input_ids

        return data_dict

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.dataset[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        fore_flag = 0
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
                fore_flag = 1
        if fore_flag == 0:
            # model background as two objects
            for _ in range(120):
                cat_ids.append(self.cat2id["background"])
        return cat_ids

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples for k, v in class_sample_idxs.items()
        }
        for key, value in class_sample_idxs.items():
            print(key, len(value))

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * ratio)
            ).tolist()
        return sample_indices


# vis coconut image
class InstanceDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, trainorval):
        self.args = args
        self.trainorval = trainorval

        dataroot = args.dataroot_path
        self.dataroot = dataroot

        # self.dataset = ["./data/instance_dataset/ff_instance_1"]
        self.dataset = ["./data/instance_dataset/plane"]

        self.dataset = sorted(self.dataset)

        transforms_list = [
            # transforms.Resize((self.args.height, self.args.width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]

        if trainorval == "train":
            transforms_list.append(transforms.Normalize([0.5], [0.5]))

        self.image_transforms = transforms.Compose(transforms_list)

        mask_transforms_list = [
            transforms.Resize(
                (self.args.height // 8, self.args.width // 8),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ]

        self.mask_transforms = transforms.Compose(mask_transforms_list)

        self.dino_transforms = A.Compose(
            [
                A.Resize(self.args.patch_size, self.args.patch_size),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(p=0.5),
                ToTensorV2(),
            ]
        )
        self.dino_transforms_noflip = A.Compose(
            [
                A.Resize(self.args.patch_size, self.args.patch_size),
                # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
                ToTensorV2(),
            ]
        )

        if self.args.use_sdxl:
            self.tokenizer, self.text_encoders = tokenizer
        else:
            self.tokenizer = tokenizer

        self.weight_dtype = torch.float16

    def tokenize_captions(self, examples, is_train=True):
        captions = []
        for caption in examples:
            if random.random() < self.args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def tokenize_captions_sdxl(self, prompt_batch, is_train=True):

        original_size = (self.args.width, self.args.height)
        target_size = (self.args.width, self.args.height)
        crops_coords_top_left = (0, 0)

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt_batch,
            self.text_encoders,
            self.tokenizer,
            self.args.proportion_empty_prompts,
            is_train,
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

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
        self,
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

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.get_data_dict(idx)

    @torch.no_grad()
    def get_data_dict(self, index):

        curr_anno = self.dataset[index]

        mul_pixel_values = []
        mul_ctrl_img = []
        mul_input_ids = []
        mul_input_ids_ins = []
        mul_prompts = []
        mul_patches = []

        file_name = curr_anno.split("/")[-1] + ".png"

        first_mask = np.array(Image.open(curr_anno + "/0000_mask.png"))
        raw_h, raw_w = first_mask.shape[:2]
        ratio = raw_w / raw_h

        curr_width = self.args.height * ratio
        curr_height = self.args.height

        curr_width = round(curr_width / 8) * 8

        with open(curr_anno + "/prompt_dict.json") as f:
            prompt_dict = json.load(f)

        ins_num = len(prompt_dict.keys()) - 1

        img = np.zeros((curr_height, curr_width, 3), dtype=np.uint8)

        img_mask = np.zeros_like(np.array(img, dtype=np.uint8))[:, :, 0]
        ins_rgb = []
        ins_rgb_id = []
        for p_id in range(ins_num):
            mask_loca = curr_anno + f"/{p_id:04d}_mask.png"
            ins_mask = np.array(
                Image.open(mask_loca).resize((curr_width, curr_height)), dtype=np.uint8
            )
            if len(ins_mask.shape) != 2:
                ins_mask = ins_mask[:, :, 0]

            if 'ff' in mask_loca:
                img_mask[ins_mask != 255] = p_id + 1  # other mask, fg0~1, bg1
            else:
                img_mask[ins_mask != 0] = p_id + 1   # coco mask, fg1 bg0

            insrgb_loca = curr_anno + f"/{p_id:04d}.png"
            if os.path.exists(insrgb_loca):
                ins_rgb.append(Image.open(insrgb_loca).convert("RGB"))
                ins_rgb_id.append(p_id + 1)
            else:
                ins_rgb.append(Image.open(mask_loca).convert("RGB"))  # fake img patch

        img_mask = Image.fromarray(img_mask)

        instance_captions = [prompt_dict[f"prompt_{p_id}"] for p_id in range(ins_num)]

        patches = []

        use_patch = 1
        if use_patch:
            sel_ins = []
            img_np_raw = np.array(img, dtype=np.uint8)
            mask_np_raw = np.array(img_mask, dtype=np.uint8)
            # ins_num = min(30, ins_num)
            # all_ins = np.arange(1, ins_num + 1).tolist()
            all_ins = ins_rgb_id

            for id_ins, curr_ins in enumerate(all_ins):

                cropped_img = ins_rgb[curr_ins - 1]

                cropped_img = self.dino_transforms_noflip(image=np.array(cropped_img))

                cropped_img = cropped_img["image"] / 255
                patches.append(cropped_img[None])
                sel_ins.append(curr_ins)

            if len(patches) > 0:
                patches = torch.cat(patches, dim=0)
            else:
                patches = torch.zeros(
                    (0, 3, self.args.patch_size, self.args.patch_size)
                )
            # patches = [torch.tensor(sel_ins), patches]
            patches = [
                torch.tensor(sel_ins),
                patches,
                None,
                file_name,
                (curr_width, curr_height),
            ]

        img_mask = img_mask.resize((curr_width // 8, curr_height // 8), Image.NEAREST)

        img = self.image_transforms(img)  # [None]

        fea_mask = torch.tensor(np.array(img_mask))

        anns_caption = [prompt_dict["global_prompt"]]

        if self.args.use_sdxl:
            input_ids = self.tokenize_captions_sdxl(anns_caption)
        else:
            input_ids = self.tokenize_captions(anns_caption)

        mul_pixel_values.append(img[None])
        mul_ctrl_img.append(fea_mask[None])
        mul_input_ids.append(input_ids)
        mul_input_ids_ins.append(instance_captions)
        mul_prompts.append(anns_caption[0])
        mul_patches.append(patches)

        mul_pixel_values = torch.cat(mul_pixel_values, 0)
        mul_ctrl_img = torch.cat(mul_ctrl_img, 0)
        mul_input_ids = torch.cat(mul_input_ids, 0)

        data_dict = {
            "pixel_values": mul_pixel_values,
            "ctrl_img": mul_ctrl_img,
            "input_ids": mul_input_ids,
            "input_ids_ins": mul_input_ids_ins,
            "prompts": mul_prompts,
            "patches": mul_patches,
        }

        return data_dict

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.dataset[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        fore_flag = 0
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
                fore_flag = 1
        if fore_flag == 0:
            # model background as two objects
            for _ in range(120):
                cat_ids.append(self.cat2id["background"])
        return cat_ids

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples for k, v in class_sample_idxs.items()
        }
        for key, value in class_sample_idxs.items():
            print(key, len(value))

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * ratio)
            ).tolist()
        return sample_indices
