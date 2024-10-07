export WANDB_DISABLED=True
export HF_HUB_OFFLINE=True

export MODEL_DIR="./ckp/stable-diffusion-xl-base-1.0"
export VAE_DIR="./ckp/sdxl-vae-fp16-fix"




export EXP_NAME="omnibooth_train"
export OUTPUT_DIR="./ckp/$EXP_NAME"
export SAVE_IMG_DIR="./vis_dir"
export TRAIN_OR_VAL="val"





CUDA_VISIBLE_DEVICES=0 python infer_instance.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=0   --ctrl_channel=1024 --width=1024  --height=1024  --patch_size=364  --gen_train_or_val=$TRAIN_OR_VAL  --pretrained_model_name_or_path=$MODEL_DIR  --pretrained_vae_model_name_or_path=$VAE_DIR  --text_or_img=text  --cfg_scale=7.5  --num_validation_images=3













