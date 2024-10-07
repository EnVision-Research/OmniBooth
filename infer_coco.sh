export WANDB_DISABLED=True

export MODEL_DIR="/hpc2hdd/home/lli181/long_video/animate-anything/download/AI-ModelScope/stable-diffusion-xl-base-1.0"

source /hpc2ssd/softwares/anaconda3/bin/activate pyt2

export HF_HUB_OFFLINE=True


# export EXP_NAME="out_coconut_dino_text_gridsam"
# export EXP_NAME="out_coconut_sdxl"
export EXP_NAME="out_coconut_sdxl_relu"
# export EXP_NAME="out_coconut_dino_text"
# export EXP_NAME="out_coconut_vith_img"
# export EXP_NAME="out_coconut_vith_anything"
export OUTPUT_DIR="./exp/$EXP_NAME"
# export SAVE_IMG_DIR="vis_dir/$EXP_NAME/dreambooth"
export SAVE_IMG_DIR="vis_dir/$EXP_NAME/val2017"
# export SAVE_IMG_DIR="vis_dir/$EXP_NAME/samples_coco_one"
export TRAIN_OR_VAL="val"





# CUDA_VISIBLE_DEVICES=0 python infer_val_dino_db.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=0   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL 

# CUDA_VISIBLE_DEVICES=0 python infer_xl_dino_coco.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=0   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL --mulscale_batch_size=1 --patch_size=364 --num_validation_images=1   --cfg_scale=7 --pretrained_model_name_or_path=$MODEL_DIR


CUDA_VISIBLE_DEVICES=0 python infer_xl_dino_coco.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=0   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL --mulscale_batch_size=1 --patch_size=364 --num_validation_images=1   --cfg_scale=7  --pretrained_model_name_or_path=$MODEL_DIR  & CUDA_VISIBLE_DEVICES=1 python infer_xl_dino_coco.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=1   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL --mulscale_batch_size=1 --patch_size=364 --num_validation_images=1  --cfg_scale=7  --pretrained_model_name_or_path=$MODEL_DIR  & CUDA_VISIBLE_DEVICES=2 python infer_xl_dino_coco.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=2   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL --mulscale_batch_size=1 --patch_size=364 --num_validation_images=1  --cfg_scale=7  --pretrained_model_name_or_path=$MODEL_DIR  & CUDA_VISIBLE_DEVICES=3 python infer_xl_dino_coco.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=3   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL --mulscale_batch_size=1 --patch_size=364 --num_validation_images=1   --cfg_scale=7  --pretrained_model_name_or_path=$MODEL_DIR  

# & CUDA_VISIBLE_DEVICES=4 python infer_xl_dino_coco.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=4   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL --mulscale_batch_size=1 --patch_size=364 --num_validation_images=1   --cfg_scale=7  --pretrained_model_name_or_path=$MODEL_DIR  & CUDA_VISIBLE_DEVICES=5 python infer_xl_dino_coco.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=5   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL --mulscale_batch_size=1 --patch_size=364 --num_validation_images=1  --cfg_scale=7  --pretrained_model_name_or_path=$MODEL_DIR  & CUDA_VISIBLE_DEVICES=6 python infer_xl_dino_coco.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=6   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL --mulscale_batch_size=1 --patch_size=364 --num_validation_images=1  --cfg_scale=7  --pretrained_model_name_or_path=$MODEL_DIR   & CUDA_VISIBLE_DEVICES=7 python infer_xl_dino_coco.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=7   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL --mulscale_batch_size=1 --patch_size=364 --num_validation_images=1   --cfg_scale=7  --pretrained_model_name_or_path=$MODEL_DIR 



# CUDA_VISIBLE_DEVICES=0 python infer_val_dino_db.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=0   --ctrl_channel=1024 --width=1024  --height=1024  --gen_train_or_val=$TRAIN_OR_VAL & CUDA_VISIBLE_DEVICES=1 python infer_val_dino_db.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=1   --ctrl_channel=1024 --gen_train_or_val=$TRAIN_OR_VAL & CUDA_VISIBLE_DEVICES=2 python infer_val_dino_db.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=2   --ctrl_channel=1024 --gen_train_or_val=$TRAIN_OR_VAL & CUDA_VISIBLE_DEVICES=3 python infer_val_dino_db.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=3   --ctrl_channel=1024 --gen_train_or_val=$TRAIN_OR_VAL 



