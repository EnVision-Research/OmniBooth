export WANDB_DISABLED=True
export HF_HUB_OFFLINE=True

export MODEL_DIR="./ckp/stable-diffusion-xl-base-1.0"
export VAE_DIR="./ckp/sdxl-vae-fp16-fix"



export EXP_NAME="omnibooth_train"
export OUTPUT_DIR="./ckp/$EXP_NAME"





# accelerate launch --gpu_ids 0,1,2,3,4,5,6,7  --num_processes 8  --main_process_port 3226  train.py \
accelerate launch --gpu_ids 0,  --num_processes 1  --main_process_port 3226  train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path=$VAE_DIR \
 --output_dir=$OUTPUT_DIR \
 --width=1024 \
 --height=1024 \
 --patch_size=364 \
 --learning_rate=4e-5 \
 --num_train_epochs=12 \
 --train_batch_size=1 \
 --mulscale_batch_size=2 \
 --mixed_precision="fp16" \
 --num_validation_images=2 \
 --validation_steps=500 \
 --checkpointing_steps=5000 \
 --checkpoints_total_limit=10 \
 --ctrl_channel=1024 \
 --use_sdxl=True \
 --enable_xformers_memory_efficient_attention \
 --report_to='wandb' \
 --resume_from_checkpoint="latest" \
 --tracker_project_name="omnibooth-demo" 

