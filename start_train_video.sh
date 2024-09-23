export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="exps/video"
mkdir $OUTPUT_DIR

accelerate launch --mixed_precision no --multi_gpu train_video.py \
 --pretrained_model_name_or_path $MODEL_DIR \
 --output_dir $OUTPUT_DIR \
 --dataset_base_path datasets/voxceleb2 \
 --dataset_list_path voxceleb2_tfrecord_train_list.txt \
 --resolution 512 \
 --learning_rate 1e-5 \
 --train_batch_size 1 \
 --video_length 10 \
 --dataloader_num_workers 4 \
 --mixed_precision no \
 --gradient_checkpointing \
 --seed 777 \
 --checkpointing_steps 1000 \
 --snr_gamma 5 \
 --pretrained_motion_module_path checkpoints/pretrained/v3_sd15_mm.ckpt \
 --pretrained_unet_path checkpoints/image/pretrained_unet \
 --pretrained_MIM checkpoints/image/mim.pth \
 --pretrained_clip_name_or_path checkpoints/image/vision_encoder \




