export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CLIP_NAME="openai/clip-vit-large-patch14"
export OUTPUT_DIR="exps/image"
mkdir $OUTPUT_DIR


accelerate launch --mixed_precision fp16 --multi_gpu train_image.py \
 --pretrained_model_name_or_path $MODEL_DIR \
 --pretrained_clip_name_or_path $CLIP_MODEL_DIR \
 --output_dir $OUTPUT_DIR \
 --dataset_base_path datasets/voxceleb2 \
 --dataset_list_path voxceleb2_tfrecord_train_list.txt \
 --resolution 512 \
 --learning_rate 1e-5 \
 --train_batch_size 4 \
 --video_length 1 \
 --dataloader_num_workers 4 \
 --mixed_precision fp16 \
 --gradient_checkpointing \
 --seed 777 \
 --checkpointing_steps 1000 \
 --snr_gamma 5 \


