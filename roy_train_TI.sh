export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./my_data/H3D_samples"

CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 29051 roy_train_TI.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<target-hand>" --initializer_token="hand" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="./runs_TI/debugs" \
  --num_vectors 2 \
  --checkpointing_steps=15000 \
  --validation_prompt="A photo of <target-hand>" \
  --num_validation_images=4 \
  --validation_steps=1 \
  --save_steps=1