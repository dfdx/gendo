#!/bin/bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu/
export PATH=${PATH}:/usr/local/cuda/bin

export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_PROMPT="a photo of a beautiful woman"
export INSTANCE_DIR="_data/input/gulnara"
export OUTPUT_DIR="_data/output/gulnara"

python3 gendo/dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt='$INSTANCE_PROMPT' \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400