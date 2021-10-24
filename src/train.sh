#!/bin/bash

# Setup environment (do not change this)
conda activate pip_torch
# pip install -r requirements.txt



# Run training
echo "Start training"
cd /home/gian/git/deeplabv3p/src
today=$(date +"%Y-%m-%d")


python3 -m mtl.scripts.train \
  --name first_${today}\
  --log_dir /home/gian/git/deeplabv3p/logs \
  --batch_size 1 \
  --num_epochs 4\
  --dataset_root /home/gian/git/deeplabv3p/data/exp \
  --optimizer adam \
  --optimizer_lr 0.0001\
  --model_name deeplabv3pp\
  --gpu True

# Wait a moment before stopping the instance to give a chance to debug
echo "Terminate instance in 2 minutes. Use Ctrl+C to cancel the termination..."
