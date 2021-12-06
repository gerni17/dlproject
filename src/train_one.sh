#!/bin/bash

# Setup environment (do not change this)
# conda activate pip_torch
# pip install -r requirements.txt

# Download dataset (do not change this)
# if [ ! -d "/home/ubuntu/miniscapes" ]; then
#   echo "Download miniscapes"
#   aws s3 cp s3://dlad-miniscapes-2021/miniscapes.zip /home/ubuntu/
#   echo "Extract miniscapes"
#   unzip /home/ubuntu/miniscapes.zip -d /home/ubuntu/ | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
#   rm /home/ubuntu/miniscapes.zip
#   echo "\n"
# fi

# Run training
echo "Start training"
cd /home/gian/git/deeplabv3p/src

CUDA_LAUNCH_BLOCKING=1 python3 -m mtl.scripts.train \
  --name sa_bs_16_e_64_w_6_4_try2\
  --log_dir /home/gian/git/deeplabv3p/logs \
  --batch_size 1 \
  --num_epochs 4\
  --dataset_root /home/gian/git/deeplabv3p/data/One \
  --optimizer adam \
  --optimizer_lr 0.0001\
  --model_name deeplabv3p\
  --gpu False

# Wait a moment before stopping the instance to give a chance to debug
echo "Terminate instance in 2 minutes. Use Ctrl+C to cancel the termination..."
