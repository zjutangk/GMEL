#!/bin/bash

# Set GPU device
CUDA_VISIBLE_DEVICES=1

# Run the Python script with nohup to keep it running after terminal closes
nohup python main.py \
    --stage "pretrain" \
    --dataset "mosi" \
    --datapath "small_dataset/mosi.pkl" \
    --name "sourcemodel/mosi.pt" \
    --num_epochs 30 \
    --backbone "latefusion" \
    > log/mosi_train.log 2>&1 &

echo "Training started in background. Check log/mosi_train.log for progress." 