#!/bin/bash
export OMP_NUM_THREADS=16
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Run the Python script with nohup to keep it running after terminal closes
CUDA_VISIBLE_DEVICES=7 nohup python main.py \
    --stage "pretrain" \
    --dataset "mosei" \
    --datapath "small_dataset/mosei.pkl" \
    --name "sourcemodel/mosei.pt" \
    --num_epochs 30 \
    --backbone "latefusion" \
    > log/base/mosei_train_late_0503.log 2>&1 &

echo "Training started in background. Check log/mosei_train.log for progress." 