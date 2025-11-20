export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=3
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Run the Python script with nohup to keep it running after terminal closes
CUDA_VISIBLE_DEVICES=3 nohup python main.py \
    --stage "pretrain" \
    --dataset "sims" \
    --datapath "/home/zjusst/tk/sentiment/small_dataset/sims.pkl" \
    --name "sourcemodel/sims.pt" \
    --num_epochs 30 \
    --backbone "latefusion" \
    > log/sims_train.log 2>&1 &

echo "Training started in background. Check log/sims_train.log for progress." 