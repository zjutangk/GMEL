export OMP_NUM_THREADS=16
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_LAUNCH_BLOCKING=1

python main.py \
    --stage "latent" \
    --pretrained_model "sourcemodel/mosi.pt" \
    --datapath "small_dataset/mosei.pkl" \
    --num_epochs 30 \
    --intere 5 \
    --name "stage1/mosi2mosei_vmf.pt" \
    --pseudolabel "pseudocheckpoint/mosi2mosei_vmf.pt" \
    --backbone "latefusion" \
    --num_cluster 20\
    --num_neighbors 5\
    --batch_size 64\
    --lr 5e-3\
    --fix_para True\
    --use_sk True\