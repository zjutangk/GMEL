export OMP_NUM_THREADS=16
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_LAUNCH_BLOCKING=1

python main.py \
    --stage "latent" \
    --pretrained_model "sourcemodel/mosi_early.pt" \
    --datapath "small_dataset/sims.pkl" \
    --num_epochs 30 \
    --intere 5 \
    --name "stage1/mosi2sims_vmf.pt" \
    --pseudolabel "pseudocheckpoint/mosi2sims_vmf.pt" \
    --backbone "earlyfusion" \
    --num_cluster 20\
    --num_neighbors 5\
    --batch_size 64\
    --lr 1e-3\
    --fix_para True\
    > log/vmf_mosi/mi2sims_early.log 2>&1 &