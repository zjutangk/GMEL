# GMEL

[AAAI 2026] Official PyTorch implementation of the paper "Group-aware Multiscale Ensemble Learning for Test-Time Multimodal Sentiment Analysis".

## Environment Requirements

The code requires the following dependencies (see `requirements.txt`):

```
torch>=1.8.0
numpy
scikit-learn
transformers
scipy
faiss
```


## Datasets

We utilize the extracted data from CMU-MOSEI, CMU-MOSI, and CH-SIMS. 

**Data Extraction:**
The [MMSA-FET Toolkit](https://github.com/thuiar/MMSA-FET) is employed for data extraction. While you may opt for different feature extraction methods, please ensure consistency across all datasets to maintain the same dimensionality for model transfer.

**Download:**
We provide our extracted datasets using one of the extraction methods at Google Drive:
[Download Datasets](https://drive.google.com/file/d/1tQSw1S16ujHQ069W3QTi3BJ49Q8Gya8N/view)

After downloading, please organize the data file (e.g., `mosi.pkl`) into the `small_dataset/` directory or modify the `--datapath` argument in the training scripts.

## Training Pipeline

The training process consists of two main steps: Pre-training and Test-Time Adaptation (TTA).

### Step 1: Pre-training

Run the pre-training script for the target dataset (e.g., MOSI):

```bash
bash scripts/pretrain/mosi_exp.sh
```

### Step 2: estimate_K
run code fro the estiamtion of intent group number K (e.g., MOSEI):
```
python estimate_k.py --datapath /home/zjusst/tk/multi-model-sentiment/small_dataset/mosei.pkl --dataset mosei --initial_k 50 --batch_size 64 
```

### Step 2: Test-Time Adaptation (TTA)

After pre-training, you can perform test-time adaptation or evaluation (e.g. MOSI2SIMS). 
```bash
bash scripts/mosi/mi2sims.sh
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{tang2026group,
  title = {Group-aware Multiscale Ensemble Learning for Test-Time Multimodal Sentiment Analysis},
  author = {Tang, Kai and Tang, Yixuan and Chen, Tianyi and Xu, Haokai and Luo, Qiqi and Zheng, Jin Guang and Zhang, Zhixin and Chen, Gang and Wang, Haobo},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year = {2026}
}
```

