import torch
from torch import nn
import time
from sklearn.cluster import KMeans
import numpy as np
import argparse
import os
import random
from utils.dataloader import getdataloader

def get_features_from_dataloader(dataloader):
    """Extract and concatenate features from different modalities (text, audio, vision)"""
    features_list = []
    
    for batch in dataloader:
        # Extract features from different modalities
        if isinstance(batch, (list, tuple)):
            # Assuming batch format: [text, audio, vision, labels]
            text_features = batch[0]  # Text modality features
            audio_features = batch[1]  # Audio modality features
            vision_features = batch[2]  # Vision modality features
        elif isinstance(batch, dict):
            # If batch is a dictionary with modality keys
            text_features = batch.get('text', batch.get('text_features'))
            audio_features = batch.get('audio', batch.get('audio_features'))
            vision_features = batch.get('vision', batch.get('vision_features'))
        else:
            # Fallback: assume single feature tensor
            text_features = batch
            audio_features = batch
            vision_features = batch
        
        # Concatenate features from all modalities along feature dimension
        if text_features is not None and audio_features is not None and vision_features is not None:
            # Concatenate along the last dimension (feature dimension)
            combined_features = torch.cat([text_features, audio_features, vision_features], dim=-1)
        elif text_features is not None and audio_features is not None:
            # Only text and audio
            combined_features = torch.cat([text_features, audio_features], dim=-1)
        elif text_features is not None and vision_features is not None:
            # Only text and vision
            combined_features = torch.cat([text_features, vision_features], dim=-1)
        elif audio_features is not None and vision_features is not None:
            # Only audio and vision
            combined_features = torch.cat([audio_features, vision_features], dim=-1)
        else:
            # Single modality
            combined_features = text_features if text_features is not None else (audio_features if audio_features is not None else vision_features)
        
        features_list.append(combined_features)
    
    features = torch.cat(features_list, dim=0)
    return features

def predict_k(features, initial_k=3):
    """Predict number of clusters using KMeans with initial k parameter"""
    features = features.cpu().numpy()
    
    # Handle 3D features (samples, time_steps, features)
    if len(features.shape) == 3:
        print(f"Input features shape: {features.shape}")
        print("Flattening temporal dimension for clustering...")
        # Option 1: Average over time dimension
        features_2d = np.mean(features, axis=1)  # Average across time steps
        print(f"Features after averaging: {features_2d.shape}")
        
        # Option 2: Flatten all dimensions (alternative approach)
        # features_2d = features.reshape(features.shape[0], -1)
    else:
        features_2d = features
    
    # Apply KMeans clustering with initial k
    km = KMeans(n_clusters=initial_k).fit(features_2d)
    y_pred = km.labels_

    pred_label_list = np.unique(y_pred)
    drop_out = len(features_2d) / initial_k  # Mimic original logic: len(feats) / data.num_labels
    print(f'drop threshold: {drop_out}')

    cnt = 0
    for label in pred_label_list:
        num = len(y_pred[y_pred == label]) 
        if num < drop_out:
            cnt += 1

    num_labels = len(pred_label_list) - cnt
    print(f'pred_num: {num_labels}')

    return num_labels

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def initiate(args, train_loader, valid_loader, test_loader):
    """Modified initiate function to work with feature data directly"""
    print("Predicting number of clusters from feature data...")
    
    # Extract features directly from dataloader
    features = get_features_from_dataloader(train_loader)
    print(f"Extracted features shape: {features.shape}")
    
    # Predict clusters
    predicted_k = predict_k(features, initial_k=args.initial_k)
    
    print(f"Predicted number of clusters: {predicted_k}")
    return predicted_k

def main():
    """Main function to run cluster prediction directly"""
    parser = argparse.ArgumentParser(description='Predict number of clusters using KMeans from feature data')
    
    # Required arguments
    parser.add_argument('--initial_k', type=int, default=3,
                        help='Initial number of clusters to start with (default: 3)')
    parser.add_argument("--datapath", type=str, default="", help="dataset path")
    
    # Essential arguments for dataloader
    parser.add_argument("--dataset", type=str, default="mosi", help="dataset to use (mosi/mosei/sims)")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading (default: 32)')
    parser.add_argument('--seed', type=int, default=666, help="random seed")
    parser.add_argument('--no_cuda', action='store_true', help="do not use cuda")
    
    args = parser.parse_args()
    
    # Setup seed and CUDA like in main.py
    setup_seed(args.seed)
    
    if torch.cuda.is_available() and not args.no_cuda:
        use_cuda = True
    else:
        use_cuda = False
    
    # Set up hyp_params like in main.py
    hyp_params = args
    hyp_params.use_cuda = use_cuda
    hyp_params.criterion = "L1Loss"
    hyp_params.output_dim = 1
    hyp_params.stage = "pretrain"
    
    try:
        # Initialize dataloaders using the same method as main.py
        print("Initializing dataloaders...")
        dataloder, orig_dim = getdataloader(args)
        train_loader = dataloder["train"]
        valid_loader = dataloder["valid"]
        test_loader = dataloder["test"]
        train_drop_loader = dataloder["droplast"]
        
        # Set additional parameters like in main.py
        hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = (
            len(train_loader),
            len(valid_loader),
            len(test_loader),
        )
        hyp_params.orig_dim = orig_dim
        
        print(f"Train samples: {hyp_params.n_train}")
        print(f"Valid samples: {hyp_params.n_valid}")
        print(f"Test samples: {hyp_params.n_test}")
        print(f"Original dimension: {orig_dim}")
        
        # Run cluster prediction
        predicted_k = initiate(hyp_params, train_loader, valid_loader, test_loader)
        
        print("\n" + "="*50)
        print(f"Cluster Prediction Results:")
        print(f"Initial K: {args.initial_k}")
        print(f"Predicted Clusters: {predicted_k}")
        print("="*50)
        
    except Exception as e:
        print(f"Error during cluster prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# python estimate_k.py --datapath /home/zjusst/tk/multi-model-sentiment/small_dataset/mosei.pkl --dataset mosei --initial_k 50 --batch_size 64 
# k==19

# python estimate_k.py --datapath /home/zjusst/tk/multi-model-sentiment/small_dataset/mosi.pkl --dataset mosi --initial_k 50 --batch_size 64 
# k==18

# python estimate_k.py --datapath /home/zjusst/tk/multi-model-sentiment/small_dataset/sims.pkl --dataset sims --initial_k 50 --batch_size 64 
# k=18