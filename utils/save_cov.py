
import torch
from torch import nn
from utils.util import *
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.model import Latefusion, Earlyfusion
from utils.align_loss import get_cov
from tqdm import tqdm

# Get covariance matrix of final model's representations
def save_cov(hyp_params, train_loader):
    model = torch.load(hyp_params.name,map_location=torch.device('cuda'))
    with torch.no_grad():
        train_representations = []
        train_representations_hidden = []
        train_y=[]
        for batch_id,batch_X in tqdm(enumerate(train_loader)):
            text, audio, vision, batch_Y = (
                batch_X["text"],
                batch_X["audio"],
                batch_X["vision"],
                batch_X["label"],
            )
            
            if hyp_params.use_cuda:
                text = text.cuda()
                audio = audio.cuda()
                vision = vision.cuda()
            
            _, h = model([text, audio, vision])
            _, rep0, rep1, rep2 = model.forward_multi_layer([text, audio, vision])
            rep = torch.cat([rep0, rep1, rep2], dim=1)
            train_representations.append(h)
            train_representations_hidden.append(rep)
            train_y.append(batch_Y)
        train_representations = torch.cat(train_representations)
        train_representations_hidden = torch.cat(train_representations_hidden)
        train_y = torch.cat(train_y)    
        print(train_representations.shape)
        print(train_representations_hidden.shape)
        final_cov = get_cov(train_representations)
        final_cov_hidden = get_cov(train_representations_hidden)
        torch.save(train_representations, f"{hyp_params.name}_train_representations.pt")
        torch.save(train_y, f"{hyp_params.name}_train_y.pt")
        torch.save(train_representations_hidden, f"{hyp_params.name}_train_representations_hidden.pt")
        torch.save(final_cov, f"{hyp_params.name}_final_cov.pt")
        torch.save(final_cov_hidden, f"{hyp_params.name}_final_cov_hidden.pt")