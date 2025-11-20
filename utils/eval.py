import torch
from torch import nn
from models import model as mm
from utils.util import *
import time



def eval_model(hyp_params, valid_loader, test_loader):

    def evaluate(model, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch in enumerate(loader):
                text, audio, vision, batch_Y = (
                    batch["text"],
                    batch["audio"],
                    batch["vision"],
                    batch["label"],
                )
                eval_attr = batch_Y.unsqueeze(-1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = (
                            text.cuda(),
                            audio.cuda(),
                            vision.cuda(),
                            eval_attr.cuda(),
                        )

                batch_size = text.size(0)

                net = nn.DataParallel(model) if batch_size > 10 else model
                preds, _ = net([text, audio, vision])
                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths
    
    model = torch.load(hyp_params.name,map_location=torch.device('cuda'))

    print("Evaluating on validation set...")
    r, t = evaluate(model,test=False)
    acc_valid = eval_senti(r, t)
    print(f"Validation accuracy: {acc_valid}")

    print("Evaluating on test set...")
    r, t = evaluate(model,test=True)
    acc_test = eval_senti(r, t)
    print(f"Test accuracy: {acc_test}")

