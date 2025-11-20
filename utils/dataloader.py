import torch
from torch.utils.data import Dataset, DataLoader
import pickle


class MSADataset(Dataset):
    def __init__(self, hyper, split):
        data_path = hyper.datapath

        with open(data_path, "rb") as file:
            data = pickle.load(file)
        self.data = data[split]

        if "sims" in hyper.datapath:
            print("sims dataset, have *3")
            self.data["regression_labels"] *= 3

        self.split = split
        self.mode = hyper.stage
        self.orig_dims = [
            self.data["text"][0].shape[1],
            self.data["audio"][0].shape[1],
            self.data["vision"][0].shape[1],
        ]
        if self.mode == "pseudo" and split == "train":
            self.pseudolabels = torch.load(hyper.selected_label)
            self.pseudoindex = torch.load(hyper.selected_indice)
            self.data["audio"] = self.data["audio"][self.pseudoindex]
            self.data["vision"] = self.data["vision"][self.pseudoindex]
            self.data["text"] = self.data["text"][self.pseudoindex]
            self.data["regression_labels"] = self.data["regression_labels"][
                self.pseudoindex
            ]

    def get_dim(self):
        return self.orig_dims

    def __len__(self):
        return self.data["audio"].shape[0]

    def __getitem__(self, idx):
        if self.mode in ["pretrain", "contrastive","latent"] or self.split in ["test", "valid"]:
            return {
                "idx": idx,
                "audio": torch.tensor(self.data["audio"][idx]).float(),
                "vision": torch.tensor(self.data["vision"][idx]).float(),
                "text": torch.tensor(self.data["text"][idx]).float(),
                "label": torch.tensor(self.data["regression_labels"][idx]).float(),
            }
        elif self.mode == "pseudo" and self.split == "train":
            return {
                "idx": idx,
                "audio": torch.tensor(self.data["audio"][idx]).float(),
                "vision": torch.tensor(self.data["vision"][idx]).float(),
                "text": torch.tensor(self.data["text"][idx]).float(),
                "label": self.pseudolabels[idx],
            }
        elif self.mode == "get_cov":
            return {
                "idx": idx,
                "audio": torch.tensor(self.data["audio"][idx]).float(),
                "vision": torch.tensor(self.data["vision"][idx]).float(),
                "text": torch.tensor(self.data["text"][idx]).float(),
                "label": torch.tensor(self.data["regression_labels"][idx]).float(),
            }


def getdataloader(args):
    dataloaders = {}
    for split in ["train", "valid", "test"]:
        datasets = MSADataset(args, split=split)
        dataloaders[split] = DataLoader(
            datasets, batch_size=args.batch_size, shuffle=True   #old  shuffle=flase 
        )
    dataset_drop = MSADataset(args, split="train")
    dataloaders["droplast"]=DataLoader(
            dataset_drop, batch_size=args.batch_size, shuffle=True,drop_last=True
        )

    orig_dim = datasets.get_dim()

    return dataloaders, orig_dim
