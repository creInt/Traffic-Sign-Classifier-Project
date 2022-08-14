from utils.visualize import make_grid
import torch
from Dataset.cat_dog_dataset import CatDogDataset
from utils.utils import load_config


cfg = load_config("./config/base.yaml")
dataset = CatDogDataset(cfg.DATA)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
data_iter = iter(train_loader)
fig, axxr = make_grid(data_iter, ("Dog", "Cat"))
fig.savefig("test.png")
