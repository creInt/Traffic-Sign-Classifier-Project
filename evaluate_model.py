import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Model.simpleCNN import SimpleCNN
# from Dataset.cat_dog_dataset import CatDogDataset
import argparse
from utils.utils import load_config, get_git_hash
import utils.utils as utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from utils.visualize import make_grid
from Model.resNetLearning import ResNet18
from torchvision import models
import pickle
import matplotlib.pyplot as plt
from Dataset.traffic_light_dataset import TrafficSignDataset
import pandas


def train(args):
    # cfg
    # Setup (save config)
    cfg = load_config("./config/base.yaml")
    # overwrite specific params for e.g. hyperparam search
    cfg.merge_from_list(args.OPTS)
    cfg.freeze()

   # Define model
   # Model
    if args.Device.isdigit():
        if torch.cuda.is_available() and int(args.Device) in range(torch.cuda.device_count()):
            device = torch.device(int(args.Device))
    elif args.Device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"The chosen device {args.Device} is not available")
    print("Device is: ", device)
    if args.Model.lower() == "SimpleCnn".lower():
        model = SimpleCNN(cfg.MODEL)
    elif args.Model.lower() == "ResNet".lower():
        model = ResNet18(cfg.MODEL)
    model.to(device)
    model.load_state_dict(torch.load(args.resume))

    # Load Dataset
    test_data = TrafficSignDataset(cfg.DATA, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=1
    )
    # Class names
    names_path = cfg.DATA.CLASS_NAMES
    data = pandas.read_csv(names_path, index_col=0)
    class_names = data.iloc[:].SignName.values
    # Run Test
    for i, sample in enumerate(test_loader):
        with torch.no_grad():
            y_hat = model(sample[0].to(device))
        y_pred = model.predict(sample[0].to(device))
        acc = torch.sum(
            y_pred == sample[1].type(torch.long).to(device)
        ) / y_pred.size(dim=0)
    print(f"Accuracy on test set: {acc.detach().item()}")

    if args.ValFigPlot == 0 or 1:
        sample = next(iter(test_loader))
        y_true = sample[1]
        sample[1] = model.predict(sample[0].to(device))
        fig, _ = make_grid(sample, class_names, keys=[0, 1, 2], y_true=y_true)
        if args.verbose:
            plt.show()
        if args.save_fig is not None:
            fig.savefig(os.path.join(args.save_fig, f"test_image.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configfile",
        type=str,
        required=False,
        default=None,
        help="Optinal config file to start",
    )
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--resume", default=False, help="Directory to continue training")
    parser.add_argument("--save_fig", default=None, help="Directory to continue training")
    parser.add_argument("--ValFigPlot", default=True)
    parser.add_argument("--Model", default="SimpleCNN", help="Sets Model", choices=["SimpleCnn", "ResNet"])
    parser.add_argument("--Device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument(
        "OPTS",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    args = parser.parse_args()
    train(args)
