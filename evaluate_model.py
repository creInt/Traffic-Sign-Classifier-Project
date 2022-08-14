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
    if "experiments\\exp4\\model\\best_model.pth":  # args.resume: # TODO
        model.load_state_dict(torch.load("experiments\\exp4\\model\\best_model.pth"))

    # Load Dataset
    test_data = TrafficSignDataset(cfg, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=1
    )
    old = False
    if old:
        training_file = ".\\traffic-signs-data\\train.p"
        validation_file = ".\\traffic-signs-data\\valid.p"
        testing_file = ".\\traffic-signs-data\\test.p"
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        X_train, y_train = train['features'], train['labels']
        X_valid, y_valid = valid['features'], valid['labels']
        X_test, y_test = test['features'], test['labels']
        img = X_test[0]
        X_train = np.moveaxis(X_train, -1, 1)
        X_valid = np.moveaxis(X_valid, -1, 1)
        X_test = np.moveaxis(X_test, -1, 1)
        train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_data = TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid))
        test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=8
    )

    for i, sample in enumerate(test_loader):
        with torch.no_grad():
            y_hat = model(sample[0].to(device))
        y_pred = model.predict(sample[0].to(device))
        acc = torch.sum(
            y_pred == sample[1].type(torch.long).to(device)
        ) / y_pred.size(dim=0)
    print(acc)

    if args.ValFigPlot == 0 or 1:
        sample = next(iter(test_loader))
        sample[1] = model.predict(sample[0].to(device))
        fig, _ = make_grid(sample, np.arange(0, 42), keys=[0, 1])

        # fig.savefig(os.path.join(exp_dir, f"Validation_Batch_Episode_{e}.png")) # TODO

    #torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pth"))


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
