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
from Dataset.traffic_light_dataset import TrafficSignDataset
import pandas


def train(args):
    # cfg
    # Setup (save config)
    cfg = load_config("./config/base.yaml")
    # overwrite specific params for e.g. hyperparam search
    cfg.merge_from_list(args.OPTS)
    # cfg, _ = get_git_hash(cfg)
    exp_dir, model_dir, tb_dir = utils.make_exp_Folder(tb_dir_flag=True)
    cfg.freeze()
    utils.save_yaml(exp_dir, cfg)
    print(f"Running training with config\n{cfg}")
    # Load Dataset
    train_data = TrafficSignDataset(cfg, split="train")
    val_data = TrafficSignDataset(cfg, split="val")
    test_data = TrafficSignDataset(cfg, split="test")

    # Train Loader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=1
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=1
    )

    # Class names
    names_path = cfg.DATA.CLASS_NAMES
    data = pandas.read_csv(names_path, index_col=0)
    class_names = data.iloc[:].SignName.values
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
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIM.LR)
    # Loop epoch
    epochs = cfg.OPTIM.EPOCHS
    it = 0
    # tensorboard
    writer = SummaryWriter(comment=tb_dir)
    example_images = iter(train_loader).next()[0].to(device)
    writer.add_graph(model, example_images)
    best_vall_acc = 0
    verbose = args.verbose
    for e in range(epochs):
        train_loss = []
        train_acc = []
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs = data[0].to(device)
            labels = data[1].type(torch.LongTensor).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            it += 1
            y_pred = model.predict(inputs.to(device))
            acc = torch.sum(
                y_pred == labels.type(torch.LongTensor).to(device)
            ) / y_pred.size(dim=0)
            train_acc.append(acc.item())
            train_loss.append(loss.item())

            writer.add_scalar("Loss/training_loss_iteration", loss.item(), it)
            writer.add_scalar("Accuracy/training_acc_iteration", acc.item(), it)

        writer.add_scalar("Loss/training_loss_episode", np.mean(train_loss), e)
        writer.add_scalar("Accuracy/training_acc_episode", np.mean(train_acc), e)

        valLoss = []
        valAcc = []
        model = model.eval()
        for i, sample in enumerate(val_loader):
            with torch.no_grad():
                y_hat = model(sample[0].to(device))
            # Loss
            loss = criterion(y_hat, sample[1].type(torch.LongTensor).to(device))
            y_pred = model.predict(sample[0].to(device))
            acc = torch.sum(
                y_pred == sample[1].type(torch.long).to(device)
            ) / y_pred.size(dim=0)
            valAcc.append(acc.item())
            valLoss.append(loss.item())
        writer.add_scalar("Loss/validation", np.mean(valLoss), e)
        writer.add_scalar("Accuracy/validation", np.mean(valAcc), e)
        if np.mean(valAcc) > best_vall_acc:
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
        if verbose:
            print(
                f"Episode: {e}: Training loss: {np.mean(train_loss):.3f}, Training accuracy: "
                f"{np.mean(train_acc):.3f}"
                f"\n Validation loss: {np.mean(valLoss):.3f}, Validation accuracy: {np.mean(valAcc):.3f}"
            )
        if args.ValFigPlot and e % 10 == 0:
            sample = next(iter(val_loader))
            y_true = sample[1]
            sample[1] = model.predict(sample[0].to(device))
            fig, _ = make_grid(sample, class_names, keys=[0, 1, 2], y_true=y_true)
            fig.savefig(os.path.join(exp_dir, f"Validation_Batch_Episode_{e}.png"))

    torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pth"))


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
    parser.add_argument("--Model", default="SimpleCnn", help="Sets Model", choices=["SimpleCnn", "ResNet"])
    parser.add_argument("--Device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument(
        "OPTS",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    args = parser.parse_args()
    train(args)
