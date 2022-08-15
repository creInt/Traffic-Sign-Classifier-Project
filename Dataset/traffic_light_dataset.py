from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
import torch
import os
import glob
import numpy as np
import pickle
from torchvision.transforms.functional import InterpolationMode


class TrafficSignDataset(Dataset):
    def __init__(self, cfg, split="train"):
        # TODO: Include split percentage
        super().__init__()
        training_file = ".\\traffic-signs-data\\train.p"
        validation_file = ".\\traffic-signs-data\\valid.p"
        testing_file = ".\\traffic-signs-data\\test.p"
        if split == "train":
            with open(training_file, mode='rb') as f:
                train = pickle.load(f)
            X_train, y_train = train['features'], train['labels']
            self.data = X_train
            self.label = y_train
        if split == "val":
            with open(validation_file, mode='rb') as f:
                valid = pickle.load(f)
            X_valid, y_valid = valid['features'], valid['labels']
            self.data = X_valid
            self.label = y_valid
        if split == "test":
            with open(validation_file, mode='rb') as f:
                test = pickle.load(f)
            X_test, y_test = test['features'], test['labels']
            self.data = X_test
            self.label = y_test
        self.compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.length = self.data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [self.compose(self.data[idx]), torch.from_numpy(np.asarray(self.label[idx]))]

    @ staticmethod
    def process_img(img, compose):
        return compose(img)


if __name__ == "__main__":
    dataset = TrafficSignDataset()
    print(dataset.__getitem__(0))
    print(dataset.__getitem__(dataset.__len__() - 1))
    dataset = TrafficSignDataset(split="val")
    print(dataset.__getitem__(0))
    print(dataset.__getitem__(dataset.__len__() - 1))
