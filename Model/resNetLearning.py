from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch


class ResNet18(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.sigm = False
        for param in self.model.parameters():
            param.requires_grad = False
        if cfg.NUM_CLASSES > 1:
            self.model.fc = nn.Linear(self.model.fc.in_features, cfg.NUM_CLASSES)
        elif cfg.NUM_CLASSES == 1:
            self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, cfg.NUM_CLASSES),
                                          nn.Sigmoid())
            self.sigm = True

    def forward(self, x):
        return self.model(x)

    def predict(self, x, return_prob=False):
        out = self.forward(x)
        if self.sigm:
            return (out >= 0.5).to(torch.long)
        prob = F.softmax(out, 1)  # TODO
        prob = out
        y_pred = torch.argmax(prob, dim=1)
        if return_prob:
            return y_pred, prob
        return y_pred


def resnet18_creation(cfg):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    if cfg.NUM_CLASSES > 1:
        model.fc = nn.Linear(model.fc.in_features, cfg.NUM_CLASSES)
    elif cfg.NUM_CLASSES == 1:
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, cfg.NUM_CLASSES),
                                 nn.Sigmoid())

    return model
