import torch
import torch.nn.functional as F
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_outs = cfg.NUM_CLASSES
        self.use_batch_norm = cfg.USE_BATCHN
        self.f1 = nn.Conv2d(3, 5, 3)
        self.f1_n = nn.BatchNorm2d(5)
        self.pool = nn.MaxPool2d(2)
        self.f2 = nn.Conv2d(5, 5, 3)
        self.f2_n = nn.BatchNorm2d(5)
        self.f3 = nn.Conv2d(5, 10, 3)
        self.f4 = nn.Linear(160, 250)
        self.f5 = nn.Linear(250, n_outs)
        if n_outs == 1:
            self.sigm = nn.Sigmoid()
        else:
            self.sigm = None

    def forward(self, x):
        out = F.relu(self.f1(x))
        out = self.pool(out)
        out = self.f1_n(out)
        if self.use_batch_norm:
            out = F.relu(self.f2(out))
        out = self.pool(out)
        if self.use_batch_norm:
            out = self.f2_n(out)
        out = F.relu(self.f3(out))
        out_f = out.flatten(1)
        out = self.f4(out_f)
        out = F.relu(out)
        out = self.f5(out)
        if self.sigm:
            out = self.sigm(out)

        return out

    def predict(self, x, return_prob=False):
        out = self.forward(x)
        if self.sigm:
            return (out >= 0.5).to(torch.long)
        prob = F.softmax(out, 1)
        y_pred = torch.argmax(prob, dim=1)
        if return_prob:
            return y_pred, prob
        return y_pred
