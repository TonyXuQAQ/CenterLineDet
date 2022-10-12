import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_segment = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.loss_point = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        losses = {
            "loss_mask": self.loss_segment(ypred[:,0:1],ytgt[:,0:1]) + self.loss_point(ypred[:,1:2],ytgt[:,1:2])
        }
        return losses
