import torch
from torch import nn


class CSDNetLoss(nn.Module):
    def __init__(self, num_classes: int):
        super(CSDNetLoss, self).__init__()

        if num_classes > 2:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, label):
        loss = self.criterion(predict[0], label[:, None, :, :].float())
        # if self.train:
        loss0 = self.criterion(predict[1], label[:, None, :, :].float())
        loss1 = self.criterion(predict[2], label[:, None, :, :].float())
        loss2 = self.criterion(predict[3], label[:, None, :, :].float())

        loss_all = loss0 * 0.2 + loss1 * 0.3 + loss2 * 0.5
        
        if loss_all > 0.074:
            return loss_all * 0.5 + loss * 0.5
        else:
            return loss
        # else:
            # return loss
        