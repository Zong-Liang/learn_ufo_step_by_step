from torch import nn
import torch.nn.functional as F
import torch


class Iou_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        iou_loss = 0.0
        for j in range(b):
            i = torch.sum(target[j, :, :] * pred[j, :, :])
            u = torch.sum(target[j, :, :]) + torch.sum(pred[j, :, :]) - i
            iou = i / u
            iou_loss += 1 - iou
        return iou_loss / b


class Weighed_Bce_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.view(-1, 1, pred.shape[1], pred.shape[2])
        target = target.view(-1, 1, target.shape[1], target.shape[2])
        label_t = (target == 1).float()
        label_f = (target == 0).float()
        p = torch.sum(label_t) / (torch.sum(label_t) + torch.sum(label_f))
        w = torch.zeros_like(target)
        w[target == 1] = p
        w[target == 0] = 1 - p
        loss = F.binary_cross_entropy(pred, target, weight=w)
        return loss


class Cls_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_cls, target_cls):
        # print(pred_cls.shape)
        # print(target_cls.shape)
        loss = F.binary_cross_entropy(pred_cls, target_cls)
        return loss


class S_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = F.smooth_l1_loss(pred, target)
        return loss


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.iou_loss = Iou_Loss()
        self.wbce_loss = Weighed_Bce_Loss()
        self.cls_loss = Cls_Loss()
        self.s_loss = S_Loss()
        self.w_iou_loss = 1
        self.w_wbce_loss = 1
        self.w_cls_loss = 1
        self.w_s_loss = 1

    def forward(self, pred, target, pred_cls, target_cls):
        i_loss = self.iou_loss(pred, target) * self.w_iou_loss
        m_loss = self.wbce_loss(pred, target) * self.w_wbce_loss
        c_loss = self.cls_loss(pred_cls, target_cls) * self.w_cls_loss
        s_loss = self.s_loss(pred, target) * self.w_s_loss
        loss = i_loss + m_loss + c_loss + s_loss

        return loss, i_loss, m_loss, c_loss, s_loss
