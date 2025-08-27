import torch
import torch.nn.functional as F
import random
import numpy as np


class DiceLoss:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, logits, targets):
        # logits: (N,1,H,W), targets: (N,1,H,W) in {0,1}
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + self.eps
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + self.eps
        dice = num / den
        return 1 - dice.mean()


def bce_dice_loss(logits, targets, alpha=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = DiceLoss()(logits, targets)
    return alpha * bce + (1 - alpha) * dice


@torch.no_grad()
def dice_metric(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    num = 2 * (preds * targets).sum(dim=(2, 3)) + eps
    den = (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))) + eps
    return (num / den).mean().item()


@torch.no_grad()
def iou_metric(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3)) + eps
    return (inter / union).mean().item()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
