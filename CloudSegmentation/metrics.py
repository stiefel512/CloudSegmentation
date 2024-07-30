import torch


@torch.no_grad()
def accuracy(preds, labels):
    _, preds = torch.max(preds, 1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds))


@torch.no_grad()
def binary_iou(preds, labels):
    SMOOTH = 1e-6
    preds = torch.sigmoid(preds) > 0.5

    intersection = (preds & labels.long()).float().sum((2, 3))
    union = (preds | labels.long()).float().sum((2, 3))
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = torch.clamp(20 * (iou-0.5), 0, 10).ceil() / 10
    return thresholded.mean()


@torch.no_grad()
def binary_dice_coefficient(preds, labels, eps=1e-5):
    preds = preds.contiguous()
    labels = labels.contiguous()

    if len(preds.size()) == 4 and len(labels.size()) == 4:
        intersection = (preds * labels).sum(dim=2).sum(dim=2)  # sum of H,W axis
        coeff = (2. * intersection + eps) / (preds.sum(dim=2).sum(dim=2) + labels.sum(dim=2).sum(dim=2) + eps)
    elif len(preds.size()) == 3 and len(labels.size()) == 3:
        intersection = (preds * labels).sum(dim=1).sum(dim=1)  # H, W f
        coeff = (2. * intersection) / (preds.sum(dim=1).sum(dim=1) + labels.sum(dim=1).sum(dim=1) + eps)

    return coeff.mean()