import torch


@torch.no_grad()
def accuracy(preds, labels):
    _, preds = torch.max(preds, 1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds))


@torch.no_grad()
def binary_iou(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = torch.sigmoid(outputs) > 0.5
    intersection = (outputs & labels.long()).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels.long()).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

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