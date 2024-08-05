import torch
from torch import nn
import torch.nn.functional as F

from CloudSegmentation.models.unet import UNet
from CloudSegmentation.models.resnetunet import ResNetUNet
from CloudSegmentation.models.csdnet import CSDNet

from CloudSegmentation.losses.csdnet_loss import CSDNetLoss
from CloudSegmentation.metrics import binary_iou
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class CloudSegmentation(pl.LightningModule):
    def __init__(self, num_channels, num_classes, lr=1e-5):
        super(CloudSegmentation, self).__init__()
        self.save_hyperparameters()

        # self.model = UNet(num_channels, num_classes)
        self.model = ResNetUNet(num_channels, num_classes)
        # self.model = CSDNet(num_channels, num_classes)
        self.activation = nn.Softmax(dim=1) if num_classes > 1 else None

        self.loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.loss = CSDNetLoss(1)

    def forward(self, x):
        out = self.model(x)
        if self.activation:
            out = self.activation(out)
        
        return out
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        predictions = self(images)
        if self.hparams.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(predictions, labels[:, None, :, :].float())
            # loss = self.loss(predictions, labels)
        else:
            ...
        iou = binary_iou(predictions, labels)

        self.log_dict({
            "train_loss": loss,
            "train_iou": iou
        }, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def _evaluate(self, batch, stage=None):
        images, labels = batch
        predictions = self(images)
        if self.hparams.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(predictions, labels[:, None, :, :].float())
            # loss = self.loss(predictions, labels)
        else:
            ...
        iou = binary_iou(predictions, labels)
        if stage:
            self.log_dict({
                f"{stage}_loss": loss,
                f"{stage}_iou": iou
            }, prog_bar=True, logger=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)



if __name__ == "__main__":
    from CloudSegmentation.datasets.cloud38 import Cloud38
    from torch.utils.data import DataLoader
    from torch.utils import data
    from pathlib import Path

    from torchvision.transforms import v2

    crop_size = (192, 192)
    trans = v2.Compose([
        v2.ToImage(),
    #     # RotateTransform(angles=(0, 90, 180, 270)),
        v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomPhotometricDistort(),  # (original paper: intensity shift- scale whole image by 0.9-1.1, and chromatic shift- scale each channel by 0.95-1.05
    #     # RandomSaltPepperNoise(),
    #     # RandomGaussianNoise(),
        v2.RandomCrop(crop_size),
        v2.ToDtype(torch.float32, scale=True),
    ])

    ds = Cloud38(Path('/home/av/data'), True, transform=trans, image_size=(192, 192), include_nir=False, grayscale=True)

    train_set_size = int(len(ds) * 0.8)
    valid_set_size = len(ds) - train_set_size

    seed = torch.Generator().manual_seed(42)

    train_set, valid_set = data.random_split(ds, [train_set_size, valid_set_size], generator=seed)

    segmenter = CloudSegmentation(1, 1, lr=1e-2)

    batch_size=64
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(segmenter, train_dl, val_dl)
