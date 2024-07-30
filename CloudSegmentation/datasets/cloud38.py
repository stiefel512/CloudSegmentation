import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from PIL import Image

from pathlib import Path
from typing import Tuple


class Cloud38(Dataset):
    def __init__(self, root: Path, train: bool, transform=None, image_size: Tuple[int, int] = (192, 192)):
        super(Cloud38, self).__init__()

        self.train = train
        if self.train:
            self.root = root / '38-Cloud' / '38-Cloud_training'
            train_test = 'train'
        else:
            self.root = root / '38-Cloud' / '38-Cloud_test'
            train_test = 'test'

        self.transform = transform
        self.image_size = image_size

        r_dir = self.root / f"{train_test}_red"
        g_dir = self.root / f"{train_test}_green"
        b_dir = self.root / f"{train_test}_blue"
        nir_dir = self.root / f"{train_test}_nir"
        gt_dir = self.root / f"{train_test}_gt"

        self.files = [Cloud38._join_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if f.is_file()]

    @classmethod
    def _join_files(cls, r_path: Path, g_dir: Path, b_dir: Path, nir_dir: Path, gt_dir: Path):
        file_dict = {
            'red': r_path,
            'green': g_dir / r_path.name.replace('red', 'green'),
            'blue': b_dir / r_path.name.replace('red', 'blue'),
            'nir': nir_dir / r_path.name.replace('red', 'nir'),
            'gt': gt_dir / r_path.name.replace('red', 'gt')
        }
        return file_dict

    def __len__(self):
        return len(self.files)

    def read_as_array(self, idx, include_nir=False, reverse_dims=True):
        file_dict = self.files[idx]
        rgb = np.stack([
            np.array(Image.open(file_dict['red'])),
            np.array(Image.open(file_dict['green'])),
            np.array(Image.open(file_dict['blue']))
        ], axis=2)

        if include_nir:
            nir = np.expand_dims(np.array(Image.open(file_dict['nir'])), 2)
            rgb = np.concatenate([rgb, nir], axis=2)

        if reverse_dims:
            rgb = rgb.transpose((2, 0, 1))

        return rgb / np.iinfo(rgb.dtype).max

    def read_mask(self, idx, add_dims=False):
        mask = np.array(Image.open(self.files[idx]['gt']))
        mask = np.where(mask == 255, 1, 0)
        return np.expand_dims(mask, 0) if add_dims else mask

    def __getitem__(self, idx):
        x = torch.tensor(self.read_as_array(idx, include_nir=False, reverse_dims=True)).float()
        x = transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)(x)

        if self.train:
            y = torch.tensor(self.read_mask(idx, add_dims=False), dtype=torch.int64)
            y = transforms.Resize(
                self.image_size, interpolation=transforms.InterpolationMode.NEAREST
            )(y[None, ...]).squeeze()
            return x, y
        else:
            return x

    def read_as_pil(self, idx):
        arr = 256 * self.read_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')

    def __repr__(self):
        s = 'Dataset Class for 38-Cloud Dataset with {} files'.format(self.__len__())
        return s


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = Cloud38(Path('/home/av/data'), train=True, transform=None)
    image, label = ds[1000]
    plt.figure(); plt.imshow(image.permute(1, 2, 0)[:, :, :3]); plt.draw(); plt.pause(0.1)

    ...
