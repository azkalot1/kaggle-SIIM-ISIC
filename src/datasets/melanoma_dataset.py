# coding: utf-8
from argparse import Namespace, ArgumentParser
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
import skimage.io
import torch
from torch.utils.data import Dataset


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class MelanomaDataset(Dataset):
    def __init__(self, mode: str, config: Namespace, transform=None):
        super().__init__()
        self.mode = mode
        if mode not in ['train', 'val']:
            raise NotImplementedError("Not implemented dataset configuration")
        self.image_folder = config.image_folder
        self.fold = config.fold
        self.df = pd.read_csv(f"{config.data_path}/{mode}_{config.fold}.csv")
        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        img_id = row.image_name
        img_path = f"{self.image_folder}/{img_id}.jpg"
        image = skimage.io.imread(img_path)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        label = row.target
        target = onehot(2, label)
        return{'features': image, 'target': target}


if __name__ == '__main__':
    # Debug:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--image_folder", default="../../data/jpeg-isic2019-128x128/train")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--data_path", default="../../data/")

    args = parser.parse_args()

    from pylab import rcParams
    import sys
    sys.path.append('../')
    from transforms.albu import get_valid_transforms

    train_ds = MelanomaDataset(
        mode='train',
        config=args,
        transform=get_valid_transforms()
        )

    rcParams['figure.figsize'] = 20, 10

    for _ in range(5):
        batch = train_ds[0]
        plt.imshow(1. - batch['features'].transpose(0, 1).transpose(1, 2).squeeze())
        plt.savefig('image_dataset.png')
