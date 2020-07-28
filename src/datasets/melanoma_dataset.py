# coding: utf-8
from argparse import Namespace, ArgumentParser
from typing import Tuple
import pandas as pd
import skimage.io
import torch
from torch.utils.data import Dataset


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class MelanomaDataset(Dataset):
    def __init__(self, mode: str, config: Namespace, transform=None, use_external=False):
        super().__init__()
        self.mode = mode
        if mode not in ['train', 'val']:
            raise NotImplementedError("Not implemented dataset configuration")
        self.image_folder = config.image_folder
        self.external_image_folder = config.external_image_folder
        self.fold = config.fold
        self.df = pd.read_csv(f"{config.data_path}/{mode}_{config.fold}.csv")
        self.df.loc[:, 'data_t'] = 'competition'
        if use_external:
            print(f'Will use external data for {mode}')
            self.external_df = pd.read_csv(f"{config.data_path}/external_{mode}_{config.fold}.csv")
            self.external_df.loc[:, 'data_t'] = 'external'
            self.df = pd.concat([self.df, self.external_df])
        self.transform = transform
        self.targets = self.df.target.values
        self.target_counts = self.df.target.value_counts().values

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        img_id = row.image_name
        img_type = row.data_t
        if img_type == 'competition':
            img_path = f"{self.image_folder}/{img_id}.jpg"
        else:
            img_path = f"{self.external_image_folder}/{img_id}.jpg"
        image = skimage.io.imread(img_path)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        label = row.target
        target = onehot(2, label)
        return{'features': image, 'target': target}


class MelanomaDatasetTest(Dataset):
    def __init__(self, config: Namespace, transform=None):
        super().__init__()
        self.image_folder = config.test_image_folder
        self.df = pd.read_csv(f"{config.data_path}/test.csv")
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
        return{'features': image, 'img_id': img_id}


class MelanomaDatasetGeneratedData(Dataset):
    def __init__(self, mode: str, config: Namespace, transform=None):
        super().__init__()
        self.mode = mode
        if mode not in ['train', 'val']:
            raise NotImplementedError("Not implemented dataset configuration")
        self.fold = config.fold
        self.df = pd.read_csv(f"{config.data_path}/{config.generated_data_csv}.csv")
        self.image_folder = config.generated_data_image_folder
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
        label_cls_1 = row.target
        label_cls_0 = 1 - label_cls_1
        target = torch.zeros(2, dtype=torch.float32)
        target[0] = label_cls_0
        target[1] = label_cls_1
        return{'features': image, 'target': target}


class AdLearningMelanomaDataset(Dataset):
    def __init__(self, mode: str, config: Namespace, transform=None, use_external=False):
        super().__init__()
        self.mode = mode
        if mode not in ['train', 'val']:
            raise NotImplementedError("Not implemented dataset configuration")
        self.image_folder = config.image_folder
        self.external_image_folder = config.external_image_folder
        self.fold = config.fold
        self.df = pd.read_csv(f"{config.data_path}/{mode}_{config.fold}.csv")
        self.df.loc[:, 'data_t'] = 'competition'
        if use_external:
            print(f'Will use external data for {mode}')
            self.external_df = pd.read_csv(f"{config.data_path}/external_{mode}_{config.fold}.csv")
            self.external_df.loc[:, 'data_t'] = 'external'
            self.df = pd.concat([self.df, self.external_df])
        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        img_id = row.image_name
        img_type = row.data_t
        if img_type == 'competition':
            img_path = f"{self.image_folder}/{img_id}.jpg"
        else:
            img_path = f"{self.external_image_folder}/{img_id}.jpg"
        image = skimage.io.imread(img_path)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        label = row.data_t
        if label == 'competition':
            label = 1
        else:
            label = 0
        target = onehot(2, label)
        return{'features': image, 'target': target}


if __name__ == '__main__':
    # Debug:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--image_folder", default="../../data/jpeg-melanoma-128x128/train")
    parser.add_argument("--external_image_folder", default="../../data/jpeg-isic2019-128x12/train")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--data_path", default="../../data/")
    parser.add_argument("--use_external", default=False)
    args = parser.parse_args()

    import sys
    sys.path.append('../')
    from transforms.albu import get_valid_transforms

    train_ds = MelanomaDataset(
        mode='train',
        config=args,
        transform=get_valid_transforms(),
        use_external=args.use_external
        )
    batch = next(iter(train_ds))
