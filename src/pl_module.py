import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CyclicLR, CosineAnnealingLR)
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from src.datasets.melanoma_dataset import MelanomaDataset
from src.models.networks import (
    ClassificationSingleHeadMax, ClassificationDounleHeadMax,
    ClassificationSingleHeadConcat, ClassificationDounleHeadConcat)
from src.transforms.albu import get_valid_transforms, get_training_trasnforms
from src.contrib.optimizers import RAdam, Lamb, QHAdamW, Ralamb, Lookahead
import numpy as np


class MelanomaModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.net = self.get_net()
        self.criterion = self.get_criterion()

        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size

        self.val_df = None

    def forward(self, x: torch.tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> dict:
        y_hat = self.forward(batch['features'])
        loss = self.criterion(y_hat, batch['target'])

        y_pred = nn.Softmax(dim=1)(y_hat).detach().cpu().numpy()[:, 1]
        y_true = batch['target'].detach().cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)  # dirty way

        batch_roc_auc = roc_auc_score(y_true, y_pred)
        batch_ap = average_precision_score(y_true, y_pred)

        train_step = {
            "loss": loss,
            "predictions": y_pred,
            "targets": y_true,
            "log":
                {
                    f"train/{self.hparams.criterion}": loss,
                    f"train/batch_roc_auc": batch_roc_auc,
                    f"train/batch_AP": batch_ap,
                },
        }

        return train_step

    def validation_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_hat = self.forward(batch['features'])
        loss = self.criterion(y_hat, batch['target'])

        y_pred = nn.Softmax(dim=1)(y_hat).detach().cpu().numpy()[:, 1]
        y_true = batch['target'].detach().cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)  # dirty way

        batch_roc_auc = roc_auc_score(y_true, y_pred)
        batch_ap = average_precision_score(y_true, y_pred)

        val_step = {
            "val_loss": loss,
            "val_predictions": y_pred,
            "val_targets": y_true,
            "log":
                {
                    f"val/{self.hparams.criterion}": loss,
                    f"val/batch_roc_auc": batch_roc_auc,
                    f"val/batch_AP": batch_ap,
                },
        }

        return val_step

    def training_epoch_end(self, outputs: torch.tensor) -> dict:
        avg_loss = torch.stack(
            [x["loss"] for x in outputs]).mean().cpu().numpy()
        predictions = np.concatenate(
            [x["predictions"] for x in outputs])
        targets = np.concatenate(
            [x["targets"] for x in outputs])

        roc_auc = roc_auc_score(targets, predictions)
        ap = average_precision_score(targets, predictions)

        train_epoch_end = {
            "loss": avg_loss,
            "roc_auc": roc_auc,
            "AP": ap,
            "log": {
                f"train/avg_{self.hparams.criterion}": avg_loss,
                "train/roc_auc": roc_auc,
                "train/AP": ap
            }
        }

        return train_epoch_end

    def validation_epoch_end(self, outputs: torch.tensor) -> dict:
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        predictions = np.concatenate(
            [x["val_predictions"] for x in outputs])
        targets = np.concatenate(
            [x["val_targets"] for x in outputs])

        roc_auc = roc_auc_score(targets, predictions)
        ap = average_precision_score(targets, predictions)

        val_epoch_end = {
            "val_loss": avg_loss,
            "val_roc_auc": roc_auc,
            "val_AP": ap,
            "log": {
                f"val/avg_{self.hparams.criterion}": avg_loss,
                "val/roc_auc": roc_auc,
                "val/AP": ap
            }
        }

        return val_epoch_end

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        train_dataset = MelanomaDataset(
                mode="train",
                config=self.hparams,
                transform=get_training_trasnforms(self.hparams.training_transforms)
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        val_dataset = MelanomaDataset(
                mode="val",
                config=self.hparams,
                transform=get_valid_transforms()
        )

        self.val_df = val_dataset.df

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    @staticmethod
    def net_mapping(model_name: str, model_type: str) -> torch.nn.Module:
        if model_type == 'DounleHeadConcat':
            return ClassificationDounleHeadConcat(model_name)
        elif model_type == 'SingleHeadConcat':
            return ClassificationSingleHeadConcat(model_name)
        elif model_type == 'DounleHeadMax':
            return ClassificationDounleHeadMax(model_name)
        elif model_type == 'SingleHeadMax':
            return ClassificationSingleHeadMax(model_name)
        else:
            raise NotImplementedError("Not a valid model configuration.")

    def get_net(self) -> torch.nn.Module:
        return MelanomaModel.net_mapping(self.hparams.model_name, self.hparams.model_type)

    def get_criterion(self):
        if "bce_with_logits" == self.hparams.criterion:
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("Not a valid criterion configuration.")

    def get_optimizer(self) -> object:
        if "adam" == self.hparams.optimizer:
            return torch.optim.Adam(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay)
        elif "adamw" == self.hparams.optimizer:
            return torch.optim.AdamW(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay)
        elif "sgd" == self.hparams.optimizer:
            return torch.optim.SGD(
                self.net.parameters(),
                lr=self.learning_rate,
                momentum=self.hparams.sgd_momentum,
                weight_decay=self.hparams.sgd_wd,
            )
        elif "radam" == self.hparams.optimizer:
            return RAdam(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif "ralamb" == self.hparams.optimizer:
            return Ralamb(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif "qhadamw" == self.hparams.optimizer:
            return QHAdamW(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif "lamb" == self.hparams.optimizer:
            return Lamb(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif "radam+lookahead" == self.hparams.optimizer:
            optimizer = RAdam(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            return Lookahead(optimizer)
        else:
            raise NotImplementedError("Not a valid optimizer configuration.")

    def get_scheduler(self, optimizer) -> object:
        if "plateau" == self.hparams.scheduler:
            return ReduceLROnPlateau(optimizer)
        elif "plateau+warmup" == self.hparams.scheduler:
            plateau = ReduceLROnPlateau(optimizer)
            return GradualWarmupScheduler(
                optimizer,
                multiplier=self.hparams.warmup_factor,
                total_epoch=self.hparams.warmup_epochs,
                after_scheduler=plateau)
        elif "cyclic" == self.hparams.scheduler:
            return CyclicLR(optimizer,
                            base_lr=self.learning_rate / 100,
                            max_lr=self.learning_rate,
                            step_size_up=4000 / self.batch_size)
        elif "cosine" == self.hparams.scheduler:
            return CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        elif "cosine+warmup" == self.hparams.scheduler:
            cosine = CosineAnnealingLR(
                optimizer, self.hparams.max_epochs - self.hparams.warmup_epochs)
            return GradualWarmupScheduler(
                optimizer,
                multiplier=self.hparams.warmup_factor,
                total_epoch=self.hparams.warmup_epochs,
                after_scheduler=cosine)
        else:
            raise NotImplementedError("Not a valid scheduler configuration.")