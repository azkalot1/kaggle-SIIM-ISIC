import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CyclicLR, CosineAnnealingLR)
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from src.datasets.melanoma_dataset import (
    MelanomaDataset,
    AdLearningMelanomaDataset,
    MelanomaDatasetGeneratedData)
from src.models.networks import (
    ClassificationSingleHeadMax, ClassificationDounleHeadMax,
    ClassificationSingleHeadConcat, ClassificationDounleHeadConcat)
from src.transforms.albu import get_valid_transforms, get_training_trasnforms
from src.transforms.extra_transfroms import mixup_data, mixup_criterion
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

        self.use_mixup = hparams.use_mixup
        if self.use_mixup:
            print('===== Initialized with mixup ====')
        self.alpha = hparams.mixup_alpha

    def forward(self, x: torch.tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> dict:
        if self.use_mixup:
            x, targets_a, targets_b, lam = mixup_data(batch['features'], batch['target'], self.alpha)
            y_hat = self.forward(x)
            loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
        else:
            y_hat = self.forward(batch['features'])
            loss = self.criterion(y_hat, batch['target'])

        # y_pred = nn.Softmax(dim=1)(y_hat).detach().cpu().numpy()[:, 1]
        y_pred = nn.Sigmoid()(y_hat).detach().cpu().numpy()
        # y_true = batch['target'].detach().cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)  # dirty way
        y_true = (batch['target'].detach().cpu().numpy() > 0.5).astype(int)

        if all(y_true == 0) or all(y_true == 1):
            batch_roc_auc = 0.0
            batch_ap = 0.0
        else:
            batch_roc_auc = roc_auc_score(y_true, y_pred)  # if we use generated \ soft labels
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

        # y_pred = nn.Softmax(dim=1)(y_hat).detach().cpu().numpy()[:, 1]
        y_pred = nn.Sigmoid()(y_hat).detach().cpu().numpy()
        # y_true = batch['target'].detach().cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)  # dirty way
        y_true = batch['target'].detach().cpu().numpy()
        if all(y_true == 0) or all(y_true == 1):
            batch_roc_auc = 0.0
            batch_ap = 0.0
        else:
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
        targets = (targets > 0.5).astype(int)
        if all(targets == 0) or all(targets == 1):
            roc_auc = 0.0
            ap = 0.0
        else:
            roc_auc = roc_auc_score(targets, predictions)  # if we use generated \ soft labels
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

        if all(targets == 0) or all(targets == 1):
            roc_auc = 0.0
            ap = 0.0
        else:
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
        if self.hparams.training_type == 'normal':
            train_dataset = MelanomaDataset(
                    mode="train",
                    config=self.hparams,
                    transform=get_training_trasnforms(self.hparams.training_transforms),
                    use_external=self.hparams.use_external
            )
        elif self.hparams.training_type == 'ad_learning':
            train_dataset = AdLearningMelanomaDataset(
                    mode="train",
                    config=self.hparams,
                    transform=get_training_trasnforms(self.hparams.training_transforms),
                    use_external=self.hparams.use_external
            )
        elif self.hparams.training_type == 'learning_from_generated_data':
            train_dataset = MelanomaDatasetGeneratedData(
                    mode="train",
                    config=self.hparams,
                    transform=get_training_trasnforms(self.hparams.training_transforms),
            )
        else:
            raise NotImplementedError
        if not self.hparams.use_weightened:
            sampler = RandomSampler(train_dataset)
        else:
            sample_count = train_dataset.target_counts
            weight = 1 / torch.Tensor(sample_count)
            targets = train_dataset.targets
            samples_weights = weight[targets]
            sampler = WeightedRandomSampler(
                samples_weights,
                num_samples=len(samples_weights),
                replacement=True)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.hparams.training_type != 'ad_learning':
            val_dataset = MelanomaDataset(
                    mode="val",
                    config=self.hparams,
                    transform=get_valid_transforms()
            )
        else:
            val_dataset = AdLearningMelanomaDataset(
                    mode="val",
                    config=self.hparams,
                    transform=get_valid_transforms(),
                    use_external=self.hparams.use_external
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
                weight_decay=self.hparams.weight_decay)
        elif "adamw" == self.hparams.optimizer:
            return torch.optim.AdamW(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay)
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
                weight_decay=self.hparams.weight_decay
            )
        elif "ralamb" == self.hparams.optimizer:
            return Ralamb(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif "qhadamw" == self.hparams.optimizer:
            return QHAdamW(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif "lamb" == self.hparams.optimizer:
            return Lamb(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif "radam+lookahead" == self.hparams.optimizer:
            optimizer = RAdam(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay
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

    def load_weights_from_checkpoint(self, checkpoint: str) -> None:
        """ Function that loads the weights from a given checkpoint file.
        Note:
            If the checkpoint model architecture is different then `self`, only
            the common parts will be loaded.

        :param checkpoint: Path to the checkpoint containing the weights to be loaded.
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)
