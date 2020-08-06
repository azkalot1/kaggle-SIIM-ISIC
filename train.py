import datetime
from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateLogger, EarlyStopping)
from src.pl_module import MelanomaModel
import os
import sys

SEED = 12
seed_everything(12)


def main(hparams: Namespace):
    now = datetime.datetime.now().strftime("%d.%H")
    if hparams.experiment_name is None:
        experiment_name = f"{now}_{hparams.model_type}_{hparams.model_name}_{hparams.criterion}_{hparams.optimizer}_{hparams.training_transforms}_fold_{hparams.fold}"
    else:
        experiment_name = f"{now}_{hparams.experiment_name}"
    model = MelanomaModel(hparams=hparams)

    if hparams.load_weights is not None:
        print(f'Restoring checkpoint {hparams.load_weights}')
        model.load_weights_from_checkpoint(hparams.load_weights)

    pl_loggers = [
        loggers.TensorBoardLogger(
            f"logs/",
            name=experiment_name),
        loggers.neptune.NeptuneLogger(
            api_key=os.getenv('NEPTUNE_API_TOKEN'),
            experiment_name=experiment_name,
            params=vars(hparams),
            project_name='azkalot1/kaggle-isic2020'
            )
    ]
    callbacks = [LearningRateLogger()]
    checkpoint_callback = ModelCheckpoint(
        filepath=f"weights/{experiment_name}_" + "best_{val_loss:.4f}_{val_roc_auc:.4f}",
        monitor='val_loss', save_top_k=5, mode='min', save_last=True)
    early_stop_callback = EarlyStopping(
        monitor='val_loss', patience=15, mode='min', verbose=True)

    # a weird way to add arguments to Trainer constructor, but we'll take it
    hparams.__dict__['logger'] = pl_loggers
    hparams.__dict__['callbacks'] = callbacks
    hparams.__dict__['checkpoint_callback'] = checkpoint_callback
    hparams.__dict__['early_stop_callback'] = early_stop_callback

    trainer = Trainer.from_argparse_args(hparams)

    trainer.fit(model)

    # to make submission without lightning
    torch.save(model.net.state_dict(), f"weights/{experiment_name}.pth")


if __name__ == "__main__":
    # TODO: move configuration to *.yaml with Hydra
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--data_path", default="./data/")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--load_weights", default=None, type=str)
    parser.add_argument("--image_folder", default="./data/jpeg-melanoma-128x128/train")
    parser.add_argument("--test_image_folder", default="./data/jpeg-melanoma-128x128/test")
    parser.add_argument("--training_transforms", default="light")
    parser.add_argument("--use_mixup", default=False, type=bool)
    parser.add_argument("--make_submission", default=False, type=bool)
    parser.add_argument("--mixup_alpha", default=1.0, type=float)
    parser.add_argument("--use_weightened", default=False, type=bool)
    parser.add_argument("--generated_data_csv", type=str)
    parser.add_argument("--generated_data_image_folder", type=str)
    parser.add_argument("--profiler", default=False, type=bool)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    parser.add_argument("--auto_lr_find", default=False, type=bool)
    parser.add_argument("--use_external", default=False, type=bool)
    parser.add_argument("--external_image_folder", default=None)
    parser.add_argument("--training_type", default='normal')
    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--limit_train_batches", default=1.0, type=float)
    parser.add_argument("--limit_val_batches", default=1.0, type=float)
    # parser.add_argument("--distributed_backend", default="horovod", type=str)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--warmup_factor", default=1., type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--deterministic", default=True, type=bool)
    parser.add_argument("--benchmark", default=True, type=bool)

    parser.add_argument("--model_type", default="SingleHeadMax", type=str)
    parser.add_argument("--model_name", default="resnet34", type=str)
    parser.add_argument("--criterion", default="bce_with_logits", type=str)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--scheduler", default="plateau", type=str)

    parser.add_argument("--sgd_momentum", default=0.9, type=float)
    parser.add_argument("--sgd_wd", default=1e-4, type=float)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--gradient_clip_val", default=10, type=float)

    args = parser.parse_args()
    if args.use_external and not args.external_image_folder:
        print("No external image folder provided, but requisted to use external data")
        sys.exit(1)
    if args.training_type == "ad_learning" and not args.external_image_folder:
        print("Ad learning is requested, but no external data is provided")
        sys.exit(1)
    if args.training_type == "learning_from_generated_data" and not args.generated_data_csv:
        print("Learning from generated data is requested, but no generated_data_csv is provided")
        sys.exit(1)
    if args.training_type == "learning_from_generated_data" and not args.generated_data_image_folder:
        print("Learning from generated data is requested, but no generated_data_image_folder is provided")
        sys.exit(1)
    main(args)
