from argparse import ArgumentParser, Namespace
import pandas as pd
import torch
import torch.utils
import torch.nn as nn
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.datasets.melanoma_dataset import MelanomaDatasetTest
from src.pl_module import MelanomaModel
from src.transforms.albu import get_valid_transforms
from typing import List
import ttach
from itertools import cycle
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
SEED = 111
seed_everything(111)


def get_test_dataloder(hparams: Namespace) -> DataLoader:
    test_dataset = MelanomaDatasetTest(config=hparams, transform=get_valid_transforms())

    return DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )


def load_model(model_name: str, model_type: str, weights: str):
    model = MelanomaModel.net_mapping(model_name, model_type)
    if weights.endswith('.pth'):
        model.load_state_dict(
            torch.load(weights)
        )
    elif weights.endswith('.ckpt'):
        checkpoint = torch.load(weights, map_location=lambda storage, loc: storage)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {k[4:]: v for k, v in pretrained_dict.items() if k[4:] in model_dict}  # net.
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    else:
        raise NotImplementedError
    model.eval()
    model.cuda()
    print("Loaded model {} from checkpoint {}".format(model_name, weights))
    return model


def run_predictions(
    models: List[torch.nn.Module],
    loader: DataLoader,
    precision: int = 16
):
    preds = list()
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            pred = [nn.Sigmoid()(model(batch['features'].cuda())) for model in models]
            pred = torch.stack(pred)
            pred = pred.mean(axis=0)
            pred = pred[:, 0].cpu().numpy()
            preds.extend(pred)
    return preds


def main(hparams: Namespace):
    loader = get_test_dataloder(hparams)
    model_names = hparams.model_name.split(',')
    model_types = hparams.model_type.split(',')
    weights = hparams.weights.split(',')
    if len(model_types) != len(weights):
        print(f'Padding model_type to match weights length, {len(model_types)} and {len(weights)}')
        model_types = cycle(model_types)
    if len(model_names) != len(weights):
        print(f'Padding model_name to match weights length, {len(model_names)} and {len(weights)}')
        model_names = cycle(model_names)
    models = [load_model(model_name, model_type, weight)
              for model_name, model_type, weight in zip(model_names, model_types, weights)]
    if hparams.tta == 'd4':
        models = [ttach.ClassificationTTAWrapper(model, ttach.aliases.d4_transform()) for model in models]
    if hparams.tta == 'ten_crop':
        models = [ttach.ClassificationTTAWrapper(model, ttach.aliases.ten_crop_transform()) for model in models]
    if hparams.tta == 'five_crop':
        models = [ttach.ClassificationTTAWrapper(model, ttach.aliases.five_crop_transform()) for model in models]
    predictions = run_predictions(models, loader)
    sample_submission = pd.read_csv(hparams.sample_submission_path)
    sample_submission.loc[:, 'target'] = predictions
    sample_submission.to_csv(f'./predictions/{hparams.submission_name}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_path", default="./data/"
    )
    parser.add_argument("--test_image_folder", type=str)
    parser.add_argument("--submission_name", default="test_sub", type=str)
    parser.add_argument("--model_type", default="SingleHeadMax", type=str)
    parser.add_argument("--model_name", default="resnet34", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--tta", default='none', type=str)
    parser.add_argument("--sample_submission_path", default='data/sample_submission.csv', type=str)
    args = parser.parse_args()
    main(args)
