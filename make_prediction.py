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
    model.load_state_dict(
        torch.load(weights)
    )
    model.eval()
    model.cuda()
    print("Loaded model {} from checkpoint {}".format(model_name, weights))
    return model


def run_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    precision: int = 16
):
    preds = list()
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            y_hat = model(batch['features'].cuda())
            pred = nn.Sigmoid()(y_hat).cpu().numpy()
            pred = pred[:, 1]
            preds.extend(pred)
    return preds


def main(hparams: Namespace):
    loader = get_test_dataloder(hparams)
    model = load_model(hparams.model_name, hparams.model_type,  hparams.weights)
    predictions = run_predictions(model, loader)
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
    parser.add_argument("--weights", default="resnet34", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--sample_submission_path", default='data/sample_submission.csv', type=str)
    args = parser.parse_args()
    main(args)
