from sklearn.metrics import roc_auc_score
from .pl_module import MelanomaModel
from .datasets.melanoma_dataset import MelanomaDataset
from .transforms.albu import get_valid_transforms
import os
import re
import numpy as np
import torch
from argparse import Namespace
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm.auto import tqdm
import torch.nn as nn


def average_weights(state_dicts):
    # source https://gist.github.com/qubvel/70c3d5e4cddcde731408f478e12ef87b
    everage_dict = OrderedDict()
    for k in state_dicts[0].keys():
        everage_dict[k] = sum([state_dict[k] for state_dict in state_dicts]) / len(state_dicts)
    return everage_dict


def evaluate_model(model, loader):
    # source https://gist.github.com/qubvel/70c3d5e4cddcde731408f478e12ef87b
    model.eval()
    predicted_class = []
    gt_class = []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            predictions = model(batch['features'].cuda())
            predictions = nn.Sigmoid()(predictions)
            predicted_class.extend(predictions.cpu().numpy())
            gt_class.extend(batch['target'].cpu().numpy())
    gt_class = np.array(gt_class).astype(int)
    predicted_class = np.array(predicted_class)
    return(roc_auc_score(gt_class, predicted_class))


def load_weight(path):
    if path.endswith('.pth'):
        weights = torch.load(path)
    elif path.endswith('.ckpt'):
        weights = torch.load(path, map_location=lambda storage, loc: storage)
        weights = weights["state_dict"]
        weights = {k[4:]: v for k, v in weights.items()}  # net.
    else:
        raise NotImplementedError
    return weights


def generate_best_weights(weights_path, model, loader):
    all_weights = [load_weight(path) for path in weights_path]

    best_score = 0
    best_weights = []

    for w in all_weights:
        current_weights = best_weights + [w]
        average_dict = average_weights(current_weights)
        model.load_state_dict(average_dict)
        score = evaluate_model(model, loader)
        print(f'Score: {score}')
        if score > best_score:
            print(f'New best score {score}')
            best_score = score
            best_weights.append(w)
    average_dict = average_weights(best_weights)
    return(average_dict)


def get_save_averaged_best_weights(params: Namespace):
    model = MelanomaModel.net_mapping(params.model_name, params.model_type)
    model.cuda()
    model.eval()
    all_weights = os.listdir(params.weights_folder)
    experiment_weights = np.array(
        [x for x in all_weights if x.startswith(params.experiment_name) and x.endswith('.ckpt')]
        )
    losses = [float(re.search('best_val_loss=(.*)_val_roc_auc', x).group(1)) for x in experiment_weights]
    order_loss = np.argsort(losses)
    experiment_weights = experiment_weights[order_loss]
    experiment_weights = [os.path.join(params.weights_folder, x) for x in experiment_weights]
    dataset = MelanomaDataset('val', params, get_valid_transforms())
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    best_weights = generate_best_weights(experiment_weights, model, dataloader)
    model.load_state_dict(best_weights)
    score = evaluate_model(model, dataloader)
    print(f'Final best score {score}')
    torch.save(model.state_dict(), os.path.join(params.weights_folder,
                                                f'{params.experiment_name}_averaged_best_weights.pth'))
