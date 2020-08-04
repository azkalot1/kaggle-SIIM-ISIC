from src.saving_weights import get_save_averaged_best_weights
from argparse import ArgumentParser
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--data_path", default="./data/", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--model_type", default="SingleHeadMax", type=str)
    parser.add_argument("--model_name", default="resnet34", type=str)
    parser.add_argument("--weights_folder", default="weights/", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    args = parser.parse_args()
    get_save_averaged_best_weights(args)
