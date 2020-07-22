import pandas as pd
from sklearn.model_selection import GroupKFold
import os
from argparse import ArgumentParser


def split_save_folds(csv_path, external_csv_path, n_splits, group_column, data_path):
    group_kfold = GroupKFold(n_splits=n_splits)
    data = pd.read_csv(csv_path)
    assert group_column in data.columns, 'Column group not found!'
    X = data.image_name
    y = data.target
    group = data[group_column]
    for fold_idx, (train_index, test_index) in enumerate(group_kfold.split(X, y, group)):
        data_train, data_val = data.iloc[train_index, :], data.iloc[test_index, :]
        data_train.to_csv(f"{data_path}/train_{fold_idx}.csv", index=False)
        data_val.to_csv(f"{data_path}/val_{fold_idx}.csv", index=False)
        # dummy check
        assert data_train[group_column].values[0] not in data_val[group_column]

    external_data = pd.read_csv(external_csv_path)
    assert group_column in external_data.columns, 'Column group not found!'
    X = external_data.image_name
    y = external_data.target
    group = external_data[group_column]
    for fold_idx, (train_index, test_index) in enumerate(group_kfold.split(X, y, group)):
        external_data_train, external_data_val = external_data.iloc[train_index, :], external_data.iloc[test_index, :]
        external_data_train.to_csv(f"{data_path}/external_train_{fold_idx}.csv", index=False)
        external_data_val.to_csv(f"{data_path}/external_val_{fold_idx}.csv", index=False)
        # dummy check
        assert external_data_train[group_column].values[0] not in external_data_val[group_column]


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--n_folds", default=5)
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--external_train_csv", default="data/external_train.csv")
    parser.add_argument("--data_path", default="data/")
    parser.add_argument("--group_column", default="tfrecord")
    args = parser.parse_args()

    print(f'Will split data into {args.n_folds} folds')
    assert not os.path.exists(f'{args.data_path}/train_0.csv'), 'Fold split already exists!'
    split_save_folds(args.train_csv, args.external_train_csv, args.n_folds, args.group_column, args.data_path)
