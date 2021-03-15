import copy
import itertools
import numpy as np
from parmap import parmap
from datasets import get_dataset
from config import set_print_options
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import warnings
import time
from config import PROJECT_PATH
warnings.filterwarnings("ignore")
set_print_options(pd)


def get_folds(df, x_cols, n_splits, stratified, shuffle):
    if shuffle:
        time_ms = round(time.time() * 1000) % 1000
        df = df.sample(frac=1, random_state=time_ms).reset_index(drop=True)

    x, y = df[x_cols].values, df['class_idx'].values
    df['k_fold_idx'] = 0

    if stratified:
        kf = StratifiedKFold(n_splits=n_splits)
    else:
        kf = KFold(n_splits=n_splits)

    for k, indexes in enumerate(kf.split(x, y)):
        df.loc[indexes[1], 'k_fold_idx'] = k

    return df


def predict_on_folds(df, x_cols, n_splits, k_neighbours, knn_voting_type, knn_metric):
    df['class_idx_pred'] = -1

    for fold_idx in range(n_splits):
        df_tr = df.loc[df['k_fold_idx'] != fold_idx]
        df_tst = df.loc[df['k_fold_idx'] == fold_idx]

        x_tr, y_tr = df_tr[x_cols].values, df_tr['class_idx'].values
        x_tst, y_tst = df_tst[x_cols].values, df_tst['class_idx'].values

        if knn_voting_type == 'custom':
            def voting_func(x):
                return np.sqrt(x)

            knn_voting_type = voting_func

        neigh = KNeighborsClassifier(
            n_neighbors=k_neighbours,
            weights=knn_voting_type,
            metric=knn_metric
        )

        neigh.fit(x_tr, y_tr)
        y_pred = neigh.predict(x_tst)

        df.loc[df['k_fold_idx'] == fold_idx, 'class_idx_pred'] = y_pred

    return df


def get_classification_metrics(df):
    y_true, y_pred = df['class_idx'].values, df['class_idx_pred'].values
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(sorted(set(y_true.tolist()))))
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return conf_matrix, precision, recall, f_score


def inference(args, datasets):
    df, class_mapping, x_cols = copy.deepcopy(datasets[args['dataset_type']])

    df = get_folds(
        df=df,
        x_cols=x_cols,
        n_splits=args['k_folds_splits'],
        stratified=args['k_folds_stratified'],
        shuffle=args['dataset_shuffle']
    )

    df = predict_on_folds(
        df=df,
        x_cols=x_cols,
        n_splits=args['k_folds_splits'],
        k_neighbours=args['knn_k_neighbours'],
        knn_voting_type=args['knn_voting_type'],
        knn_metric=args['knn_metric']
    )

    conf_matrix, precision, recall, f_score = get_classification_metrics(df)

    return f_score


def main():
    args_list = {
        'dataset_type': ['iris', 'glass', 'wine'],
        'dataset_shuffle': [False, True],
        'k_folds_splits': [2, 5, 10],
        'k_folds_stratified': [False, True],
        'knn_k_neighbours': list(range(1, 15 + 1)),
        'knn_voting_type': ['uniform', 'distance', 'custom'],
        'knn_metric': ['euclidean', 'manhattan', 'chebyshev'],
    }

    args_values = list(itertools.product(*args_list.values()))
    args_list = [dict(zip(args_list.keys(), x)) for x in args_values]

    datasets = {
        x: get_dataset(x, normalize_x=True)
        for x in ['iris', 'glass', 'wine']
    }

    results = []

    for i in range(10):
        inference_args_list = [(x, datasets) for x in args_list]

        f_scores = parmap.starmap(
            function=inference,
            iterables=inference_args_list,
            pm_pbar=True,
            pm_processes=6
        )

        inference_args_list = [x[0] for x in inference_args_list]

        for inference_args, f_score in zip(inference_args_list, f_scores):
            inference_args['f_score'] = f_score

        results += inference_args_list

    results = pd.DataFrame(results)
    results.to_csv('{}/lab_1/results.csv'.format(PROJECT_PATH), index=False)


if __name__ == '__main__':
    main()
