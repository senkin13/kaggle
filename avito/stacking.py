# -*- coding: utf-8 -*-
"""
Lv1 stacking
"""
from logging import (
    getLogger,
    Formatter,
    StreamHandler,
    FileHandler,
    INFO)
from pathlib import Path

from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pandas as pd
import click
import numpy as np


MODEL_NAME = __file__.split('.')[0]
LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'

logger = getLogger('avito')

OOF_FILE_LIST = [
    'data/working/models/dpcnn_seed_777_emb_en_splits_5_score_0.21860_0.00037.npy',
    'data/working/models/dpcnn_seed_777_emb_ft_splits_5_score_0.21781_0.00054.npy',
    'data/working/models/dpcnn_seed_777_emb_w2v_splits_5_score_0.21721_0.00060.npy',
    'data/working/models/grucnn_seed_777_emb_en_splits_5_score_0.21876_0.00047.npy',
    'data/working/models/grucnn_seed_777_emb_ft_splits_5_score_0.21696_0.00032.npy',
    'data/working/models/grucnn_seed_777_emb_w2v_splits_5_score_0.21672_0.00040.npy',
    'data/working/models/lightgbm_gbdt_seed_777_emb_ft_splits_5_score_0.21545_0.00037.npy',

    'data/working/models/v87_lgbm_oof_pred.npy',
    'data/working/models/v91_cat_oof_pred.npy',
    'data/working/models/v93_cat_oof_pred.npy',
    'data/working/models/v90_lgbm_oof_pred.npy',
    'data/working/models/v94_rnn_oof_pred.npy',
    'data/working/models/v101_lgbm_oof_pred.npy',
    'data/working/models/v102_rnn_wvemb_oof_pred.npy',
    'data/working/models/v108_lgbm_oof_pred.npy',

    # TODO: Ridge, xgboost, RGF
]


def init():
    if not Path("data/output").exists():
        Path("data/output").mkdir(parents=True)
    if not Path("data/working/models").exists():
        Path("data/working/models").mkdir(parents=True)
    if not Path("data/working/log").exists():
        Path("data/working/log").mkdir(parents=True)


@click.group()
def cli():
    # Add handlers
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter(LOGFORMAT))
    fh_handler = FileHandler(
        "./data/working/log/{}.log".format(MODEL_NAME))
    fh_handler.setFormatter(Formatter(LOGFORMAT))

    logger.setLevel(INFO)
    logger.addHandler(handler)
    logger.addHandler(fh_handler)


@cli.command()
def check_cvscore():
    scores = []
    y = np.load('data/working/feat/y.npy')
    cvfold = pd.read_csv(
        'data/working/team_5fold.csv').fold_id.values

    for fold_id in range(5):
        y_pred = np.load(
            f'data/working/models/{MODEL_NAME}_trn_fold{fold_id}.npy')
        y_valid = y[np.array(cvfold == fold_id)]
        fold_score = np.sqrt(mean_squared_error(
            y_valid, y_pred))
        scores.append(fold_score)
        print(f"- Fold{fold_id} score: {fold_score:.6f}")
    print(f"CV score: {np.mean(scores):.6f}")


@cli.command()
def save_oof_prediction_file():
    __save_oof_prediction_file()


def __save_oof_prediction_file():
    df_fold = pd.read_csv('data/working/team_5fold.csv')
    cvfold = df_fold.fold_id.values
    df_fold['pred'] = np.nan

    for fold_id in range(5):
        y_pred = np.load(
            f'data/working/models/{MODEL_NAME}_trn_fold{fold_id}.npy')
        df_fold.loc[np.array(cvfold == fold_id), 'pred'] = y_pred.ravel()

    df_cv = pd.read_csv(
        f'data/output/{MODEL_NAME}_foldmean.csv.gz')
    np.save(
        f'data/working/models/{MODEL_NAME}_oof_pred.npy',
        np.concatenate([
            df_fold.pred.values,
            df_cv.deal_probability.values,
        ]))


@cli.command()
def cvfold_avg_pred_submission():
    df_sub = pd.read_csv('data/sample_submission.csv')
    df_sub['deal_probability'] = 0.0

    for fold_id in range(5):
        y_pred = np.load(
            f'data/working/models/{MODEL_NAME}_tst_fold{fold_id}.npy')
        print(y_pred.shape, len(df_sub))
        df_sub['deal_probability'] += y_pred

    df_sub['deal_probability'] = df_sub.deal_probability / 5.0
    df_sub['deal_probability'] = np.clip(
        df_sub.deal_probability.values, 0.0, 1.0)
    df_sub[[
        'item_id',
        'deal_probability',
    ]].to_csv(f'data/output/{MODEL_NAME}_foldmean.csv.gz',
              compression='gzip',
              index=False)


@cli.command()
def validate():
    for fold_id in range(5):
        run_cvfold(fold_id)


def run_cvfold(fold_id):
    cols = OOF_FILE_LIST
    y = np.load('data/working/feat/y.npy')
    cvfold = pd.read_csv(
        'data/working/team_5fold.csv').fold_id.values
    df = pd.DataFrame({'fold_id': cvfold})

    names = []
    preds = []
    for i, col in enumerate(cols):
        name = f'model_{i}'
        pred = np.load(col).ravel()
        print(name, pred.shape)
        preds.append(pred.reshape((-1, 1)))
        names.append(name)

    X_ = np.hstack(preds)
    X = X_[:cvfold.shape[0]]
    X_test = X_[cvfold.shape[0]:]
    print(X.shape, X_test.shape)

    X_train = X[np.array(cvfold != fold_id)]
    X_valid = X[np.array(cvfold == fold_id)]
    y_train = y[np.array(cvfold != fold_id)]
    y_valid = y[np.array(cvfold == fold_id)]

    lgtrain = lgb.Dataset(
        X_train, y_train,
        feature_name=names)
    lgvalid = lgb.Dataset(
        X_valid, y_valid,
        feature_name=names)

    lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 3,
        'bagging_fraction': 0.8,
        'learning_rate': 0.01,
        'verbose': -1,
    }

    bst = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=20000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train', 'valid'],
        early_stopping_rounds=100,
        verbose_eval=10,
    )

    logger.info(f"Best iteration: {bst.best_iteration}")
    y_pred = bst.predict(X_valid, num_iteration=bst.best_iteration)
    logger.info('RMSE: {:.6f}'.format(
        np.sqrt(mean_squared_error(
            y_valid, y_pred))))
    np.save(f'data/working/models/{MODEL_NAME}_trn_fold{fold_id}.npy', y_pred)

    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
    np.save(f'data/working/models/{MODEL_NAME}_tst_fold{fold_id}.npy', y_pred)


if __name__ == '__main__':
    init()
    cli()
