# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def load_meta_features(names, fold_id=None):
    if fold_id is not None:
        # CV predictions
        cols = ['data/working/models/' + name for name in names]
        cols = [col + '_trn_fold{}.npy' for col in cols]

        y = np.load('data/working/feat/y.npy')
        cvfold = np.load('data/working/feat/cvfold.npy')
        df = pd.DataFrame({'fold_id': cvfold})
        for i, name in enumerate(names):
            df[name] = 0

        for i, name in enumerate(names):
            for fold_id_ in range(5):
                idx_fold = np.array(df.fold_id == fold_id_)
                df.loc[idx_fold, name] = (
                    np.load(cols[i].format(fold_id_)))

        X = df[names].as_matrix()
        X_train = X[np.array(cvfold != fold_id)]
        X_valid = X[np.array(cvfold == fold_id)]
        y_train = y[np.array(cvfold != fold_id)]
        y_valid = y[np.array(cvfold == fold_id)]
        return X_train, y_train, X_valid, y_valid
    else:
        # test predictions
        sz = len(pd.read_csv('data/sample_submission.csv'))
        X = np.zeros((sz, len(names)))
        print(X.shape)
        for idx, name in enumerate(names):
            pred = pd.read_csv(name)['deal_probability'].ravel()
            print(name, pred.shape)
            X[:, idx] = pred
        return X


def run_test():
    name_score_pairs = [
     
        ('data/output/v17.csv.gz',     0.2218),  # LGBM including GRU features
        ('data/output/v22.csv.gz',     0.2216),  # LGBM including GRU features
        ('data/output/v34_lgb.csv.gz', 0.2217),  # LGBM
        ('data/output/v25.csv.gz',     0.2210),  # Blend
        ('data/output/v31.csv.gz',     0.2205),  # Blend
        ('data/output/v29_ens.csv.gz', 0.2212),  # 1L-Stack-LGBM (LGBM only)
        ('data/output/v41_ens.csv.gz', 0.2211),  # 1L-Stack-LGBM including NN model

        # ('data/output/181_ensemble.csv', 0.2181),
        ('data/output/180_ensemble.csv', 0.2180),
        ('data/output/2206_lightgbm_gbdt_seed_777_emb_ft_splits_5_score_0.21611_0.00039.csv',
            0.2206),
        ('data/output/2194_grucnn_seed_777_emb_w2v_splits_5_score_0.21746_0.00044.csv.gz',
            0.2194),
        ('data/output/2177_oof_prediction_0.21346_seed_777.csv',
            0.2177),
        ('data/output/2222_oof_prediction_0.21381_seed_777.csv',
            0.2222),
        ('data/output/2228_ensemble.csv', 0.2228),
        ('data/output/2192_grucnn_seed_777_emb_w2v_splits_5_score_0.21672_0.00040.csv',
            0.2192),

        # Kernels ---------------------
        ('blend06_lb0.2216.csv',       0.2216),  # Kernel: Liner
        ('data/output/catsub.csv',     0.2281),  # Kernel_cat_boost_2281.py

        # kernel_xgb_text2vec_2239.R (xgb_tfidf0.218521.csv)
        ('data/output/xgb_tfidf0.218521.csv',   0.2239),

        # kernel RNN detailed explonation 0.2246
        # https://www.kaggle.com/shanth84/rnn-detailed-explanation-0-2246/output
        ('data/output/rnn_detailed_explo_2246.csv', 0.2246),

        # LightGBM with aggregated features V 2.0 0.2221
        # https://www.kaggle.com/him4318/lightgbm-with-aggregated-features-v-2-0?scriptVersionId=3835828
        ('data/output/lgbm_aggregated_v2.csv', 0.2221),

        # Boosting MLP 0.2297
        # (https://www.kaggle.com/peterhurford/boosting-mlp-lb-0-2297?scriptVersionId=3534385)
        ('data/output/boosting_mlp_2297.csv',  0.2297),
    ]

    names = [p[0] for p in name_score_pairs]
    eval_scores = [p[1] for p in name_score_pairs]
    zero_score = 0.3032
    X = load_meta_features(names)

    reg_lambda = 20.0
    first_term = np.linalg.inv(
        np.dot(X.T, X) +
        reg_lambda * np.identity(X.shape[1]))
    # first_term = np.linalg.inv(np.dot(X.T, X))
    second_term = np.zeros(first_term.shape[1])
    for idx in range(second_term.shape[0]):
        second_term[idx] = 0.5 * (
            np.power(zero_score, 2) * X.shape[0] +
            np.dot(X[:, idx], X[:, idx]) -
            np.power(eval_scores[idx], 2) * X.shape[0]
        )
    beta = np.dot(first_term, second_term)
    print(beta, beta.sum())

    # Weighted mean
    y_pred = np.zeros(X.shape[0])
    for idx in range(second_term.shape[0]):
        y_pred += beta[idx] * X[:, idx]

    df = pd.read_csv('data/sample_submission.csv')
    df.loc[:, 'deal_probability'] = np.clip(y_pred, 0.0, 1.0)
    df[[
        'item_id',
        'deal_probability',
    ]].to_csv('data/output/v59_reg.csv.gz',
              compression='gzip',
              index=False)


def validate():
    name_score_pairs = [
        ('v15', 0.2201),
        ('v17', 0.2188),
        ('v22', 0.2187),
        ('v23_nn', 0.2291),
        # ('output/fdsajfda.csv', 0.2123),
    ]
    names = [p[0] for p in name_score_pairs]
    eval_scores = [p[1] for p in name_score_pairs]
    zero_score = 0.2957

    X_train, y_train, X_valid, y_valid = load_meta_features(names, fold_id=0)
    for idx, name in enumerate(names):
        score = np.sqrt(mean_squared_error(
            X_valid[:, idx], y_valid))
        print("{:<8s}{:.6f}".format(name, score))

    print('-' * 40)
    fold_score = np.sqrt(mean_squared_error(
        np.zeros(y_valid.shape[0]), y_valid))
    print("{:<8s}{:.6f}".format('zero', fold_score))
    fold_score = np.sqrt(mean_squared_error(
        X_valid.mean(axis=1), y_valid))
    print("{:<8s}{:.6f}".format('mean', fold_score))

    # Quiz feedback blending
    # \hat{\beta} = (X^T X)^{-1} (X^Ty)
    reg_lambda = 1000.0
    first_term = np.linalg.inv(
        np.dot(X_valid.T, X_valid) +
        reg_lambda * np.identity(X_valid.shape[1]))
    second_term = np.zeros(first_term.shape[1])
    for idx in range(second_term.shape[0]):
        second_term[idx] = 0.5 * (
            np.power(zero_score, 2) * X_valid.shape[0] +
            np.dot(X_valid[:, idx], X_valid[:, idx]) -
            np.power(eval_scores[idx], 2) * X_valid.shape[0]
        )
    beta = np.dot(first_term, second_term)
    print(beta, beta.sum())

    # Weighted mean
    y_pred = np.zeros(y_valid.shape[0])
    for idx in range(second_term.shape[0]):
        y_pred += beta[idx] * X_valid[:, idx]
    fold_score = np.sqrt(mean_squared_error(y_pred, y_valid))
    print("{:<8s}{:.6f}".format('reg', fold_score))


def zero_submission():
    df = pd.read_csv('data/sample_submission.csv')
    df.loc[:, 'deal_probability'] = 0.0
    df[['item_id', 'deal_probability']].to_csv(
        'data/baseline.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    # validate()
    # zero_submission()
    run_test()
