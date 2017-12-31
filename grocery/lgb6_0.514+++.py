from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

df_2017 = pd.read_csv(
    'input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 80735413)  # from 2016-05-31
)

df_test = pd.read_csv(
    "input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "input/items.csv",
).set_index("item_nbr")

#df_2017 = df_train.loc[df_train.date>=pd.datetime(2016,6,1)]
#del df_train

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
            
#        "std_3_2017": get_timespan(df_2017, t2017, 3, 3).std(axis=1).values,
        "std_7_2017": get_timespan(df_2017, t2017, 7, 7).std(axis=1).values,
        "std_14_2017": get_timespan(df_2017, t2017, 14, 14).std(axis=1).values,
        "std_28_2017": get_timespan(df_2017, t2017, 28, 28).std(axis=1).values,
#        "std_56_2017": get_timespan(df_2017, t2017, 56, 56).std(axis=1).values,
#        "std_112_2017": get_timespan(df_2017, t2017, 112, 112).std(axis=1).values,
#        "std_224_2017": get_timespan(df_2017, t2017, 224, 224).std(axis=1).values,
              
#        "var_3_2017": get_timespan(df_2017, t2017, 3, 3).var(axis=1).values,
        "var_7_2017": get_timespan(df_2017, t2017, 7, 7).var(axis=1).values,
        "var_14_2017": get_timespan(df_2017, t2017, 14, 14).var(axis=1).values,
        "var_28_2017": get_timespan(df_2017, t2017, 28, 28).var(axis=1).values,
#        "var_56_2017": get_timespan(df_2017, t2017, 56, 56).var(axis=1).values,
#        "var_112_2017": get_timespan(df_2017, t2017, 112, 112).var(axis=1).values,
#        "var_224_2017": get_timespan(df_2017, t2017, 224, 224).var(axis=1).values,
             
        "median_3_2017": get_timespan(df_2017, t2017, 3, 3).median(axis=1).values,
        "median_7_2017": get_timespan(df_2017, t2017, 7, 7).median(axis=1).values,
        "median_14_2017": get_timespan(df_2017, t2017, 14, 14).median(axis=1).values,
        "median_28_2017": get_timespan(df_2017, t2017, 28, 28).median(axis=1).values,
        "median_56_2017": get_timespan(df_2017, t2017, 56, 56).median(axis=1).values,
        "median_112_2017": get_timespan(df_2017, t2017, 112, 112).median(axis=1).values,
#        "median_224_2017": get_timespan(df_2017, t2017, 224, 224).median(axis=1).values,         
               
        "max_3_2017": get_timespan(df_2017, t2017, 3, 3).max(axis=1).values,
        "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
#        "max_14_2017": get_timespan(df_2017, t2017, 14, 14).max(axis=1).values,
#        "max_28_2017": get_timespan(df_2017, t2017, 28, 28).max(axis=1).values,
#        "max_56_2017": get_timespan(df_2017, t2017, 56, 56).max(axis=1).values,
#        "max_112_2017": get_timespan(df_2017, t2017, 112, 112).max(axis=1).values,
#        "max_224_2017": get_timespan(df_2017, t2017, 224, 224).max(axis=1).values, 
            
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_28_2017": get_timespan(df_2017, t2017, 28, 28).mean(axis=1).values,
        "mean_56_2017": get_timespan(df_2017, t2017, 56, 56).mean(axis=1).values,
        "mean_112_2017": get_timespan(df_2017, t2017, 112, 112).mean(axis=1).values,
#        "mean_224_2017": get_timespan(df_2017, t2017, 224, 224).mean(axis=1).values,
            
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_28_2017": get_timespan(promo_2017, t2017, 28, 28).sum(axis=1).values,
        "promo_56_2017": get_timespan(promo_2017, t2017, 56, 56).sum(axis=1).values,
        "promo_112_2017": get_timespan(promo_2017, t2017, 112, 112).sum(axis=1).values,            
        "promo_224_2017": get_timespan(promo_2017, t2017, 224, 224).sum(axis=1).values         
    })
    for i in range(7):
#        X['mean_2_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 14-i, 2, freq='7D').mean(axis=1).values
#        X['median_2_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 14-i, 2, freq='7D').median(axis=1).values
#        X['max_2_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 14-i, 2, freq='7D').max(axis=1).values
#        X['var_2_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 14-i, 2, freq='7D').var(axis=1).values
#        X['std_2_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 14-i, 2, freq='7D').std(axis=1).values
                   
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['median_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').median(axis=1).values
#        X['max_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').max(axis=1).values
#        X['var_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').var(axis=1).values
#        X['std_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').std(axis=1).values

        X['mean_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 8, freq='7D').mean(axis=1).values
        X['median_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 8, freq='7D').median(axis=1).values
#        X['max_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 8, freq='7D').max(axis=1).values
#        X['var_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 8, freq='7D').var(axis=1).values
#        X['std_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 8, freq='7D').std(axis=1).values
  
        X['mean_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 16, freq='7D').mean(axis=1).values
        X['median_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 16, freq='7D').median(axis=1).values
#        X['max_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 16, freq='7D').max(axis=1).values
#        X['var_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 16, freq='7D').var(axis=1).values
#        X['std_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 16, freq='7D').std(axis=1).values
  
        X['mean_32_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 224-i, 32, freq='7D').mean(axis=1).values
        X['median_32_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 224-i, 32, freq='7D').median(axis=1).values
#        X['max_32_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 224-i, 32, freq='7D').max(axis=1).values
#        X['var_32_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 224-i, 32, freq='7D').var(axis=1).values
#        X['std_32_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 224-i, 32, freq='7D').std(axis=1).values
        
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X
  
print("Preparing dataset...")
t2017 = date(2017, 3, 8)
X_l, y_l = [], []
for i in range(17):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

print("Training and predicting models...")
params = {
    'num_leaves': 64,
    'objective': 'regression',
    'min_data_in_leaf': 250,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'device': 'gpu',
    'num_threads': 8
}

MAX_ROUNDS = 3000
val_pred = []
test_pred = []
cate_vars = []
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * 17) * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=items["perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100
    )
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
    
n_public = 5 # Number of days in public test set
weights=pd.concat([items["perishable"]]) * 0.25 + 1
print("Unweighted validation mse: ", mean_squared_error(
    y_val, np.array(val_pred).transpose()) )
print("Full validation mse:       ", mean_squared_error(
    y_val, np.array(val_pred).transpose(), sample_weight=weights) )
print("'Public' validation mse:   ", mean_squared_error(
    y_val[:,:n_public], np.array(val_pred).transpose()[:,:n_public], sample_weight=weights) )
print("'Private' validation mse:  ", mean_squared_error(
    y_val[:,n_public:], np.array(val_pred).transpose()[:,n_public:], sample_weight=weights) )

print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('lgb6.csv', float_format='%.4f', index=None)
