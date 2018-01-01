from datetime import date, timedelta
import calendar as ca
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn import preprocessing
import gc

def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            df[c] = lbl.fit_transform(df[c])
            print(c)
    return df

print('Loading Data')
df_train = pd.read_csv(
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
    parse_dates=["date"]
)

df_items = pd.read_csv(
    "input/items.csv",
)

df_stores = pd.read_csv(
    "input/stores.csv"
)

print('Creating Sales Dataset')
df_sales_train = df_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_sales_train.columns = df_sales_train.columns.get_level_values(1)

print('Creating Promotion Dataset')
df_promo_train = df_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
df_promo_train.columns = df_promo_train.columns.get_level_values(1)
df_promo_test = df_test.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
df_promo_test.columns = df_promo_test.columns.get_level_values(1)
df_promo_test = df_promo_test.reindex(df_promo_train.index).fillna(False)
df_promo = pd.concat([df_promo_train, df_promo_test], axis=1)

items = df_items.set_index("item_nbr").reindex(df_sales_train.index.get_level_values(1))

del df_promo_test, df_promo_train

print('Creating items Dataset')
items_enc = df_lbl_enc(df_items)

# merge train,test with items
df_items_train = pd.merge(df_train, items_enc, how='left', on=['item_nbr'])
df_items_test = pd.merge(df_test, items_enc, how='left', on=['item_nbr'])

# family of item
df_family_train = df_items_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["family"]].unstack(
        level=-1).fillna(999)
df_family_train.columns = df_family_train.columns.get_level_values(1)

df_family_test = df_items_test.set_index(
    ['store_nbr', 'item_nbr', 'date'])[["family"]].unstack(
        level=-1).fillna(999)
df_family_test.columns = df_family_test.columns.get_level_values(1)
df_family_test = df_family_test.reindex(df_family_train.index).fillna(False)
df_family = pd.concat([df_family_train, df_family_test], axis=1)
del df_family_train, df_family_test

# class of item
df_class_train = df_items_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["class"]].unstack(
        level=-1).fillna(999)
df_class_train.columns = df_class_train.columns.get_level_values(1)

df_class_test = df_items_test.set_index(
    ['store_nbr', 'item_nbr', 'date'])[["class"]].unstack(
        level=-1).fillna(999)
df_class_test.columns = df_class_test.columns.get_level_values(1)
df_class_test = df_class_test.reindex(df_class_train.index).fillna(False)
df_class = pd.concat([df_class_train, df_class_test], axis=1)
del df_class_train, df_class_test

# perishable of item
df_perishable_train = df_items_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["perishable"]].unstack(
        level=-1).fillna(999)
df_perishable_train.columns = df_perishable_train.columns.get_level_values(1)

df_perishable_test = df_items_test.set_index(
    ['store_nbr', 'item_nbr', 'date'])[["perishable"]].unstack(
        level=-1).fillna(999)
df_perishable_test.columns = df_perishable_test.columns.get_level_values(1)
df_perishable_test = df_perishable_test.reindex(df_perishable_train.index).fillna(False)
df_perishable = pd.concat([df_perishable_train, df_perishable_test], axis=1)
del df_perishable_train, df_perishable_test

# del used df
del df_items_train,df_items_test,df_items,items_enc

print('Creating stores Dataset')
stores_enc = df_lbl_enc(df_stores)

# merge train,test with stores
df_stores_train = pd.merge(df_train, stores_enc, how='left', on=['store_nbr'])
df_stores_test = pd.merge(df_test, stores_enc, how='left', on=['store_nbr'])

# state of store
df_state_train = df_stores_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["state"]].unstack(
        level=-1).fillna(999)
df_state_train.columns = df_state_train.columns.get_level_values(1)

df_state_test = df_stores_test.set_index(
    ['store_nbr', 'item_nbr', 'date'])[["state"]].unstack(
        level=-1).fillna(999)
df_state_test.columns = df_state_test.columns.get_level_values(1)
df_state_test = df_state_test.reindex(df_state_train.index).fillna(False)
df_state = pd.concat([df_state_train, df_state_test], axis=1)
del df_state_train, df_state_test

# city of store
df_city_train = df_stores_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["city"]].unstack(
        level=-1).fillna(999)
df_city_train.columns = df_city_train.columns.get_level_values(1)

df_city_test = df_stores_test.set_index(
    ['store_nbr', 'item_nbr', 'date'])[["city"]].unstack(
        level=-1).fillna(999)
df_city_test.columns = df_city_test.columns.get_level_values(1)
df_city_test = df_city_test.reindex(df_city_train.index).fillna(False)
df_city = pd.concat([df_city_train, df_city_test], axis=1)
del df_city_train, df_city_test

# type of store
df_type_train = df_stores_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["type"]].unstack(
        level=-1).fillna(999)
df_type_train.columns = df_type_train.columns.get_level_values(1)

df_type_test = df_stores_test.set_index(
    ['store_nbr', 'item_nbr', 'date'])[["type"]].unstack(
        level=-1).fillna(999)
df_type_test.columns = df_type_test.columns.get_level_values(1)
df_type_test = df_type_test.reindex(df_type_train.index).fillna(False)
df_type = pd.concat([df_type_train, df_type_test], axis=1)
del df_type_train, df_type_test

# cluster of store
df_cluster_train = df_stores_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["cluster"]].unstack(
        level=-1).fillna(999)
df_cluster_train.columns = df_cluster_train.columns.get_level_values(1)

df_cluster_test = df_stores_test.set_index(
    ['store_nbr', 'item_nbr', 'date'])[["cluster"]].unstack(
        level=-1).fillna(999)
df_cluster_test.columns = df_cluster_test.columns.get_level_values(1)
df_cluster_test = df_cluster_test.reindex(df_cluster_train.index).fillna(False)
df_cluster = pd.concat([df_cluster_train, df_cluster_test], axis=1)
del df_cluster_train, df_cluster_test

# del used df
del df_stores_train,df_stores_test,stores_enc,df_stores

# del df_train
del df_train
gc.collect()

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "day_1_2017": get_timespan(df_sales_train, t2017, 1, 1).values.ravel(),
        "std_7_2017": get_timespan(df_sales_train, t2017, 7, 7).std(axis=1).values,
        "std_14_2017": get_timespan(df_sales_train, t2017, 14, 14).std(axis=1).values,
        "std_28_2017": get_timespan(df_sales_train, t2017, 28, 28).std(axis=1).values,
        "var_7_2017": get_timespan(df_sales_train, t2017, 7, 7).var(axis=1).values,
        "var_14_2017": get_timespan(df_sales_train, t2017, 14, 14).var(axis=1).values,
        "var_28_2017": get_timespan(df_sales_train, t2017, 28, 28).var(axis=1).values,
        "median_3_2017": get_timespan(df_sales_train, t2017, 3, 3).median(axis=1).values,
        "median_7_2017": get_timespan(df_sales_train, t2017, 7, 7).median(axis=1).values,
        "median_14_2017": get_timespan(df_sales_train, t2017, 14, 14).median(axis=1).values,
        "median_28_2017": get_timespan(df_sales_train, t2017, 28, 28).median(axis=1).values,
        "median_56_2017": get_timespan(df_sales_train, t2017, 56, 56).median(axis=1).values,
        "median_112_2017": get_timespan(df_sales_train, t2017, 112, 112).median(axis=1).values,
        "max_3_2017": get_timespan(df_sales_train, t2017, 3, 3).max(axis=1).values,
        "max_7_2017": get_timespan(df_sales_train, t2017, 7, 7).max(axis=1).values,            
        "mean_3_2017": get_timespan(df_sales_train, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_sales_train, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_sales_train, t2017, 14, 14).mean(axis=1).values,
        "mean_28_2017": get_timespan(df_sales_train, t2017, 28, 28).mean(axis=1).values,
        "mean_56_2017": get_timespan(df_sales_train, t2017, 56, 56).mean(axis=1).values,
        "mean_112_2017": get_timespan(df_sales_train, t2017, 112, 112).mean(axis=1).values,
        "promo_14_2017": get_timespan(df_promo, t2017, 14, 14).sum(axis=1).values,
        "promo_28_2017": get_timespan(df_promo, t2017, 28, 28).sum(axis=1).values,
        "promo_56_2017": get_timespan(df_promo, t2017, 56, 56).sum(axis=1).values,
        "promo_112_2017": get_timespan(df_promo, t2017, 112, 112).sum(axis=1).values
    })

    for i in range(16):
        X["promo_{}".format(i)] = df_promo[
            t2017 + timedelta(days=i)].values.astype(np.uint8)

    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_sales_train, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['median_4_dow{}_2017'.format(i)] = get_timespan(df_sales_train, t2017, 28-i, 4, freq='7D').median(axis=1).values        
        X['mean_8_dow{}_2017'.format(i)] = get_timespan(df_sales_train, t2017, 56-i, 8, freq='7D').mean(axis=1).values
        X['median_8_dow{}_2017'.format(i)] = get_timespan(df_sales_train, t2017, 56-i, 8, freq='7D').median(axis=1).values
        X['mean_16_dow{}_2017'.format(i)] = get_timespan(df_sales_train, t2017, 112-i, 16, freq='7D').mean(axis=1).values
        X['median_16_dow{}_2017'.format(i)] = get_timespan(df_sales_train, t2017, 112-i, 16, freq='7D').median(axis=1).values
        X['mean_32_dow{}_2017'.format(i)] = get_timespan(df_sales_train, t2017, 224-i, 32, freq='7D').mean(axis=1).values
        X['median_32_dow{}_2017'.format(i)] = get_timespan(df_sales_train, t2017, 224-i, 32, freq='7D').median(axis=1).values        

    X["family"] = df_family[t2017].values.astype(np.uint8)
    X["class"] = df_class[t2017].values.astype(np.uint8)
    X["perishable"] = df_perishable[t2017].values.astype(np.uint8)  
    X["state"] = df_state[t2017].values.astype(np.uint8)
    X["city"] = df_city[t2017].values.astype(np.uint8)
    X["type"] = df_type[t2017].values.astype(np.uint8)
    X["cluster"] = df_cluster[t2017].values.astype(np.uint8)
    
    if is_train:
        y = df_sales_train[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

print("Preparing dataset...")
t2017 = date(2017, 5, 31)
X_l, y_l = [], []
for i in range(6):
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
    'num_leaves': 33,
    'objective': 'regression',
    'min_data_in_leaf': 250,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'num_threads': 8
}

MAX_ROUNDS = 3000
val_pred = []
test_pred = []
cate_vars = [97,98,99,100,101,102,103]
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * 6) * 0.25 + 1
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
submission.to_csv('lgb7.csv', float_format='%.4f', index=None)
