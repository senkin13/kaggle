import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import pickle

%%time
X = pickle.load(open('/data/X.pkl','rb'))

##model1 full data
tra = X[X['is_attributed'].notnull()]
##model2 >8th data
tra = X[X['is_attributed'].notnull()][(X.click_time>='2017-11-08 00:00:00')]


gc.collect()
tes = X[(X.click_time >= '2017-11-10 12:00:00') & (X.click_time<='2017-11-10 23:00:00')]
del X
gc.collect()

X_train = tra.drop(['ip','is_attributed','click_time','day'], axis=1)
y_train = tra.is_attributed
del tra
gc.collect()

all_features = list(X_train)
gc.collect()

cat_features=['app','device','os','channel','hour']

num_features = []
for c in all_features:
    if c not in cat_features:
        num_features.append(c)

## random split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=2014,stratify=y_train) 
gc.collect()        

MAX_ROUNDS = 3000

dtrain = lgb.Dataset(
       X_train, label=y_train, feature_name=all_features, categorical_feature=cat_features)
dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, feature_name=all_features, categorical_feature=cat_features)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 255,  
    'max_depth': 8,  
    'min_child_samples': 100,  
    'max_bin': 512,  
    'subsample': 0.8,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.8,  
    'min_child_weight': 0,  
    'subsample_for_bin': 200000,  
    'min_split_gain': 0,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
    'verbose': 1,
    'scale_pos_weight':99.7
    }


bst = lgb.train(
        params, dtrain,  num_boost_round=MAX_ROUNDS,
        #valid_sets=[dtrain, dval], early_stopping_rounds=40, verbose_eval=10)
        valid_sets=[dval], early_stopping_rounds=20, verbose_eval=10)
print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
)))

import datetime
X_test = tes.drop(['ip','is_attributed','click_time','day'], axis=1)
test_sup = tes[['ip', 'app', 'device', 'os', 'channel', 'click_time']]
del tes
gc.collect()

test_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
test_sup['is_attributed'] = test_pred
test_sup['click_time'] = pd.to_datetime(test_sup['click_time'])
test_sup['click_time'] = test_sup['click_time'] + datetime.timedelta(hours=-8)

join_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
all_cols = join_cols + ['is_attributed']

test = pd.read_csv('../input/test.csv')
test['click_time'] = pd.to_datetime(test['click_time'])
test = test.merge(test_sup[all_cols], how='left', on=join_cols)
test = test.drop_duplicates(subset=['click_id'])

test[['click_id', 'is_attributed']].to_csv('../models/lgbm.csv', index=False, float_format='%.9f')


## kfold
X_train = X_train.as_matrix()
y_train = y_train.as_matrix()
X_test = tes.drop(['ip','is_attributed','click_time','day'], axis=1)

from sklearn.model_selection import StratifiedKFold
NFOLDS = 10
MAX_ROUNDS = 2000
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=123)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 255,  
    'max_depth': 8,  
    'min_child_samples': 100,  
    'max_bin': 512,  
    'subsample': 0.8,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.8,  
    'min_child_weight': 0,  
    'subsample_for_bin': 200000,  
    'min_split_gain': 0,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
    'verbose': 1,
    'scale_pos_weight':99.7
    }

for i,(train_index,val_index) in enumerate(kfold.split(X_train,y_train)):
    print("Running fold {} / {}".format(i + 1, NFOLDS))
    print("Train Index:",train_index,",Val Index:",val_index)
    X_tra,X_val,y_tra,y_val = X_train[train_index, :], X_train[val_index, :], y_train[train_index], y_train[val_index]
    if i >=0:

        dtrain = lgb.Dataset(
            X_tra, label=y_tra, feature_name=all_features, categorical_feature=cat_features)
        dval = lgb.Dataset(
            X_val, label=y_val, reference=dtrain, feature_name=all_features, categorical_feature=cat_features)    
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dval], early_stopping_rounds=50, verbose_eval=10)

        del dtrain,dval
        del X_tra,y_tra,y_val,X_val
        gc.collect()

        test_pred = bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
        test_pred.dump('../models/kfold_senkin' + str(i) + '.pkl')
        del test_pred
        gc.collect()
                
#test_pred /= NFOLDS
print ("Kfold Done")
gc.collect()


test_sup = tes[['ip', 'app', 'device', 'os', 'channel', 'click_time']]

kfold0 = pickle.load(open('../models/kfold_senkin0.pkl','rb'))
kfold1 = pickle.load(open('../models/kfold_senkin1.pkl','rb'))
kfold2 = pickle.load(open('../models/kfold_senkin2.pkl','rb'))
kfold3 = pickle.load(open('../models/kfold_senkin3.pkl','rb'))
kfold4 = pickle.load(open('../models/kfold_senkin4.pkl','rb'))
kfold5 = pickle.load(open('../models/kfold_senkin5.pkl','rb'))
kfold6 = pickle.load(open('../models/kfold_senkin6.pkl','rb'))
kfold7 = pickle.load(open('../models/kfold_senkin7.pkl','rb'))
kfold8 = pickle.load(open('../models/kfold_senkin8.pkl','rb'))
kfold9 = pickle.load(open('../models/kfold_senkin9.pkl','rb'))

test_sup['kfold0'] = kfold0
test_sup['kfold1'] = kfold1
test_sup['kfold2'] = kfold2
test_sup['kfold3'] = kfold3
test_sup['kfold4'] = kfold4
test_sup['kfold5'] = kfold5
test_sup['kfold6'] = kfold6
test_sup['kfold7'] = kfold7
test_sup['kfold8'] = kfold8
test_sup['kfold9'] = kfold9

test_sup['is_attributed'] = (test_sup['kfold0'] + test_sup['kfold1'] + test_sup['kfold2'] + 
                             test_sup['kfold3'] + test_sup['kfold4'] + test_sup['kfold5'] +
                             test_sup['kfold6'] + test_sup['kfold7'] + test_sup['kfold8'] +
                             test_sup['kfold9']) / 10

test_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
test_sup['is_attributed'] = test_pred
test_sup['click_time'] = pd.to_datetime(test_sup['click_time'])
test_sup['click_time'] = test_sup['click_time'] + datetime.timedelta(hours=-8)

join_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
all_cols = join_cols + ['is_attributed']

test = pd.read_csv('../input/test.csv')
test['click_time'] = pd.to_datetime(test['click_time'])
test = test.merge(test_sup[all_cols], how='left', on=join_cols)
test = test.drop_duplicates(subset=['click_id'])

test[['click_id', 'is_attributed']].to_csv('../models/lgb_2014.csv.gz', index=False, float_format='%.9f', compression='gzip')                             
