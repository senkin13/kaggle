%%time
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings 
warnings.filterwarnings('ignore')

##############train test
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
df = pd.concat([train,test],axis=0)
del train,test
gc.collect()
# fillna
df['first_active_month'].fillna('2017-09',inplace=True)
# date
df['yearmonth'] = df['first_active_month'].map(lambda x: str(x).replace('-','')).astype(int)
df['first_active_month'] = pd.to_datetime(df['first_active_month'])
df['year'] = df['first_active_month'].dt.year
df['month'] = df['first_active_month'].dt.month
df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
df['elapsed_time_month'] = df['elapsed_time'] // 30
################historical
df_hist = pd.read_csv("../input/historical_transactions.csv")
df_hist = df_hist.sort_values(by=['card_id', 'purchase_date'], ascending=True)
df_hist = df_hist[df_hist['purchase_amount']<1000000]
df_hist['authorized_flag'] = df_hist['authorized_flag'].map({'Y':1, 'N':0})
df_hist['category_1'] = df_hist['category_1'].map({'Y':1, 'N':0})
# ohe
df_hist['category_2_ohe'] = df_hist['category_2'] 
df_hist['category_3_ohe'] = df_hist['category_3'] 
df_hist = pd.get_dummies(df_hist, columns=['category_2_ohe', 'category_3_ohe'])
# date
df_hist['purchase_date'] = pd.to_datetime(df_hist['purchase_date'])
df_hist['year'] = df_hist['purchase_date'].dt.year
df_hist['month'] = df_hist['purchase_date'].dt.month
df_hist['woy'] = df_hist['purchase_date'].dt.weekofyear
df_hist['doy'] = df_hist['purchase_date'].dt.dayofyear
df_hist['wday'] = df_hist['purchase_date'].dt.dayofweek
df_hist['weekend'] = (df_hist.purchase_date.dt.weekday >=5).astype(int)
df_hist['day'] = df_hist['purchase_date'].dt.day
df_hist['hour'] = df_hist['purchase_date'].dt.hour
df_hist['datehour'] = df_hist['purchase_date'].dt.strftime('%Y%m%d%H')
df_hist['date'] = df_hist['purchase_date'].dt.strftime('%Y%m%d')
df_hist['purchase_year_month'] = df_hist['year'].map(lambda x:0 if x==2011 
                                            else 12 if x==2012
                                            else 24 if x==2013
                                            else 36 if x==2014
                                            else 48 if x==2015
                                            else 60 if x==2016
                                            else 72 if x==2017
                                            else 84 if x==2018
                                            else x
                                            ) + df_hist['month'] 
df_hist['month_diff'] = ((datetime.date(2018, 5, 1)  - df_hist['purchase_date'].dt.date).dt.days)//30
df_hist['month_diff'] += df_hist['month_lag']
df_hist['week_diff'] = ((datetime.date(2018, 5, 1)  - df_hist['purchase_date'].dt.date).dt.days)//7
df_hist['day_diff'] = ((datetime.date(2018, 5, 1)  - df_hist['purchase_date'].dt.date).dt.days)
df_hist['pre_purchase_diff'] = df_hist[['card_id','purchase_date']].groupby(['card_id'])['purchase_date'].transform(lambda x: x.diff(1).dt.days)
df_hist['purchase_amount_new'] = np.round(df_hist['purchase_amount'] / 0.00150265118 + 497.06,2)
df_hist['pre_purchase_amount_new_diff'] = df_hist[['card_id','purchase_amount_new']].groupby(['card_id'])['purchase_amount_new'].transform(lambda x: x.diff().shift(0))
df_hist['refer_purchase_amount_new'] = df_hist['purchase_amount_new'] / df_hist['refer_date']

################new
df_new = pd.read_csv("../input/new_merchant_transactions.csv")
df_new = df_new.sort_values(by=['card_id', 'purchase_date'], ascending=True)
df_new['authorized_flag'] = df_new['authorized_flag'].map({'Y':1, 'N':0})
df_new['category_1'] = df_new['category_1'].map({'Y':1, 'N':0})
df_new['category_2_ohe'] = df_new['category_2'] 
df_new['category_3_ohe'] = df_new['category_3'] 
# ohe
df_new = pd.get_dummies(df_new, columns=['category_2_ohe', 'category_3_ohe'])
# date
df_new['purchase_date'] = pd.to_datetime(df_new['purchase_date'])
df_new['year'] = df_new['purchase_date'].dt.year
df_new['month'] = df_new['purchase_date'].dt.month
df_new['woy'] = df_new['purchase_date'].dt.weekofyear
df_new['doy'] = df_new['purchase_date'].dt.dayofyear
df_new['wday'] = df_new['purchase_date'].dt.dayofweek
df_new['weekend'] = (df_new.purchase_date.dt.weekday >=5).astype(int)
df_new['day'] = df_new['purchase_date'].dt.day
df_new['hour'] = df_new['purchase_date'].dt.hour
df_new['datehour'] = df_new['purchase_date'].dt.strftime('%Y%m%d%H')
df_new['date'] = df_new['purchase_date'].dt.strftime('%Y%m%d')
df_new['purchase_year_month'] = df_new['year'].map(lambda x:0 if x==2011 
                                            else 12 if x==2012
                                            else 24 if x==2013
                                            else 36 if x==2014
                                            else 48 if x==2015
                                            else 60 if x==2016
                                            else 72 if x==2017
                                            else 84 if x==2018
                                            else x
                                            ) + df_new['month'] 
df_new['month_diff'] = ((datetime.date(2018, 5, 1) - df_new['purchase_date'].dt.date).dt.days)//30
df_new['month_diff'] += df_new['month_lag']
df_new['week_diff'] = ((datetime.date(2018, 5, 1)  - df_new['purchase_date'].dt.date).dt.days)//7
df_new['day_diff'] = ((datetime.date(2018, 5, 1)  - df_new['purchase_date'].dt.date).dt.days)
df_new['pre_purchase_diff'] = df_new[['card_id','purchase_date']].groupby(['card_id'])['purchase_date'].transform(lambda x: x.diff(1).dt.days)
df_new['purchase_amount_new'] = np.round(df_new['purchase_amount'] / 0.00150265118 + 497.06,2)
df_new['pre_purchase_amount_new_diff'] = df_new[['card_id','purchase_amount_new']].groupby(['card_id'])['purchase_amount_new'].transform(lambda x: x.diff().shift(0))
df_new['refer_purchase_amount_new'] = df_new['purchase_amount_new'] / df_new['refer_date']


################merchant
merchant = pd.read_csv("../input/merchants.csv")
merchant.drop_duplicates(subset=['merchant_id'],keep='first',inplace=True)
merchant['merchant_category_1'] = merchant['category_1'].map({'Y':1, 'N':0})
merchant['merchant_category_4'] = merchant['category_4'].map({'Y':1, 'N':0})
merchant['merchant_category_2'] = merchant['category_2']
# ohe
merchant = pd.get_dummies(merchant, columns=['merchant_category_2'])
merchant.drop(['category_1','category_2','category_4'],axis=1,inplace=True)

##############merge merchant to hist and new
df_hist = df_hist.merge(merchant,on=['merchant_id'],how='left')
df_new = df_new.merge(merchant,on=['merchant_id'],how='left')
del merchant
gc.collect()

##############merge train/test to hist and new
df_hist_train = df.merge(df_hist,on='card_id',how='left')
df_new_train = df.merge(df_new,on='card_id',how='left')
print ('Load Done') 

%%time
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# lable encoder 
def lbl_enc(df):
    for c in ['merchant_id','most_recent_sales_range','most_recent_purchases_range','category_3']:
        if df[c].dtype == 'object':
            lbl = LabelEncoder()
            df[c] = lbl.fit_transform(df[c].astype(str))
    return df

df_hist_train = lbl_enc(df_hist_train)
df_new_train = lbl_enc(df_new_train)

df_hist_new_train = pd.concat([df_hist_train,df_new_train],axis=0)

%%time
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
from scipy import sparse
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.feature_selection import chi2, SelectPercentile


def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

# use df_hist_train df_new_train df_hist_new_train to train 3 models
train_df = df_hist_new_train[df_hist_new_train['target'].notnull()]
test_df = df_hist_new_train[df_hist_new_train['target'].isnull()]

drop_features=['card_id', 'target', 'purchase_date','first_active_month',
               'outliers',
              ]

feats = [f for f in df_hist_new_train.columns if f not in drop_features]


cat_features = [c for c in feats if 'feature_' in c]
n_splits= 5
folds = GroupKFold(n_splits=n_splits)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])

print ('feats:' + str(len(feats)))

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'],groups=train_df['card_id'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx] 
    
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
               "objective" : "regression", 
               "boosting" : "gbdt", 
               "metric" : "rmse",  
               "max_depth": 10, 
               "num_leaves" : 31, 
               "max_bin" : 255, 
               "learning_rate" : 0.01, 
               "subsample" : 1,
               "colsample_bytree" : 0.8, 
               "verbosity": -1,
    }
    

    if n_fold >= 0:
        evals_result = {}
        dtrain = lgb.Dataset(
            train_x, label=train_y,categorical_feature=cat_features)
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain,categorical_feature=cat_features)
        bst = lgb.train(
            params, dtrain, num_boost_round=30000,
            valid_sets=[dval], early_stopping_rounds=200, verbose_eval=100,)#feval = evalerror
        
        new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
        for item in new_list:
            print (item) 

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)

        sub_preds += bst.predict(test_df[feats], num_iteration=bst.best_iteration) / folds.n_splits # test_df_new

cv = rmse(train_df['target'],  oof_preds)
print('Full OOF RMSE %.6f' % cv)  

# oof_df = pd.DataFrame()
# oof_df['card_id'] = train_df['card_id']
# oof_df['target'] = oof_preds
# oof_df[['card_id','target']].to_csv('../ensemble/lgb_oof_' + str(cv) + '.csv',index=False)

# test_df['target'] = sub_preds
# test_df[['card_id','target']].to_csv('../ensemble/lgb_pred_' + str(cv) + '.csv',index=False)
