import datetime as dt
from datetime import timedelta

import os
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
from sklearn import preprocessing


def get_training_data(df , first_train_day):
    """ 
    Args:
        df: 
        first_train_day:
        
    Returns:
        DataFrame: one row for each fullVisitorId in df
    """
    last_train_day = first_train_day + timedelta(days=168)
    first_test_day = last_train_day + timedelta(days=46)
    last_test_day = first_test_day + timedelta(days=62)

    before_train_visits = df[(df.visitStartTime< first_train_day)]
    train_visits = df[(df.visitStartTime>= first_train_day) & (df.visitStartTime<last_train_day)]
    test_visits = df[(df.visitStartTime>=first_test_day) & (df.visitStartTime<last_test_day)]

    y = get_target(train_visits, test_visits)
    X = create_features(train_visits,before_train_visits)
    X['target']=y.target
    return X

def create_features(df,before_df):
    """ feature engineering... constant fields + average last months + months ago
    Args:
        df: 
        
    Returns:
        DataFrame: one row for each fullVisitorId in df
    """
    before_df_visitor = pd.DataFrame(before_df.fullVisitorId.unique(),columns=['fullVisitorId'])
    before_df_visitor['before_fullVisitorId_count'] = before_df_visitor['fullVisitorId'].map(before_df.groupby(['fullVisitorId'])['totals_pageviews'].size())
    before_df_visitor['before_totals_transactionRevenue_sum'] = before_df_visitor['fullVisitorId'].map(before_df.groupby(['fullVisitorId'])['totals_transactionRevenue'].sum())
    before_df_visitor['before_totals_pageviews_sum'] = before_df_visitor['fullVisitorId'].map(before_df.groupby(['fullVisitorId'])['totals_pageviews'].sum())
    before_df_visitor['before_totals_hits_sum'] = before_df_visitor['fullVisitorId'].map(before_df.groupby(['fullVisitorId'])['totals_hits'].sum())
    before_df_visitor['before_totals_timeOnSite_sum'] = before_df_visitor['fullVisitorId'].map(before_df.groupby(['fullVisitorId'])['totals_timeOnSite'].sum())
    before_df_visitor['before_totals_transactions_sum'] = before_df_visitor['fullVisitorId'].map(before_df.groupby(['fullVisitorId'])['totals_transactions'].sum())
    before_df_visitor['before_totals_bounces_sum'] = before_df_visitor['fullVisitorId'].map(before_df.groupby(['fullVisitorId'])['totals_bounces'].sum())
    before_df_visitor['before_totals_newVisits_sum'] = before_df_visitor['fullVisitorId'].map(before_df.groupby(['fullVisitorId'])['totals_newVisits'].sum())
    
    cat_cols = ['fullVisitorId','device_browser', 'device_isMobile', 
                'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country', 
                'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region', 'geoNetwork_subContinent', 
                'totals_bounces', 'totals_newVisits', 'trafficSource_adContent', 'trafficSource_adwordsClickInfo.adNetworkType', 
                'trafficSource_adwordsClickInfo.isVideoAd', 'trafficSource_adwordsClickInfo.gclId','trafficSource_campaign',
                'trafficSource_adwordsClickInfo.page', 'trafficSource_adwordsClickInfo.slot', 
                'trafficSource_isTrueDirect', 'trafficSource_keyword', 'trafficSource_medium', 'trafficSource_source', 'trafficSource_referralPath']
                
    df_visitor = pd.DataFrame(df.fullVisitorId.unique(),columns=['fullVisitorId'])
    
    df_visitor_last = df.drop_duplicates(subset=['fullVisitorId'], keep='last')
    df_visitor_last = df_visitor_last[cat_cols]
    #df_visitor = df.groupby(['fullVisitorId'])['totals_transactionRevenue'].sum().reset_index()
    df_visitor['fullVisitorId_count'] = df_visitor['fullVisitorId'].map(df.groupby(['fullVisitorId'])['totals_pageviews'].size())
    df_visitor['totals_transactionRevenue_sum'] = df_visitor['fullVisitorId'].map(df.groupby(['fullVisitorId'])['totals_transactionRevenue'].sum())
    df_visitor['totals_pageviews_sum'] = df_visitor['fullVisitorId'].map(df.groupby(['fullVisitorId'])['totals_pageviews'].sum())
    df_visitor['totals_hits_sum'] = df_visitor['fullVisitorId'].map(df.groupby(['fullVisitorId'])['totals_hits'].sum())
    df_visitor['totals_timeOnSite_sum'] = df_visitor['fullVisitorId'].map(df.groupby(['fullVisitorId'])['totals_timeOnSite'].sum())
    df_visitor['totals_transactions_sum'] = df_visitor['fullVisitorId'].map(df.groupby(['fullVisitorId'])['totals_transactions'].sum())
    df_visitor['totals_bounces_sum'] = df_visitor['fullVisitorId'].map(df.groupby(['fullVisitorId'])['totals_bounces'].sum())
    df_visitor['totals_newVisits_sum'] = df_visitor['fullVisitorId'].map(df.groupby(['fullVisitorId'])['totals_newVisits'].sum())

    df_visitor['transactionRevenue_pageviews_ratio'] = df_visitor['totals_transactionRevenue_sum'] / df_visitor['totals_pageviews_sum']
    df_visitor['transactionRevenue_hits_ratio'] = df_visitor['totals_transactionRevenue_sum'] / df_visitor['totals_hits_sum']
    df_visitor['transactionRevenue_timeOnSite_ratio'] = df_visitor['totals_transactionRevenue_sum'] / df_visitor['totals_timeOnSite_sum']
    df_visitor['transactionRevenue_transactions_ratio'] = df_visitor['totals_transactionRevenue_sum'] / df_visitor['totals_transactions_sum']
    df_visitor['pageviews_hits_ratio'] = df_visitor['totals_pageviews_sum'] / df_visitor['totals_hits_sum']
    df_visitor['pageviews_timeOnSite_ratio'] = df_visitor['totals_pageviews_sum'] / df_visitor['totals_timeOnSite_sum']
    df_visitor['pageviews_transactions_ratio'] = df_visitor['totals_pageviews_sum'] / df_visitor['totals_transactions_sum']
    df_visitor['hits_timeOnSite_ratio'] = df_visitor['totals_hits_sum'] / df_visitor['totals_timeOnSite_sum']
    df_visitor['hits_transactions_ratio'] = df_visitor['totals_hits_sum'] / df_visitor['totals_transactions_sum']
    df_visitor['transactions_timeOnSite_ratio'] = df_visitor['totals_transactions_sum'] / df_visitor['totals_timeOnSite_sum']

    
    for d in df['channelGrouping'].unique():
        df_visitor['fullVisitorId_channelGrouping_' + str(d)] = df_visitor['fullVisitorId'].map(df[df['channelGrouping']==d].groupby(['fullVisitorId'])['channelGrouping'].count())
        df_visitor['fullVisitorId_channelGrouping_' + str(d)] = df_visitor['fullVisitorId_channelGrouping_' + str(d)].fillna(0)

    for d in df['trafficSource_medium'].unique():
        df_visitor['fullVisitorId_trafficSource_medium_' + str(d)] = df_visitor['fullVisitorId'].map(df[df['trafficSource_medium']==d].groupby(['fullVisitorId'])['trafficSource_medium'].count())
        df_visitor['fullVisitorId_trafficSource_medium_' + str(d)] = df_visitor['fullVisitorId_trafficSource_medium_' + str(d)].fillna(0)

    for d in df['device_operatingSystem'].unique():
        df_visitor['fullVisitorId_device_operatingSystem_' + str(d)] = df_visitor['fullVisitorId'].map(df[df['device_operatingSystem']==d].groupby(['fullVisitorId'])['device_operatingSystem'].count())
        df_visitor['fullVisitorId_device_operatingSystem_' + str(d)] = df_visitor['fullVisitorId_device_operatingSystem_' + str(d)].fillna(0)

    for d in df['device_deviceCategory'].unique():
        df_visitor['fullVisitorId_device_deviceCategory_' + str(d)] = df_visitor['fullVisitorId'].map(df[df['device_deviceCategory']==d].groupby(['fullVisitorId'])['device_deviceCategory'].count())
        df_visitor['fullVisitorId_device_deviceCategory_' + str(d)] = df_visitor['fullVisitorId_device_deviceCategory_' + str(d)].fillna(0)
        
    df_visitor = df_visitor.merge(df_visitor_last,on='fullVisitorId',how='left')
    
    df_visitor = df_visitor.merge(before_df_visitor,on='fullVisitorId',how='left')
    
    return df_visitor
#     last_train_day = df.date.max()
#     feature_dfs=[]

#     feature_dfs.append(get_fixed_fields(df)) 

#     for i in [1,2,3,6]:
#         start = last_train_day - timedelta(days=i*30)
#         end = last_train_day
#         suffix = "_last_%s_months" %(i)
#         feature_dfs.append(get_cummulate_numeric_fields(df, start, end, suffix=suffix))

#     for i in [2,3,4]:
#         start = last_train_day - timedelta(days=i*30)
#         end = start + timedelta(days=30)
#         suffix = "_%s_months_ago" %(i)
#         feature_dfs.append(get_cummulate_numeric_fields(df, start, end, suffix=suffix))
    
#     return pd.concat(feature_dfs, axis=1)



def get_cummulate_numeric_fields(df, start_date, end_date, suffix=''):
    """ cummulate of numeric fields 
    Args:
        df: 
        
    Returns:
        DataFrame: one row for each fullVisitorId in df and
            one column per float64 field
    """
    numeric_column_names = df.select_dtypes(include='float64').columns
    
    result = pd.DataFrame(index=df.fullVisitorId.unique())
    result.index.name= 'fullVisitorId'

    relevant_df = df[(df.date>= start_date)& (df.date<=end_date)]
    different_visitors = relevant_df.groupby('fullVisitorId')
    
    for col in numeric_column_names:
        cummulative_sum = different_visitors[col].sum()
        result[col+suffix] = cummulative_sum
    return result.fillna(0)



def get_fixed_fields(df, suffix=''):
    """ mode of all  
 
    Args:
        df: 
        
    Returns:
        DataFrame: one row for each fullVisitorId in df and
            one column per object field
    """
    
    object_column_names = list(df.select_dtypes(include='object').columns) 
    object_column_names.remove('fullVisitorId')
    
    result = pd.DataFrame(index=df.fullVisitorId.unique())
    result.index.name= 'fullVisitorId'
    different_visitors = df.groupby('fullVisitorId')
    
    for col in object_column_names:
        most_common = different_visitors[col].last()
        result[col+suffix] = most_common
    return result


def get_target(train_visits, test_visits):
    """gets the target.
 
    Args:
        train_visits: DataFrame. Each row is a visit in the past
        test_visits: DataFrame. Each row is a visit in the future
 
    Returns:
        DataFrame: one row for each fullVisitorId in train_visit and
            one column (log_total_spent) with the natural log 
            of the sum of all transactions in test_visits dataframe
    """
    target = pd.DataFrame(index=train_visits.fullVisitorId.unique())
    target.index.name = 'fullVisitorId'
    total_spent = test_visits.groupby('fullVisitorId')['totals_transactionRevenue'].sum() #agg(np.sum)
    target['total_spent']=total_spent
    target = target.fillna(0)
    target['target']= target.total_spent.apply(lambda x: np.log(x+1))
    target=target.drop(columns=['total_spent'])
    return target.reset_index()



def label_encode_object_dtypes(df):
    """Label encodes all the columns of a DataFrames df1 and df2 that have
       dtype='object'
 
    Args:
        df
        
    Returns:
        DataFrame: df with object types encoded.
    """
    object_column_names = list(df.select_dtypes(include='object').columns)    
    
    for col in object_column_names:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col].values)
        df[col] = lbl.transform(df[col].values)
    return df



def fill_empty_values(df):
    """'
    Args:
        df1
        
    Returns:
        DataFrame: df1 with filled empty values and columns types formated
    """
    df['totals_newVisits'] = df['totals_newVisits'].astype('object')
    df['totals_bounces'] = df['totals_bounces'].astype('object')
    df['device_isMobile'] = df['device_isMobile'].astype('object')

    object_column_names = df.select_dtypes(include='object').columns 
    for col in object_column_names:
        df[col] = df[col].apply(unicode).apply(lambda x:x.encode('utf-8'))

    numeric_column_names = set(df.columns).difference(set(object_column_names))
    for col in numeric_column_names:
        df[col] = df[col].astype('float64')

    df['date'] = pd.to_datetime(df.visitStartTime, unit='s').astype('datetime64')
    df = df.drop(columns = ['visitStartTime', 'visitId'])
    df = df.fillna(0)
    return df



def get_basic_info(df):
    """gets basic information from a data frame
 
    Args:
        df
        
    Returns:
        A dataframe that has one row for each column in df1 and 5 columns:
        unique_elements(the number of unique values), 'mode' (most common element), 
        'empty_values' (number of empty values), 'dtype', 'types' (python types
        contained in that column)
    """
    colnames = df.columns
    Info= pd.DataFrame({
        'column_name': colnames, 
        'unique_elements': [df[col].nunique() for col in colnames],
        'mode': [df[col].mode()[0] for col in colnames],
        'empty_values': [df[col].isna().sum() for col in colnames],
        'dtype': df.dtypes.values,
        'types': [df[col].apply(type).value_counts().to_dict() for col in colnames]
    })
    return Info.set_index('column_name')
    
%%time
import datetime as dt
import matplotlib as mp
import numpy as np
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

train = pd.read_pickle('../input/train_v2_clean.pkl')
test = pd.read_pickle('../input/test_v2_clean.pkl')
sub = pd.read_csv('../input/sample_submission_v2.csv')

train.totals_timeOnSite = train.totals_timeOnSite.astype(float)
train.totals_transactions = train.totals_transactions.astype(float)
train['visitStartTime'] = train['visitStartTime'].apply(
        lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
train['visitStartTime'] = pd.to_datetime(train['visitStartTime'] )
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].astype(float)
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].fillna(0)

test.totals_timeOnSite = test.totals_timeOnSite.astype(float)
test.totals_transactions = test.totals_transactions.astype(float)
test['visitStartTime'] = test['visitStartTime'].apply(
        lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
test['visitStartTime'] = pd.to_datetime(test['visitStartTime'] )
test['totals_transactionRevenue'] = test['totals_transactionRevenue'].astype(float)
test['totals_transactionRevenue'] = test['totals_transactionRevenue'].fillna(0)

## Drop columns
cols_to_drop = [col for col in train.columns if train[col].nunique(dropna=False) == 1]
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop([col for col in cols_to_drop if col in test.columns], axis=1, inplace=True)

## only one not null value
# train.drop(['trafficSource_campaignCode'], axis=1, inplace=True)
# test.drop(['trafficSource_campaignCode'], axis=1, inplace=True)

###converting columns format
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].astype(float)
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].fillna(0)
#train['totals_transactionRevenue'] = np.log1p(train_df['totals_transactionRevenue'])

train['totals_bounces'] = train['totals_bounces'].fillna('0')
test['totals_bounces'] = test['totals_bounces'].fillna('0')
train['totals_newVisits'] = train['totals_newVisits'].fillna('0')
test['totals_newVisits'] = test['totals_newVisits'].fillna('0')

train['flag'] = 'train'
test['flag'] = 'test'
df = pd.concat([train,test],axis=0)

cat_cols = ['device_browser', 'device_isMobile', 
                'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country', 
                'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region', 'geoNetwork_subContinent', 
                'totals_bounces', 'totals_newVisits', 'trafficSource_adContent', 'trafficSource_adwordsClickInfo.adNetworkType', 
                'trafficSource_adwordsClickInfo.isVideoAd', 'trafficSource_adwordsClickInfo.gclId','trafficSource_campaign',
                'trafficSource_adwordsClickInfo.page', 'trafficSource_adwordsClickInfo.slot', 
                'trafficSource_isTrueDirect', 'trafficSource_keyword', 'trafficSource_medium', 'trafficSource_source', 'trafficSource_referralPath']

for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(df[col].values.astype('str')))
    df[col] = lbl.transform(list(df[col].values.astype('str')))

df.to_pickle('../input/df.pkl')    
train = df[df['flag']=='train']    
test = df[df['flag']=='test'] 
print ('train shape:' + str(train.shape))
print ('test shape:' + str(test.shape))
print ('sub shape:' + str(sub.shape))
print ('df shape:' + str(df.shape))

%%time
import datetime as dt
import matplotlib as mp
import numpy as np
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_pickle('../input/df.pkl')    
train = df[df['flag']=='train']    
test = df[df['flag']=='test'] 
print ('train shape:' + str(train.shape))
print ('test shape:' + str(test.shape))
print ('df shape:' + str(df.shape))

%%time
train_1 = get_training_data(df , first_train_day = dt.datetime(2017, 1, 1))
print (train_1[train_1['target']>0].shape)
train_2 = get_training_data(df , first_train_day = dt.datetime(2017, 2, 1))
print (train_2[train_2['target']>0].shape)
train_3 = get_training_data(df , first_train_day = dt.datetime(2017, 3, 1))
print (train_3[train_3['target']>0].shape)
train_4 = get_training_data(df , first_train_day = dt.datetime(2017, 4, 1))
print (train_4[train_4['target']>0].shape)
train_5 = get_training_data(df , first_train_day = dt.datetime(2017, 5, 1))
print (train_5[train_5['target']>0].shape)
train_6 = get_training_data(df , first_train_day = dt.datetime(2017, 6, 1))
print (train_6[train_6['target']>0].shape)
train_7 = get_training_data(df , first_train_day = dt.datetime(2017, 7, 1))
print (train_7[train_7['target']>0].shape)
train_8 = get_training_data(df , first_train_day = dt.datetime(2017, 8, 1))
print (train_8[train_8['target']>0].shape)
train_9 = get_training_data(df , first_train_day = dt.datetime(2017, 9, 1))
print (train_9[train_9['target']>0].shape)
train_10 = get_training_data(df , first_train_day = dt.datetime(2017, 10, 1))
print (train_10[train_10['target']>0].shape)
train_11 = get_training_data(df , first_train_day = dt.datetime(2017, 11, 1))
print (train_11[train_11['target']>0].shape)
train_12 = get_training_data(df , first_train_day = dt.datetime(2017, 12, 1))
print (train_12[train_12['target']>0].shape)

train_10 = get_training_data(df , first_train_day = dt.datetime(2017, 1, 16))

train_20 = get_training_data(df , first_train_day = dt.datetime(2017, 2, 16))

train_30 = get_training_data(df , first_train_day = dt.datetime(2017, 3, 16))

train_40 = get_training_data(df , first_train_day = dt.datetime(2017, 4, 16))

train_50 = get_training_data(df , first_train_day = dt.datetime(2017, 5, 16))

train_60 = get_training_data(df , first_train_day = dt.datetime(2017, 6, 16))

train_70 = get_training_data(df , first_train_day = dt.datetime(2017, 7, 16))

train_80 = get_training_data(df , first_train_day = dt.datetime(2017, 8, 16))

train_90 = get_training_data(df , first_train_day = dt.datetime(2017, 9, 16))

train_100 = get_training_data(df , first_train_day = dt.datetime(2017, 10, 16))

train_110 = get_training_data(df , first_train_day = dt.datetime(2017, 11, 16))

train_120 = get_training_data(df , first_train_day = dt.datetime(2017, 12, 16))

train_13 = get_training_data(df , first_train_day = dt.datetime(2018, 1, 1))
print (train_13[train_13['target']>0].shape)

# train_13 = get_training_data(df , first_train_day = dt.datetime(2016, 8, 1))
# print (train_13[train_13['target']>0].shape)
# train_14 = get_training_data(df , first_train_day = dt.datetime(2016, 9, 1))
# print (train_14[train_14['target']>0].shape)
# train_15 = get_training_data(df , first_train_day = dt.datetime(2016, 10, 1))
# print (train_15[train_15['target']>0].shape)
# train_16 = get_training_data(df , first_train_day = dt.datetime(2016, 11, 1))
# print (train_16[train_16['target']>0].shape)
# train_17 = get_training_data(df , first_train_day = dt.datetime(2016, 12, 1))
# print (train_17[train_17['target']>0].shape)
# 
train_df = pd.concat([ train_1, train_2, train_3,  train_4,  train_5,  train_6,
                      train_7,  train_8, train_9,  train_10,  train_11,  train_12,train_13,
                      train_10, train_20, train_30,  train_40,  train_50,  train_60,
                      train_70,  train_80, train_90,  train_100,  train_110,  train_120,], ignore_index=True)

#test_df = get_training_data(df , first_train_day = dt.datetime(2018, 1, 1))

test_df = create_features(test,train)


##### train

train_df['flag'] = 'train'
test_df['flag'] = 'test'

train_test_df = pd.concat([train_df,test_df],axis=0)

train_df = train_test_df[train_test_df['flag']=='train']
test_df = train_test_df[train_test_df['flag']=='test']

%%time
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

drop_features=['fullVisitorId', 'target','flag','pageviews_hits_ratio',
               
#                'transactionRevenue_pageviews_ratio',
#                'transactionRevenue_hits_ratio',
#                'transactionRevenue_timeOnSite_ratio',
#                'transactionRevenue_transactions_ratio',
#                'transactions_timeOnSite_ratio',
               'pageviews_hits_ratio',
               'pageviews_timeOnSite_ratio',
               'pageviews_transactions_ratio',
               ]

feats = [f for f in train_df.columns if f not in drop_features]
cat_features=[]
folds = KFold(n_splits= 5, shuffle=True, random_state=9999)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
  
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]    
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
               "objective" : "regression", 
               "boosting" : "gbdt", 
               "metric" : "rmse",  #None
               #"max_depth":8, 
               #"min_child_samples": 20, 
               #"reg_alpha": 10,
               #"reg_lambda": 50,
               "num_leaves" : 256, #128
               "max_bin" : 255, 
               "learning_rate" : 0.05, 
               "subsample" : 0.9, 
               "colsample_bytree" : 0.8, 
               "verbosity": -1
    }

    if n_fold >= 0:
        evals_result = {}
        dtrain = lgb.Dataset(
            train_x, label=train_y,)#feature_name=tfvocab categorcical_feature=lgb_cat
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain,)
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

# lb = rmse(test_df['target'],  sub_preds)
# print('Full TEST RMSE %.6f' % lb)  
