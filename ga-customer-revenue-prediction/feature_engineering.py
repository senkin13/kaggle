import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import time
from datetime import datetime
import gc
from sklearn.preprocessing import LabelEncoder
from gensim.sklearn_api.ldamodel import LdaTransformer
from gensim.models import LdaMulticore
from gensim import corpora
from gensim.models import Word2Vec
import pickle
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
from sklearn.metrics import mean_squared_error

PATH="../input/"

def read_parse_dataframe(file_name):
    #full path for the data file
    path = PATH + file_name
    #read the data file, convert the columns in the list of columns to parse using json loader,
    #convert the `fullVisitorId` field as a string
    data_df = pd.read_csv(path, 
        converters={column: json.loads for column in cols_to_parse}, 
        dtype={'fullVisitorId': 'str'})
    #parse the json-type columns
    for col in cols_to_parse:
        #each column became a dataset, with the columns the fields of the Json type object
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
        #we drop the object column processed and we add the columns created from the json fields
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)
    return data_df

#the columns that will be parsed to extract the fields from the jsons
cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']

## Load data
print('reading train')
train_df = read_parse_dataframe('train.csv')
trn_len = train_df.shape[0]

print('reading test')
test_df = read_parse_dataframe('test.csv')

## Drop columns
cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)

## only one not null value
train_df.drop(['trafficSource_campaignCode'], axis=1, inplace=True)
test_df.drop(['trafficSource_campaignCode'], axis=1, inplace=True)

###converting columns format
train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].astype(float)
train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].fillna(0)
train_df['totals_transactionRevenue'] = np.log1p(train_df['totals_transactionRevenue'])

train_df['totals_bounces'] = train_df['totals_bounces'].fillna('0')
test_df['totals_bounces'] = test_df['totals_bounces'].fillna('0')
train_df['totals_newVisits'] = train_df['totals_newVisits'].fillna('0')
test_df['totals_newVisits'] = test_df['totals_newVisits'].fillna('0')

###concat train and test
df = pd.concat([train_df,test_df],axis=0)
print (train_df.shape)
print (test_df.shape)
print (df.shape)

del train_df,test_df
gc.collect()

def process_device(data_df):
    print("process device ...")
    data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
    data_df['campaign_medium'] = data_df['trafficSource_campaign'] + '_' + data_df['trafficSource_medium']
    data_df['browser_category'] = data_df['device_browser'] + '_' + data_df['device_deviceCategory']
    data_df['browser_os'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
    return data_df

df = process_device(df)

def custom(data):
    print('custom..')
    data['device_deviceCategory_channelGrouping'] = data['device_deviceCategory'] + "_" + data['channelGrouping']
    data['channelGrouping_browser'] = data['device_browser'] + "_" + data['channelGrouping']
    data['channelGrouping_OS'] = data['device_operatingSystem'] + "_" + data['channelGrouping']
    
    for i in ['geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country','geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region','geoNetwork_subContinent']:
        for j in ['device_browser','device_deviceCategory', 'device_operatingSystem', 'trafficSource_source']:
            data[i + "_" + j] = data[i] + "_" + data[j]
    
    data['content_source'] = data['trafficSource_adContent'] + "_" + data['source_country']
    data['medium_source'] = data['trafficSource_medium'] + "_" + data['source_country']
    return data

df = custom(df)

print("process format ...")
for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
    df[col] = df[col].astype(float)
df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
df['trafficSource_isTrueDirect'].fillna(False, inplace=True)


# from sklearn import model_selection, preprocessing, metrics
# label encode the categorical variables and convert the numerical variables to float
cat_cols = ['channelGrouping', 'device_browser', 'device_deviceCategory', 'device_isMobile', 
'device_operatingSystem', 'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country', 
'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region', 'geoNetwork_subContinent', 
'totals_bounces', 'totals_newVisits', 'trafficSource_adContent', 'trafficSource_adwordsClickInfo.adNetworkType', 
'trafficSource_adwordsClickInfo.isVideoAd', 'trafficSource_adwordsClickInfo.gclId','trafficSource_campaign',
'trafficSource_adwordsClickInfo.page', 'trafficSource_adwordsClickInfo.slot', 
'trafficSource_isTrueDirect', 'trafficSource_keyword', 'trafficSource_medium', 'trafficSource_source', 
'trafficSource_referralPath','source_country', 'campaign_medium', 'browser_category', 'browser_os',
'device_deviceCategory_channelGrouping', 'channelGrouping_browser',
'channelGrouping_OS', 'geoNetwork_city_device_browser',
'geoNetwork_city_device_deviceCategory',
'geoNetwork_city_device_operatingSystem',
'geoNetwork_city_trafficSource_source',
'geoNetwork_continent_device_browser',
'geoNetwork_continent_device_deviceCategory',
'geoNetwork_continent_device_operatingSystem',
'geoNetwork_continent_trafficSource_source',
'geoNetwork_country_device_browser',
'geoNetwork_country_device_deviceCategory',
'geoNetwork_country_device_operatingSystem',
'geoNetwork_country_trafficSource_source',
'geoNetwork_metro_device_browser',
'geoNetwork_metro_device_deviceCategory',
'geoNetwork_metro_device_operatingSystem',
'geoNetwork_metro_trafficSource_source',
'geoNetwork_networkDomain_device_browser',
'geoNetwork_networkDomain_device_deviceCategory',
'geoNetwork_networkDomain_device_operatingSystem',
'geoNetwork_networkDomain_trafficSource_source',
'geoNetwork_region_device_browser',
'geoNetwork_region_device_deviceCategory',
'geoNetwork_region_device_operatingSystem',
'geoNetwork_region_trafficSource_source',
'geoNetwork_subContinent_device_browser',
'geoNetwork_subContinent_device_deviceCategory',
'geoNetwork_subContinent_device_operatingSystem',
'geoNetwork_subContinent_trafficSource_source', 'content_source',
'medium_source'
            ] # 

for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[col].values.astype('str')))
    df[col] = lbl.transform(list(df[col].values.astype('str')))

# time feature 
df['visitStartTime'] = df['visitStartTime'].apply(
        lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
df['visitStartTime'] = pd.to_datetime(df['visitStartTime'] )

df["year"] = df['visitStartTime'].dt.year
df["quarter"] = df['visitStartTime'].dt.quarter
df["month"] = df['visitStartTime'].dt.month
df["day"] = df['visitStartTime'].dt.day
df["hour"] = df['visitStartTime'].dt.hour
df["weekday"] = df['visitStartTime'].dt.weekday
df['weekofyear'] = df['visitStartTime'].dt.weekofyear
df['dayofyear'] = df['visitStartTime'].dt.dayofyear

# sort data by time
df = df.sort_values('visitStartTime').reset_index(drop=True)    

df['visitdate'] = df['visitStartTime'].dt.date
df['yearquarter'] = df['year'].map(str) + df['quarter'].map(str)
df['yearmonth'] = df['year'].map(str) + df['month'].map(str)
df['yearweek'] = df['year'].map(str) + df['weekofyear'].map(str)

# cumcount and sum
df['dummy'] = 1

df['user_cumcnt_per_visitmonth'] = (df[['fullVisitorId','yearmonth', 'dummy']].groupby(['fullVisitorId','yearmonth'])['dummy'].cumcount()+1)
df['user_sum_per_visitmonth'] = df[['fullVisitorId','yearmonth', 'dummy']].groupby(['fullVisitorId','yearmonth'])['dummy'].transform(sum)
df['user_cumcnt_sum_ratio_per_visitmonth'] = df['user_cumcnt_per_visitmonth'] / df['user_sum_per_visitmonth'] 

df['user_cumcnt_per_visitdate'] = (df[['fullVisitorId','visitdate', 'dummy']].groupby(['fullVisitorId','visitdate'])['dummy'].cumcount()+1)
df['user_sum_per_visitdate'] = df[['fullVisitorId','visitdate', 'dummy']].groupby(['fullVisitorId','visitdate'])['dummy'].transform(sum)
df['user_cumcnt_sum_ratio_per_visitdate'] = df['user_cumcnt_per_visitdate'] / df['user_sum_per_visitdate'] 

df['user_cumcnt'] = (df[['fullVisitorId', 'dummy']].groupby(['fullVisitorId'])['dummy'].cumcount()+1)
df['user_sum'] = df[['fullVisitorId', 'dummy']].groupby(['fullVisitorId'])['dummy'].transform(sum)
df['user_cumcnt_sum_ratio'] = df['user_cumcnt'] / df['user_sum'] 

df.drop('dummy', axis=1, inplace=True)

# cumsum and sum
df['user_pv_cumsum_per_yearmonth'] = (df[['fullVisitorId','yearmonth', 'totals_pageviews']].groupby(['fullVisitorId','yearmonth'])['totals_pageviews'].cumsum())
df['user_pv_sum_per_yearmonth'] = df[['fullVisitorId','yearmonth', 'totals_pageviews']].groupby(['fullVisitorId','yearmonth'])['totals_pageviews'].transform(sum)
df['user_pv_cumsum_sum_ratio_per_yearmonth'] = df['user_pv_cumsum_per_yearmonth'] / df['user_pv_sum_per_yearmonth'] 

df['user_pv_cumsum_per_yearweek'] = (df[['fullVisitorId','yearweek', 'totals_pageviews']].groupby(['fullVisitorId','yearweek'])['totals_pageviews'].cumsum())
df['user_pv_sum_per_yearweek'] = df[['fullVisitorId','yearweek', 'totals_pageviews']].groupby(['fullVisitorId','yearweek'])['totals_pageviews'].transform(sum)
df['user_pv_cumsum_sum_ratio_per_yearweek'] = df['user_pv_cumsum_per_yearweek'] / df['user_pv_sum_per_yearweek'] 

df['user_pv_cumsum'] = (df[['fullVisitorId','totals_pageviews']].groupby(['fullVisitorId'])['totals_pageviews'].cumsum())
df['user_pv_sum'] = df[['fullVisitorId','totals_pageviews']].groupby(['fullVisitorId'])['totals_pageviews'].transform(sum)
df['user_pv_cumsum_sum_ratio'] = df['user_pv_cumsum'] / df['user_pv_sum']

df['user_hits_cumsum'] = (df[['fullVisitorId','totals_hits']].groupby(['fullVisitorId'])['totals_hits'].cumsum())
df['user_hits_sum'] = df[['fullVisitorId','totals_hits']].groupby(['fullVisitorId'])['totals_hits'].transform(sum)
df['user_hits_cumsum_sum_ratio'] = df['user_hits_cumsum'] / df['user_hits_sum']

# cummax and sum
df['user_pv_cummax'] = (df[['fullVisitorId','totals_pageviews']].groupby(['fullVisitorId'])['totals_pageviews'].cummax())
df['user_pv_cummax_sum_ratio'] = df['user_pv_cummax'] / df['user_pv_sum']
df['user_pv_cummax_cum_ratio'] = df['user_pv_cummax'] / df['user_pv_cumsum']


df['user_hits_cummax'] = (df[['fullVisitorId','totals_hits']].groupby(['fullVisitorId'])['totals_hits'].cummax())
df['user_hits_cummax_sum_ratio'] = df['user_hits_cummax'] / df['user_hits_sum']
df['user_hits_cummax_cum_ratio'] = df['user_hits_cummax'] / df['user_hits_cumsum']

df['user_hits_cumsum_per_yearweek'] = (df[['fullVisitorId','yearweek', 'totals_hits']].groupby(['fullVisitorId','yearweek'])['totals_hits'].cumsum())
df['user_hits_sum_per_yearweek'] = df[['fullVisitorId','yearweek', 'totals_hits']].groupby(['fullVisitorId','yearweek'])['totals_hits'].transform(sum)
df['user_hits_cumsum_sum_ratio_per_yearweek'] = df['user_hits_cumsum_per_yearweek'] / df['user_hits_sum_per_yearweek'] 


# last and next session
df['next_session_1'] = (
        df['visitStartTime'] - df[['fullVisitorId', 'visitStartTime']].groupby('fullVisitorId')['visitStartTime'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
df['next_session_2'] = (
        df['visitStartTime'] - df[['fullVisitorId', 'visitStartTime']].groupby('fullVisitorId')['visitStartTime'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60

# aggreagation
from tqdm import tqdm
def agg(df,agg_cols):
    for c in tqdm(agg_cols):
        new_feature = '{}_{}_{}'.format('_'.join(c['groupby']), c['agg'], c['target'])
        gp = df.groupby(c['groupby'])[c['target']].agg(c['agg']).reset_index().rename(index=str, columns={c['target']:new_feature})
        df = df.merge(gp,on=c['groupby'],how='left')
    return df

agg_cols = [  
    {'groupby': ['trafficSource_source'], 'target':'totals_pageviews', 'agg':'mean'},    
    {'groupby': ['trafficSource_source'], 'target':'totals_pageviews', 'agg':'max'},    
    {'groupby': ['visitdate'], 'target':'totals_pageviews', 'agg':'mean'},    
    {'groupby': ['month'], 'target':'fullVisitorId', 'agg':'nunique'},     
    {'groupby': ['dayofyear'], 'target':'fullVisitorId', 'agg':'nunique'},     
    {'groupby': ['weekofyear'], 'target':'fullVisitorId', 'agg':'nunique'},    
    {'groupby': ['medium_source'], 'target':'totals_pageviews', 'agg':'mean'},
    {'groupby': ['source_country'], 'target':'totals_pageviews', 'agg':'mean'},     
    {'groupby': ['geoNetwork_continent_device_browser'], 'target':'totals_pageviews', 'agg':'mean'},  
    {'groupby': ['geoNetwork_continent_device_deviceCategory'], 'target':'totals_pageviews', 'agg':'mean'},  
    {'groupby': ['geoNetwork_continent_device_operatingSystem'], 'target':'totals_pageviews', 'agg':'mean'},  
    {'groupby': ['geoNetwork_continent_trafficSource_source'], 'target':'totals_pageviews', 'agg':'mean'},     
    {'groupby': ['device_deviceCategory_channelGrouping'], 'target':'totals_pageviews', 'agg':'mean'},  
    {'groupby': ['fullVisitorId'], 'target':'visitStartTime', 'agg':'max'},      
    {'groupby': ['fullVisitorId'], 'target':'visitStartTime', 'agg':'min'},   
    {'groupby': ['fullVisitorId'], 'target':'visitNumber', 'agg':'max'},
    {'groupby': ['medium_source'], 'target':'totals_hits', 'agg':'mean'},
    {'groupby': ['source_country'], 'target':'totals_hits', 'agg':'mean'},     
    {'groupby': ['geoNetwork_continent_device_browser'], 'target':'totals_hits', 'agg':'mean'},  
    {'groupby': ['geoNetwork_continent_device_deviceCategory'], 'target':'totals_hits', 'agg':'mean'},  
    {'groupby': ['geoNetwork_continent_device_operatingSystem'], 'target':'totals_hits', 'agg':'mean'},  
    {'groupby': ['geoNetwork_continent_trafficSource_source'], 'target':'totals_hits', 'agg':'mean'},     
    {'groupby': ['device_deviceCategory_channelGrouping'], 'target':'totals_hits', 'agg':'mean'},      
]

df = agg(df,agg_cols)

# diff
df['minmax_diff_date'] = (df['fullVisitorId_max_visitStartTime'] - df['fullVisitorId_min_visitStartTime']).map(lambda x:x.days)
df['max_diff_date'] = (df['fullVisitorId_max_visitStartTime'] - df['visitStartTime']).map(lambda x:x.days)
df['min_diff_date'] = (df['visitStartTime'] - df['fullVisitorId_min_visitStartTime']).map(lambda x:x.days)

df['diff_geoNetwork_continent_device_browser_mean_totals_pageviews'] = df['totals_pageviews'] - df['geoNetwork_continent_device_browser_mean_totals_pageviews']
df['diff_geoNetwork_continent_device_deviceCategory_mean_totals_pageviews'] = df['totals_pageviews'] - df['geoNetwork_continent_device_deviceCategory_mean_totals_pageviews']
df['diff_geoNetwork_continent_device_operatingSystem_mean_totals_pageviews'] = df['totals_pageviews'] - df['geoNetwork_continent_device_operatingSystem_mean_totals_pageviews']
df['diff_geoNetwork_continent_trafficSource_source_mean_totals_pageviews'] = df['totals_pageviews'] - df['geoNetwork_continent_trafficSource_source_mean_totals_pageviews']

# duplicated count
df["visitId_duplicates"] = df.visitId.map(df.visitId.value_counts())
df["session_duplicates"] = df.sessionId.map(df.sessionId.value_counts())

# target mean
def target_mean(df_train,train_df,test_df,cols):
    min_samples_leaf=100
    smoothing=10
    noise_level=0.01 #0.01
    for c in tqdm(cols):
        new_feature = '{}_{}'.format('_'.join(c['groupby']), c['func'])
        averages = df_train.groupby(c['groupby'])[['bin']].agg(['mean','count']).bin.reset_index()
        smoothing = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing))
        averages[new_feature] = df_train['bin'].mean() * (1 - smoothing) + averages['mean'] * smoothing
        averages.drop(['mean', 'count'], axis=1, inplace=True)

        np.random.seed(42)
        noise = np.random.randn(len(averages[new_feature])) * noise_level
        averages[new_feature] = averages[new_feature] + noise

        train_df = train_df.merge(averages,on=c['groupby'],how='left')
        test_df = test_df.merge(averages,on=c['groupby'],how='left')
        
    return train_df,test_df      

# frequency encoding
def frequency_encoding(df_train,train_df,test_df,col):
    freq_encoding = df_train.groupby([col]).size()/df_train.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    train_df = train_df.merge(freq_encoding,on=col,how='left')
    test_df = test_df.merge(freq_encoding,on=col,how='left')    
    del freq_encoding
    gc.collect()
    return train_df,test_df
    
# Load leak data
train_store_1 = pd.read_csv('../input/Train_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
train_store_2 = pd.read_csv('../input/Train_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_1 = pd.read_csv('../input/Test_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_2 = pd.read_csv('../input/Test_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})

#Getting VisitId to Join with our train, test data
for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(str)
    df["visitId"] = df["visitId"].astype('int64')
    
df_exdata = pd.concat([train_store_1, train_store_2,test_store_1, test_store_2], sort=False)

df_exdata["Revenue"].fillna('$', inplace=True)
df_exdata["Revenue"] = df_exdata["Revenue"].apply(lambda x: x.replace('$', '').replace(',', ''))
df_exdata["Revenue"] = pd.to_numeric(df_exdata["Revenue"], errors="coerce")
df_exdata["Revenue"].fillna(0.0, inplace=True)
df_exdata["Revenue_log1p"] = np.log1p(df_exdata["Revenue"]*1000000)

df_exdata["Sessions"] = df_exdata["Sessions"].fillna(0)
df_exdata["Avg. Session Duration"] = df_exdata["Avg. Session Duration"].fillna(0)
df_exdata["Bounce Rate"] = df_exdata["Bounce Rate"].fillna(0)
df_exdata["Transactions"] = df_exdata["Transactions"].fillna(0)
df_exdata["Goal Conversion Rate"] = df_exdata["Goal Conversion Rate"].fillna(0)

def str_to_seconds(t):
    x = time.strptime(t,'%H:%M:%S')
    y = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    return y

df_exdata["Avg. Session Duration"] = df_exdata["Avg. Session Duration"].apply(lambda x: str_to_seconds(x))

df_exdata["Bounce Rate"] = df_exdata["Bounce Rate"].apply(lambda x: x.replace('%', ''))
df_exdata["Bounce Rate"] = df_exdata["Bounce Rate"].astype('float64') / 100

df_exdata["Goal Conversion Rate"] = df_exdata["Goal Conversion Rate"].apply(lambda x: x.replace('%', ''))
df_exdata["Goal Conversion Rate"] = df_exdata["Goal Conversion Rate"].astype('float64') / 100

df = df.merge(df_exdata, how="left", on="visitId")
del df_exdata
gc.collect()

df['total_session_time'] = df['Avg. Session Duration'] * df['Sessions'] 
df['revenue_session_ratio'] = df['Revenue'] / df['Sessions']
df['revenue_transaction_ratio'] = df['Revenue'] / df['Transactions'] 
df['transaction_session_ratio'] = df['Transactions'] / df['Sessions']

def agg(df,agg_cols):
    for c in tqdm(agg_cols):
        new_feature = '{}_{}_{}'.format('_'.join(c['groupby']), c['agg'], c['target'])
        gp = df.groupby(c['groupby'])[c['target']].agg(c['agg']).reset_index().rename(index=str, columns={c['target']:new_feature})
        df = df.merge(gp,on=c['groupby'],how='left')
    return df

agg_cols = [  
    {'groupby': ['medium_source'], 'target':'Revenue', 'agg':'mean'},
    {'groupby': ['source_country'], 'target':'Revenue', 'agg':'mean'},     
    {'groupby': ['geoNetwork_continent_device_browser'], 'target':'Revenue', 'agg':'mean'},  
    {'groupby': ['geoNetwork_continent_device_deviceCategory'], 'target':'Revenue', 'agg':'mean'},  
    {'groupby': ['geoNetwork_continent_device_operatingSystem'], 'target':'Revenue', 'agg':'mean'},  
    {'groupby': ['geoNetwork_continent_trafficSource_source'], 'target':'Revenue', 'agg':'mean'},     
    {'groupby': ['device_deviceCategory_channelGrouping'], 'target':'Revenue', 'agg':'mean'},  

]
df = agg(df,agg_cols)

%%time

def rmse1(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

def rmse2(y_true, y_pred, m):
    return ((mean_squared_error(y_true, y_pred) * y_true.shape[0]) /m )** .5

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    valid_df = pd.DataFrame({"fullVisitorId":train_df['fullVisitorId'].iloc[valid_idx]})
    valid_df["transactionRevenue"] = np.expm1(labels)
    valid_df["PredictedRevenue"] = np.expm1(preds)
    valid_df = valid_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
    sum_rmse = mean_squared_error(np.log1p(valid_df["transactionRevenue"].values), np.log1p(valid_df["PredictedRevenue"].values)) ** .5
    return 'SUM_RMSE', sum_rmse, False

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

train_df = df[df['totals_transactionRevenue'].notnull()].reset_index()
test_df = df[df['totals_transactionRevenue'].isnull()].reset_index()

print("Starting Session based LightGBM. Train shape: {}, test shape: {}".format(train_df.shape,test_df.shape))
folds = get_folds(df=train_df, n_splits=5)


# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])

drop_features=['date', 'fullVisitorId', 'sessionId', 'visitdate','yearmonth','yearweek','yearquarter','index','bin','rev','local_time', #'auc_score',
               'fullVisitorId_max_visitStartTime','fullVisitorId_min_visitStartTime','month','day','weekofyear','device_isMobile',
               'content_source','nb_pageviews','ratio_pageviews','next_session_3','next_session_4',
               'totals_bounces','trafficSource_adContent','trafficSource_adwordsClickInfo.isVideoAd',
               'trafficSource_adwordsClickInfo.page','trafficSource_adwordsClickInfo.slot',
               'Revenue','trafficSource_adwordsClickInfo.adNetworkType','df_pv_mean_rank','chrismas_diff_date',
               'user_hits_cumsum','user_hits_sum','user_hits_cumsum_sum_ratio','user_hits_cummax_sum_ratio','visitdate_count_bin',
               'Client Id', 'fullVisitorId_delta_shift_-3', 'fullVisitorId_delta_shift_-2','fullVisitorId_delta_shift_-1',
               'fullVisitorId_delta_shift_2', 'fullVisitorId_delta_shift_3','fullVisitorId_delta_shift_1',
               'visitId', 'visitStartTime', 'totals_transactionRevenue']
feats = [f for f in train_df.columns if f not in drop_features]
print ('regular feats:' + str(len(feats)))

cat_features=[
'channelGrouping', 'device_browser', 
'device_operatingSystem', 
'geoNetwork_country',
'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region',
'trafficSource_isTrueDirect', 'trafficSource_keyword', 'trafficSource_medium', 'trafficSource_source', 
'trafficSource_referralPath',
'source_country', 'campaign_medium', 'browser_category', 'browser_os',
'device_deviceCategory_channelGrouping', 'channelGrouping_browser',
'channelGrouping_OS', 'geoNetwork_city_device_browser',
'geoNetwork_city_device_deviceCategory',
'geoNetwork_city_device_operatingSystem',
'geoNetwork_city_trafficSource_source',
'geoNetwork_continent_device_browser',
'geoNetwork_continent_device_deviceCategory',
'geoNetwork_continent_device_operatingSystem',
'geoNetwork_continent_trafficSource_source',
'geoNetwork_country_device_browser',
'geoNetwork_country_device_deviceCategory',
'geoNetwork_country_device_operatingSystem',
'geoNetwork_country_trafficSource_source',
'geoNetwork_metro_device_browser',
'geoNetwork_metro_device_deviceCategory',
'geoNetwork_metro_device_operatingSystem',
'geoNetwork_metro_trafficSource_source',
'geoNetwork_networkDomain_device_browser',
'geoNetwork_networkDomain_device_deviceCategory',
'geoNetwork_networkDomain_device_operatingSystem',
'geoNetwork_networkDomain_trafficSource_source',
'geoNetwork_region_device_browser',
'geoNetwork_region_device_deviceCategory',
'geoNetwork_region_device_operatingSystem',
'geoNetwork_region_trafficSource_source',
'geoNetwork_subContinent_device_browser',
'geoNetwork_subContinent_device_deviceCategory',
'geoNetwork_subContinent_device_operatingSystem',
'geoNetwork_subContinent_trafficSource_source',
'medium_source'             
]

for n_fold, (train_idx, valid_idx) in enumerate(folds):
    
# target mean
    df_train = train_df.iloc[train_idx]
    train_df_new,test_df_new = train_df,test_df
    for col in tqdm(cat_features):
        cols = [{'groupby': [col], 'func':'targetmean'}]
        train_df_new,test_df_new = target_mean(df_train,train_df_new,test_df_new,cols)

#frequency
    for col in tqdm(cat_features):
        train_df_new,test_df_new = frequency_encoding(df_train,train_df_new,test_df_new,col)

    
    feats = [f for f in train_df_new.columns if f not in drop_features]
    print ('ALL feats:' + str(len(feats)))
    train_x, train_y = train_df_new[feats].iloc[train_idx], train_df_new['totals_transactionRevenue'].iloc[train_idx]
    valid_x, valid_y = train_df_new[feats].iloc[valid_idx], train_df_new['totals_transactionRevenue'].iloc[valid_idx]
 
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
               "objective" : "regression", 
               "boosting" : "gbdt", 
               "metric" : "rmse",  #None
               "max_depth":8, 
               "reg_alpha": 30,
               "reg_lambda": 150,
               "num_leaves" : 128, 
               "max_bin" : 255, 
               "learning_rate" : 0.02, 
               "subsample" : 0.8, 
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
            params, dtrain, num_boost_round=10000,
            valid_sets=[dval], early_stopping_rounds=300, verbose_eval=100,)#feval = evalerror
        
        new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
        for item in new_list:
            print (item) 

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)

        sub_preds += np.expm1(bst.predict(test_df_new[feats], num_iteration=bst.best_iteration)) / len(folds) #folds.n_splits test_df_new

oof_preds[oof_preds < 0] = 0 
sub_preds[sub_preds < 0] = 0 

oof = pd.DataFrame({"fullVisitorId":train_df["fullVisitorId"].values})
oof["transactionRevenue"] = np.expm1(train_df["totals_transactionRevenue"])
oof["PredictedRevenue"] = np.expm1(oof_preds)
oof = oof.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()

cv = rmse1(train_df['totals_transactionRevenue'],  oof_preds)
print('Full RMSE %.6f' % cv)  
sum_cv = rmse2(np.log1p(oof["transactionRevenue"]),  np.log1p(oof["PredictedRevenue"]),  oof.shape[0])
print('SUM RMSE %.6f' % sum_cv)

# userbase
%%time
# create new target with expm1
train_df_new['predictions'] = np.expm1(oof_preds)
test_df_new['predictions'] = sub_preds

# Aggregate data at User level
trn_data = train_df_new[feats + ['fullVisitorId']].groupby('fullVisitorId').mean()

# Create a list of predictions for each Visitor
trn_pred_list = train_df_new[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
    
# Create a DataFrame with VisitorId as index
# trn_pred_list contains dict 
# so creating a dataframe from it will expand dict values into columns
trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
trn_feats = trn_all_predictions.columns

# for i in range(277):
#     tmp = 0
#     tmp += trn_all_predictions['pred_' + str(i)] 
#     trn_all_predictions['pred_sum_0'] = 0
#     trn_all_predictions['pred_sum_' + str(i+1)] = trn_all_predictions['pred_sum_' + str(i)] + tmp
#     print (trn_all_predictions.shape)
    
trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
del trn_data, trn_all_predictions
gc.collect()    

sub_pred_list = test_df_new[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
    
sub_data = test_df_new[feats + ['fullVisitorId']].groupby('fullVisitorId').mean()
sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)
for f in trn_feats:
    if f not in sub_all_predictions.columns:
        sub_all_predictions[f] = np.nan

# for i in range(277):
#     tmp = 0
#     tmp += sub_all_predictions['pred_' + str(i)] 
#     sub_all_predictions['pred_sum_0'] = 0
#     sub_all_predictions['pred_sum_' + str(i+1)] = sub_all_predictions['pred_sum_' + str(i)] + tmp
#     print (sub_all_predictions.shape)        
sub_all_predictions['t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1))
sub_all_predictions['t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1))
sub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1)
sub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1))
sub_all_predictions['t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1)
sub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)
del sub_data, sub_all_predictions
gc.collect()    


%%time
# create user based model
train_df['target'] = np.expm1(train_df['totals_transactionRevenue'])
trn_user_target = np.log1p(train_df[['fullVisitorId', 'target']].groupby('fullVisitorId').sum())

# Create arrays and dataframes to store results
oof_preds = np.zeros(full_data.shape[0])
sub_preds = np.zeros(sub_full_data.shape[0])
feature_importance_df = pd.DataFrame()
drop_features=[]
feats = [f for f in full_data.columns if f not in drop_features]
# for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, train_df['totals_transactionRevenue'],groups=train_df['fullVisitorId'])):
#     train_x, train_y = X[train_idx], train_df['totals_transactionRevenue'].iloc[train_idx]
#     valid_x, valid_y = X[valid_idx], train_df['totals_transactionRevenue'].iloc[valid_idx]
folds = get_folds(df=full_data[['totals_pageviews']].reset_index(), n_splits=5)
for n_fold, (train_idx, valid_idx) in enumerate(folds):
#for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['totals_transactionRevenue'],groups=train_df['fullVisitorId'])):
    train_x, train_y = full_data[feats].iloc[train_idx], trn_user_target['target'].iloc[train_idx]
    valid_x, valid_y = full_data[feats].iloc[valid_idx], trn_user_target['target'].iloc[valid_idx]    
    print("Train Index:",train_idx,",Val Index:",valid_idx)


    params = {
               "objective" : "regression", 
               "boosting" : "gbdt", 
               "metric" : "rmse",  #None
               #"max_depth": 12, 
               "num_leaves" : 33, #
               "reg_alpha": 30,
               "reg_lambda": 150,
               "max_bin" : 300, 
               "learning_rate" : 0.02, 
               "subsample" : 0.9, 
               "colsample_bytree" : 0.4, 
               "verbosity": -1
    }       


    if n_fold >= 0:
        evals_result = {}
        dtrain = lgb.Dataset(
            train_x, label=train_y,)#feature_name=tfvocab categorcical_feature=lgb_cat
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain,)
        bst = lgb.train(
            params, dtrain, num_boost_round=10000,
            valid_sets=[dval], early_stopping_rounds=300, verbose_eval=100,)#feval = evalerror
        
#         new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
#         for item in new_list:
#             print (item) 

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)

        sub_preds += bst.predict(sub_full_data[feats], num_iteration=bst.best_iteration) / len(folds) #folds.n_splits

oof_preds[oof_preds < 0] = 0     
cv = rmse1(trn_user_target['target'],  oof_preds)
print('Full RMSE %.6f' % cv)  


sub_preds[sub_preds < 0] = 0   
sub_full_data['PredictedLogRevenue'] = sub_preds
sub_full_data[['PredictedLogRevenue']].to_csv('../model/lgb_' + str(cv) + '.csv', index=True)

%%time
##########binary classification##########
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

train_df = df[df['bin'].notnull()].reset_index()
test_df = df[df['bin'].isnull()].reset_index()
print("Starting Session based LightGBM. Train shape: {}, test shape: {}".format(train_df.shape,test_df.shape))
folds = get_folds(df=train_df, n_splits=5)


# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])

sub_preds = np.zeros(test_df.shape[0])

feature_importance_df = pd.DataFrame()

drop_features=['date', 'fullVisitorId', 'sessionId', 'visitdate','yearmonth','yearweek','yearquarter','index','bin','local_time',
               'fullVisitorId_max_visitStartTime','fullVisitorId_min_visitStartTime',
               'visitId', 'visitStartTime', 'totals_transactionRevenue']
feats = [f for f in train_df.columns if f not in drop_features]
cat_features=[]
print ('feats:' + str(len(feats)))


for n_fold, (train_idx, valid_idx) in enumerate(folds):
#for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['totals_transactionRevenue'],groups=train_df['fullVisitorId'])):

# target encoding
#     df_train = train_df.iloc[train_idx]
    
#     cols = [{'groupby': ['trafficSource_referralPath',], 'func':'woe'},]
#     train_df_new,test_df_new = target_mean(df_train,train_df,test_df,cols)
 
#     feats = [f for f in train_df_new.columns if f not in drop_features]
#     train_x, train_y = train_df_new[feats].iloc[train_idx], train_df_new['totals_transactionRevenue'].iloc[train_idx]
#     valid_x, valid_y = train_df_new[feats].iloc[valid_idx], train_df_new['totals_transactionRevenue'].iloc[valid_idx]    
    
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['bin'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['bin'].iloc[valid_idx]    
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
               "objective" : "binary", 
               "boosting" : "gbdt", 
               "metric" : "auc",  #None
               "max_depth":10, 
               #"min_child_samples": 20, 
               "reg_alpha": 10,
               "reg_lambda": 10,
               "num_leaves" : 128, 
               "num_bin" : 255, 
               "learning_rate" : 0.02, 
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
            params, dtrain, num_boost_round=10000,
            valid_sets=[dval], early_stopping_rounds=300, verbose_eval=100,)#feval = evalerror
        
        new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
        for item in new_list:
            print (item) 

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)

        sub_preds += bst.predict(test_df[feats], num_iteration=bst.best_iteration) / len(folds) #folds.n_splits test_df_new

oof_preds[oof_preds < 0] = 0 
sub_preds[sub_preds < 0] = 0 

from sklearn.metrics import mean_squared_error, roc_auc_score
print('AUC : %.6f' % (roc_auc_score(train_df['bin'],  oof_preds)))
