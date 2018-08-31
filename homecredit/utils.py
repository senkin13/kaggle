import numpy as np
import pandas as pd
import pickle
import gc
import time
import lightgbm as lgb
from tqdm import tqdm
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def drop_same_cols(df1,df2):
    for col in df1.columns:
        if col in df2.columns and col != 'SK_ID_CURR' :
            df2.drop([col],axis=1,inplace=True)
            
def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)
 
        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break

    return dups
    
from scipy.stats import ranksums

def corr_feature_with_target(feature, target):
    c0 = feature[target == 0].dropna()
    c1 = feature[target == 1].dropna()
        
    if set(feature.unique()) == set([0, 1]):
        diff = abs(c0.mean(axis = 0) - c1.mean(axis = 0))
    else:
        #diff = abs(c0.median(axis = 0) - c1.median(axis = 0))
        diff = abs(c0.mean(axis = 0) - c1.mean(axis = 0))        
    p = ranksums(c0, c1)[1] if ((len(c0) >= 20) & (len(c1) >= 20)) else 2
        
    return [diff, p]

def clean_data(data):
    warnings.simplefilter(action = 'ignore')
    
    # Removing empty features
    nun = data.nunique()
    empty = list(nun[nun <= 1].index)
    
    data.drop(empty, axis = 1, inplace = True)
    print('After removing empty features there are {0:d} features'.format(data.shape[1]))
    
    # Removing features with the same distribution on 0 and 1 classes
#     corr = pd.DataFrame(index = ['diff', 'p'])
#     ind = data[data['TARGET'].notnull()].index
    
#     for c in data.columns.drop('TARGET'):
#         corr[c] = corr_feature_with_target(data.loc[ind, c], data.loc[ind, 'TARGET'])

#     corr = corr.T
#     corr['diff_norm'] = abs(corr['diff'] / data.mean(axis = 0))
    
#     to_del_1 = corr[((corr['diff'] == 0) & (corr['p'] > .05))].index
#     to_del_2 = corr[((corr['diff_norm'] < .5) & (corr['p'] > .05))].drop(to_del_1).index
#     to_del = list(to_del_1) + list(to_del_2)
#     if 'SK_ID_CURR' in to_del:
#         to_del.remove('SK_ID_CURR')
        
#     data.drop(to_del, axis = 1, inplace = True)
#     print('After removing features with the same distribution on 0 and 1 classes there are {0:d} features'.format(data.shape[1]))
    
    
from sklearn.metrics import log_loss

def log_loss(true_y,pred_h):
    return -np.mean(true_y*np.log(pred_h)+(1-true_y)*np.log(1-pred_h))

log_loss(train_df['TARGET'],oof_preds)

def get_rank(x):
    return pd.Series(x).rank(pct=True).values
    
# get feature dtype
binary_feature = df.select_dtypes(include=['uint8']).columns.values
float_feature = df.select_dtypes(include=['float']).columns.values
int_feature = df.select_dtypes(include=['integer']).select_dtypes(exclude=['uint8']).columns.values

feature_gain_max = feature_importance_df.groupby(['feature'])['gain'].max().reset_index()

gain001 = feature_gain_max[feature_gain_max['gain'] < 0.001].feature.values
np.save('../features/_gain001.npy', gain001)

feature_gain_mean = feature_importance_df.groupby(['feature'])['gain'].mean().reset_index()

gain001mean = feature_gain_mean[feature_gain_mean['gain'] < 0.001].feature.values
np.save('../features/_gain001mean.npy', gain001mean)

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
    
import gc
import time
from time import strftime,gmtime
import numpy as np
import pandas as pd
import os
load = True
from sklearn.decomposition import TruncatedSVD,PCA
cache_path = '../cache/'

from time import strftime,gmtime
#reload(utils)


def concat(L):
    result = None
    for l in L:
#        print(l.columns.tolist())
#        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))        
#        print("done")
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except:
                print(l.head())
    return result


def left_merge(data1,data2,on):
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp,on=on,how='left')
    result = result[columns]
    return result


def rank(data, feat1, feat2, ascending=True):
    data.sort_values(feat1 + feat2, inplace=True, ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)['rank'].agg({'min_rank': 'min','max_rank':'max'})
    data = pd.merge(data, min_rank, on=feat1, how='left')
    data['rank'] = (data['rank'] - data['min_rank'])/(data['max_rank']-data['min_rank'])
    del data['min_rank'], data['max_rank']
    return data['rank']



def get_feat_size(train,size_feat):
    """计算A组的数量大小（忽略NaN等价于count）"""
    result_path = cache_path +  ('_').join(size_feat)+'_count'+'.csv'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
#        print("get_size_feat" + result_path)
        result = train[size_feat].groupby(by=size_feat).size().reset_index().rename(columns={0: ('_').join(size_feat)+'_count'})
        result = left_merge(train,result,on=size_feat)
#        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_feat_size_feat(train,base_feat,other_feat):
    """计算唯一计数（等价于unique count）"""
    result_path = cache_path + ('_').join(base_feat)+'_count_'+('_').join(other_feat)+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
#        print("get_size_feat_size" + result_path)
        result = train[base_feat].groupby(base_feat).size().reset_index()\
                      .groupby(other_feat).size().reset_index().rename(columns={0: ('_').join(base_feat)+'_count_'+('_').join(other_feat)})
        result = left_merge(train,result,on=other_feat)
#        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_feat_stat_feat(train,base_feat,other_feat,stat_list=['min','max','var','size','mean','skew']):
    result_path = cache_path + ('_').join(base_feat)+'_'+('_').join(stat_list)+'_'+('_').join(other_feat)+'.hdf'
    name = ('_').join(base_feat) +'_'+('_').join(other_feat)+'_'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
#        print("get_stat" + result_path)
        agg_dict = {}
        for stat in stat_list:
            agg_dict[name+stat] = stat
        result = train[base_feat + other_feat].groupby(base_feat)[",".join(other_feat)]\
        .agg(agg_dict)
        result = left_merge(train,result,on=base_feat)
#        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result
    
def get_last_feat3(train_base,data,dup_feat,base_feat,time_col,ascending=False,prefix=''):
    """计算A组的数量大小（忽略NaN等价于count）"""
    result_path = cache_path + prefix+ 'last_feat'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
#        print("get_size_feat" + result_path)
        result = get_last_n_data(data,dup_feat,time_col,n=1,ascending=ascending)[base_feat]
        result.columns = [prefix+'_last_'+column if column not in dup_feat else column for column in result.columns ]
        result = left_merge(train_base,result,on=dup_feat)
#        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_last_n_data(data,dup_feat,time_col,n=1,ascending=False):
    data = data.sort_values(time_col,ascending=ascending)
    data = data.groupby(by=dup_feat).head(n)
    return data




def get_next_time(train,base_feat,time_col,cat='next',N=1):
    if cat=='next':
        shiftnum = -N
    elif cat=='last':
        shiftnum = 0
    name = ('_').join(base_feat)+'_next_click'+'_' + cat +str(N)
    result_path = cache_path + ('_').join(base_feat)+'_next_click'+'_'+cat+str(N)+'.hdf'
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path)
    else:
        train = train[base_feat+[time_col]] 
        train['category'] = train.groupby(base_feat).ngroup().astype('uint32')
        train.sort_values(['category',time_col],inplace=True)
        train['catdiff'] = train.category.diff(N).shift(shiftnum).fillna(1).astype('uint8')
        train.drop(['category'],axis=1,inplace=True)
        train[name] = train[time_col].diff(N).shift(shiftnum)
        train[name] = np.clip(train[name],0,10800) # train have only three hour a day
        train.loc[train.catdiff!=0, name] = np.nan
        train.sort_index(inplace=True)
        result = train[[name]]
        train.drop(['catdiff',name],axis=1,inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result
    
from multiprocessing import Pool
WORKER_NUM = 32


def split_df(df, n_chunck=8):
    cols = df.columns.tolist()
    col_num_per_chunck = len(cols)//n_chunck
    retval = []
    for i in range(n_chunck):
        if i==n_chunck-1:
            temp_cols = cols[i*col_num_per_chunck:]
        else:
            temp_cols = cols[i*col_num_per_chunck:(i+1)*col_num_per_chunck]
        retval.append(df[temp_cols])
    return retval

def fill_na(base):
    for c in base.columns.tolist():
        if c=='SK_ID_CURR':
            continue
        min_value = base.loc[~np.isinf(base[c]),c].min()
        max_value = base.loc[~np.isinf(base[c]),c].max()

        if min_value>=0:
            base[c].replace([np.inf],-999999,inplace=True)
        else:
            base[c].replace([np.inf], max(999999,base.loc[~np.isinf(base[c]),c].max()+100),inplace=True)
        if max_value<=0:
            base[c].replace([-np.inf], 999999, inplace=True)
        else:
            base[c].replace([-np.inf], min(-999999,base.loc[~np.isinf(base[c]),c].min()-100),inplace=True)

        base[c] = base[c].fillna(-999999).astype(np.float32)
        gc.collect()
    return base

x = split_df(df, n_chunck=WORKER_NUM) # 分割df
pool = Pool(processes=WORKER_NUM) # 建pool
outputs = pool.map(fill_na, x)
output_df = pd.concat(outputs,axis=1)

def missing_columns(dataframe):
    """
    Returns a dataframe that contains missing column names and 
    percent of missing values in relation to the whole dataframe.
    
    dataframe: dataframe that gives the column names and their % of missing values
    """
    
    # find the missing values
    missing_values = dataframe.isnull().sum().sort_values(ascending=False)
    
    # percentage of missing values in relation to the overall size
    missing_values_pct = 100 * missing_values/len(dataframe)
    
    # create a new dataframe which is a concatinated version
    concat_values = pd.concat([missing_values, missing_values/len(dataframe),missing_values_pct.round(1)],axis=1)

    # give new col names
    concat_values.columns = ['Missing Count','Missing Count Ratio','Missing Count %']
    
    # return the required values
    return concat_values[concat_values.iloc[:,1]!=0]

missing_columns(df)


    
    
