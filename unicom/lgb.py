%%time

import pandas as pd
import numpy as np
import gc
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

before = pd.read_csv('../input/train_all.csv')
print ('before shape:' + str(before.shape))

train2 = pd.read_csv('../input/train_2.csv') #new
train = train2

test2 = pd.read_csv('../input/test_2.csv') #new
test = test2

%%time

# 对标签编码 映射关系
label2current_service = dict(zip(range(0,len(set(before['current_service']))),sorted(list(set(before['current_service'])))))
current_service2label = dict(zip(sorted(list(set(before['current_service']))),range(0,len(set(before['current_service'])))))

# 原始数据的标签映射
before['current_service'] = before['current_service'].map(current_service2label)


before['2_total_fee'] = pd.to_numeric(before['2_total_fee'].replace('\\N','-1'))
before['3_total_fee'] = pd.to_numeric(before['3_total_fee'].replace('\\N','-1'))
before['age'] = pd.to_numeric(before['age'].replace('\\N','-1'))
before['gender'] = pd.to_numeric(before['gender'].replace('\\N','-1'))

print ('before shape:' + str(before.shape))


# binning

before['age_bin'] = before['age'].map(lambda x:10 if x<=11 and x>3 else 78 if x>77 else x)
## 1024/100 
before['month_traffic_1024'] = before['month_traffic'].map(lambda x:1 if x==1024 or x==2048 or x==3072
                                                     or x==4096  else 0)
# before['month_traffic_00'] = before['month_traffic'].map(lambda x:1 if x==100 
#                                                    or x==200 
#                                                    or x==300 
#                                                    or x==400 
#                                                    or x==500 
#                                                    or x==600 
#                                                    or x==700 
#                                                    or x==800 
#                                                    or x==900 
#                                                    or x==1000 
#                                                    else 0)

## demical 
# before['1_total_fee_digit1'] = (df1['1_total_fee']*1%10).astype('int64')
# before['2_total_fee_digit1'] = (df1['2_total_fee']*1%10).astype('int64')
# before['3_total_fee_digit1'] = (df1['3_total_fee']*1%10).astype('int64')
# before['4_total_fee_digit1'] = (df1['4_total_fee']*1%10).astype('int64')

before['1_total_fee_demical1'] = (before['1_total_fee']*10%10).astype('int64')
before['2_total_fee_demical1'] = (before['2_total_fee']*10%10).astype('int64')
before['3_total_fee_demical1'] = (before['3_total_fee']*10%10).astype('int64')
before['4_total_fee_demical1'] = (before['4_total_fee']*10%10).astype('int64')

before['1_total_fee_demical2'] = (before['1_total_fee']*100%10).astype('int64')
before['2_total_fee_demical2'] = (before['2_total_fee']*100%10).astype('int64')
before['3_total_fee_demical2'] = (before['3_total_fee']*100%10).astype('int64')
before['4_total_fee_demical2'] = (before['4_total_fee']*100%10).astype('int64')

# before['1_total_fee_demical12'] = (before['1_total_fee']*100%100).astype('int64')
# before['2_total_fee_demical12'] = (before['2_total_fee']*100%100).astype('int64')
# before['3_total_fee_demical12'] = (before['3_total_fee']*100%100).astype('int64')
# before['4_total_fee_demical12'] = (before['4_total_fee']*100%100).astype('int64')

before['1_total_fee_demical3'] = (before['1_total_fee']*100%15).astype('int64')
before['2_total_fee_demical3'] = (before['2_total_fee']*100%15).astype('int64')
before['3_total_fee_demical3'] = (before['3_total_fee']*100%15).astype('int64')
before['4_total_fee_demical3'] = (before['4_total_fee']*100%15).astype('int64')

## min
before['min_total_fee'] = before[['1_total_fee','2_total_fee','3_total_fee','4_total_fee']].min(axis=1)
#min_total_fee - 1_total_fee

## ratio
before['month_traffic_fee_ratio'] = (before['1_total_fee'] / (before['month_traffic']+1))
before['caller_time_ratio'] = (before['service1_caller_time'] / (before['service2_caller_time']+1))
## diff
before['diff_month_traffic'] = before['month_traffic'] - before['local_trafffic_month']
before['diff_caller_time'] = before['service1_caller_time'] - before['service2_caller_time']


## aggregate features
from tqdm import tqdm
def agg(df,agg_cols):
    for c in tqdm(agg_cols):
        new_feature = '{}_{}_{}'.format('_'.join(c['groupby']), c['agg'], c['target'])
        gp = df.groupby(c['groupby'])[c['target']].agg(c['agg']).reset_index().rename(index=str, columns={c['target']:new_feature})
        df = df.merge(gp,on=c['groupby'],how='left')
    return df

agg_cols = [
############################unique aggregation##################################
    {'groupby': ['is_mix_service','is_promise_low_consume','many_over_bill'], 'target':'contract_type', 'agg':'nunique'},
    {'groupby': ['pay_times','is_mix_service','is_promise_low_consume','many_over_bill'], 'target':'contract_type', 'agg':'nunique'},
    {'groupby': ['age_bin','gender'], 'target':'contract_type', 'agg':'nunique'},
############################count aggregation##################################  
    
############################mean/median/sum/min/max aggregation##################################        
    {'groupby': ['contract_time','contract_type'], 'target':'1_total_fee', 'agg':'mean'},
    {'groupby': ['contract_time','contract_type'], 'target':'1_total_fee', 'agg':'max'},

    
    {'groupby': ['is_mix_service','is_promise_low_consume','many_over_bill'], 'target':'2_total_fee', 'agg':'mean'},    

]

before = agg(before,agg_cols)

before['diff_max_1_total_fee'] = before['1_total_fee'] - before['contract_time_contract_type_max_1_total_fee']

%%time
# encoding=utf8
import pandas as pd
import lightgbm as lgb
import re
import time
import numpy as np
import math
import gc
import pickle
import os
from sklearn.metrics import roc_auc_score, log_loss, f1_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
# from com_util import *
import lightgbm
from sklearn.model_selection import KFold, StratifiedKFold


def get_type_feature_all(sample, train_df, key, on, type_c, mark):
    filename = "_".join([mark + "_%s_features" % type_c, "_".join(key), on, str(len(sample))]) + ".pkl"
    try:
        with open("../pickle/" + filename, "rb") as fp:
            print("load {} {} feature from pickle file: key: {}, on: {}...".format(mark, type_c, "_".join(key), on))
            col = pickle.load(fp)
        for c in col.columns:
            sample[c] = col[c]
        gc.collect()
    except:
        sam = sample[key].copy()
        print('get {} {} feature, key: {}, on: {}'.format(mark, type_c, "_".join(key), on))
        if type_c == "count":
            tmp = pd.DataFrame(train_df[key + [on]].groupby(key)[on].count()).reset_index()
        if type_c == "mean":
            tmp = pd.DataFrame(train_df[key + [on]].groupby(key)[on].mean()).reset_index()
        if type_c == "nunique":
            tmp = pd.DataFrame(train_df[key + [on]].groupby(key)[on].nunique()).reset_index()
        if type_c == "max":
            tmp = pd.DataFrame(train_df[key + [on]].groupby(key)[on].max()).reset_index()
        if type_c == "min":
            tmp = pd.DataFrame(train_df[key + [on]].groupby(key)[on].min()).reset_index()
        if type_c == "sum":
            tmp = pd.DataFrame(train_df[key + [on]].groupby(key)[on].sum()).reset_index()
        if type_c == "std":
            tmp = pd.DataFrame(train_df[key + [on]].groupby(key)[on].std()).reset_index()
        if type_c == "median":
            tmp = pd.DataFrame(train_df[key + [on]].groupby(key)[on].median()).reset_index()
        tmp.columns = key + [mark + "_" + "_".join(key) + '_%s_' % type_c + on]
        tmp[mark + "_" + "_".join(key) + '_%s_' % type_c + on] = tmp[
            mark + "_" + "_".join(key) + '_%s_' % type_c + on].astype('float32')
        sam = sam.merge(tmp, on=key, how='left')
        with open("../pickle/" + filename, "wb") as fp:
            col = sam[[mark + "_" + "_".join(key) + '_%s_' % type_c + on]]
            pickle.dump(col, fp)
        del tmp
        with open("../pickle/" + filename, "rb") as fp:
            col = pickle.load(fp)
        for c in col.columns:
            sample[c] = col[c]
        gc.collect()

    del col, train_df
    gc.collect()
    return sample, mark + "_" + "_".join(key) + '_%s_' % type_c + on


def get_cat_feature(sample, train_df, key, column, value,type_c="mean"):
    new_column = column + "_" + str(value)
    sample_filename = "_".join(["cat_features", "_".join(key), column, str(value),type_c, str(len(sample))]) + ".pkl"
    try:
        with open("../pickle/" + sample_filename, "rb") as fp:
            print("load cat feature from pickle_sample_small file: key: {}, on: {}...".format("_".join(key),
                                                                                              column + "_" + str(
                                                                                                  value)))
            col = pickle.load(fp)
        for c in col.columns:
            sample[c] = col[c]
    except:
        print("get cat feature from pickle_sample_small file: key: {}, on: {}...".format("_".join(key),
                                                                                         column + "_" + str(value)))
        #df = train_df.copy()
        # df[new_column]=(df[column]+1.0)/(value+1.0)
        # df[df[new_column]!=1]=0
        try:
            train_df[new_column] = train_df[column].apply(lambda x: 1 if value in x else 0)
        except:
            train_df[new_column] = train_df[column].apply(lambda x: 1 if value == x else 0)
        if type_c=="mean":
            gp = pd.DataFrame(train_df.groupby(key)[new_column].mean()).reset_index()
        if type_c=="sum":
            gp = pd.DataFrame(train_df.groupby(key)[new_column].sum()).reset_index()
        del train_df[new_column]
        gc.collect()
        gp.columns = key + ["cat_features_" + "_".join(key) + "_"+type_c+"_"+ new_column]
        # print(gp.head())
        sample = sample.merge(gp, on=key, how="left").fillna(0)
        with open("../pickle/" + sample_filename, "wb") as fp:
            col = sample[["cat_features_" + "_".join(key) + "_"+type_c+"_"+ new_column]]
            pickle.dump(col, fp)
        del gp
        gc.collect()
    del train_df
    gc.collect()
    return sample,"cat_features_" + "_".join(key) + "_"+type_c+"_"+ new_column




def encode_onehot(df, column_name):
    feature_df = pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1), feature_df], axis=1)
    return all


def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds=preds.reshape(11, -1)
    weight=np.array([[0.8663461245074525,1.0050158436657424,1.0469212189712187,1.000289693798254,0.9992626174350012,0.9963578420213977,1.0032155232533406,0.9150234670407598,1.0418515160085273,0.9886697369147659,1.4654870560708395]])
    #preds=preds*weight.T
    preds = np.argmax(preds, axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, labels=range(0, 11), average='macro') ** 2
    return 'f1_score', score_vali, True


lb = LabelEncoder()


features = pd.read_csv("../input/train_all.csv")
features['2_total_fee'] = pd.to_numeric(features['2_total_fee'].replace('\\N', '-9999'))
features['3_total_fee'] = pd.to_numeric(features['3_total_fee'].replace('\\N', '-9999'))
features['1_total_fee_demical1'] = (features['1_total_fee'] * 10 % 10)  # .astype('int64')
features['2_total_fee_demical1'] = (features['2_total_fee'] * 10 % 10)  # .astype('int64')
features['3_total_fee_demical1'] = (features['3_total_fee'] * 10 % 10)  # .astype('int64')
features['4_total_fee_demical1'] = (features['4_total_fee'] * 10 % 10)  # .astype('int64')

features['1_total_fee_demical2'] = (features['1_total_fee'] * 100 % 10)  # .astype('int64')
features['2_total_fee_demical2'] = (features['2_total_fee'] * 100 % 10)  # .astype('int64')
features['3_total_fee_demical2'] = (features['3_total_fee'] * 100 % 10)  # .astype('int64')
features['4_total_fee_demical2'] = (features['4_total_fee'] * 100 % 10)  # .astype('int64')

features['1_total_fee_demical3'] = (features['1_total_fee'] * 100 % 15)  # .astype('int64')
features['2_total_fee_demical3'] = (features['2_total_fee'] * 100 % 15)  # .astype('int64')
features['3_total_fee_demical3'] = (features['3_total_fee'] * 100 % 15)  # .astype('int64')
features['4_total_fee_demical3'] = (features['4_total_fee'] * 100 % 15)  # .astype('int64')

## demical
before['1_total_fee_v1']=before['1_total_fee']+before['pay_num']%1
before['1_total_fee_v2']=before['1_total_fee']+before['pay_num']%10


before['1_total_fee_demical4'] = (before['1_total_fee'] % 10).astype('int64')
before['2_total_fee_demical4'] = (before['2_total_fee'] % 10).astype('int64')
before['3_total_fee_demical4'] = (before['3_total_fee'] % 10).astype('int64')
before['4_total_fee_demical4'] = (before['4_total_fee'] % 10).astype('int64')



before['1_total_fee_demical1_a']=before['1_total_fee_demical1'] /before['1_total_fee']
before['2_total_fee_demical1_a']=before['2_total_fee_demical1'] /before['2_total_fee']
before['3_total_fee_demical1_a']=before['3_total_fee_demical1'] /before['3_total_fee']
before['4_total_fee_demical1_a']=before['4_total_fee_demical1'] /before['4_total_fee']


before['last_month_traffic_mod_500'] = before['last_month_traffic'] % 500
before['last_month_traffic_mod_1024'] = before['last_month_traffic'] % 1024
before['last_month_traffic_mod_100'] = before['last_month_traffic'] % 100
before['contract_time_mod_12'] = before['contract_time'] % 12
before['pay_num_mod'] = before['pay_num'] % 10
before['pay_num_mod_1'] = before['pay_num'] % 1


# for i in list(features.current_service.value_counts().index):
#     before,_=get_cat_feature(before, features, ['contract_type','is_mix_service', 'is_promise_low_consume', 'many_over_bill'], "current_service", i, type_c="mean")
#     before,_=get_cat_feature(before, features, ['contract_type','is_mix_service', 'is_promise_low_consume', 'many_over_bill'], "current_service", i, type_c="sum")
    #before,_=get_cat_feature(before, features, ['1_total_fee_demical1','1_total_fee_demical2', '1_total_fee_demical3'], "current_service", i, type_c="mean")
    #before,_=get_cat_feature(before, features, ['2_total_fee_demical1','2_total_fee_demical2', '2_total_fee_demical3'], "current_service", i, type_c="mean")


for i in ['1_total_fee_demical1','2_total_fee_demical1','3_total_fee_demical1','4_total_fee_demical1',
              '1_total_fee_demical2','2_total_fee_demical2','3_total_fee_demical2','4_total_fee_demical2',
              '1_total_fee_demical3','2_total_fee_demical3','3_total_fee_demical3','4_total_fee_demical3']:
    #before, _ = get_type_feature_all(before, before, [i], "user_id", "count", "auto_1")
    before, _ = get_type_feature_all(before, before, [i], "local_trafffic_month", "mean", "auto_1")
    #before, _ = get_type_feature_all(before, before, [i], "month_traffic", "mean", "auto_1")
    #before, _ = get_type_feature_all(before, before, [i], "month_traffic", "mean", "auto_1")
before, f1 = get_type_feature_all(before, before, ["1_total_fee"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["1_total_fee"], "2_total_fee", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
before, f1 = get_type_feature_all(before, before, ["2_total_fee"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["2_total_fee"], "3_total_fee", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
before, f1 = get_type_feature_all(before, before, ["3_total_fee"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["3_total_fee"], "4_total_fee", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
before, f1 = get_type_feature_all(before, before, ["4_total_fee"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["4_total_fee"], "1_total_fee", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]


before, f1 = get_type_feature_all(before, before, ["1_total_fee_demical1"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["1_total_fee_demical1"], "2_total_fee_demical1", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
before, f1 = get_type_feature_all(before, before, ["2_total_fee_demical1"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["2_total_fee_demical1"], "3_total_fee_demical1", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
before, f1 = get_type_feature_all(before, before, ["3_total_fee_demical1"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["3_total_fee_demical1"], "4_total_fee_demical1", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
before, f1 = get_type_feature_all(before, before, ["4_total_fee_demical1"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["4_total_fee_demical1"], "1_total_fee_demical1", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]


before, f1 = get_type_feature_all(before, before, ["1_total_fee_demical3"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["1_total_fee_demical3"], "2_total_fee_demical3", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
before, f1 = get_type_feature_all(before, before, ["2_total_fee_demical3"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["2_total_fee_demical3"], "3_total_fee_demical3", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
before, f1 = get_type_feature_all(before, before, ["3_total_fee_demical3"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["3_total_fee_demical3"], "4_total_fee_demical3", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
before, f1 = get_type_feature_all(before, before, ["4_total_fee_demical3"], "user_id", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["4_total_fee_demical3"], "1_total_fee_demical3", "nunique", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]


before['maxmin_total_fee'] = before[['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']].max(axis=1) - before[
        ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']].min(axis=1)

    # before['min_total_fee_demical1'] = (before['min_total_fee'] * 10 % 10)#.astype('int64')
    # before['min_total_fee_demical2'] = (before['min_total_fee'] * 100 % 10)#.astype('int64')
    # before['min_total_fee_demical3'] = (before['min_total_fee'] * 100 % 15)#.astype('int64')



    ## ratio

before['fee_local_caller_time_ratio'] = (before['1_total_fee'] / (before['local_caller_time'] + 1))

before['local_caller_time_ratio'] = (before['service1_caller_time'] / (before['local_caller_time'] + 1))

before['pay_num_fee_ratio'] = (before['pay_num'] / (before['1_total_fee'] + before['2_total_fee']))

    #before['12_34_rt'] = (before['3_total_fee']+ before['4_total_fee'])/ (before['1_total_fee'] + before['2_total_fee'])
    #before['1_234_rt'] = (before['3_total_fee']+ before['4_total_fee'] + before['2_total_fee'])/ (before['1_total_fee'])
    #before['123_4_rt'] = (before['4_total_fee'])/ (before['3_total_fee']+ before['1_total_fee'] + before['2_total_fee'])

def correl_cos(y_true, y_pred):
    X = pd.Series(y_true).astype(np.float32)
    Y = pd.Series(y_pred).astype(np.float32)
    num = X.dot(Y)
    denom = np.linalg.norm(X) * np.linalg.norm(Y)
    cos = num / denom  # 余弦值
    return cos

    #before['cos']=list(map(lambda a,b,c,d:correl_cos([a,b,c],[b,c,d]),before["1_total_fee"],before["2_total_fee"],before["3_total_fee"],before["4_total_fee"]))

## diff

before['last_month_traffic_dif_2'] = 2*before['last_month_traffic'] - before['local_trafffic_month']



before['return__num_20'] = before['pay_num'] % before['pay_times'] * 20
before['return__num_30'] = before['pay_num'] % before['pay_times'] * 30
before['return__num_50'] = before['pay_num'] % before['pay_times'] * 50
before['return__num_100'] = before['pay_num'] % before['pay_times'] * 100
before['return__num_300'] = before['pay_num'] % before['pay_times'] * 300

before['return_num_20'] = before['pay_num'] - before['pay_times'] * 20
before['return_num_30'] = before['pay_num'] - before['pay_times'] * 30
before['return_num_50'] = before['pay_num'] - before['pay_times'] * 50
before['return_num_100'] = before['pay_num'] - before['pay_times'] * 100
before['return_num_300'] = before['pay_num'] - before['pay_times'] * 300


cat_features = [
    ["contract_type"],
    ["service_type", "many_over_bill"],
    ["service_type", "gender"],
        # ["is_mix_service", "many_over_bill","is_promise_low_consume","contract_type","service_type"],
        # ["is_mix_service", "many_over_bill","is_promise_low_consume","contract_type","service_type","month_traffic_00","month_traffic_1204"],
]
num_features = ["1_total_fee"] # "2_total_fee", "3_total_fee",]
                
for cat in cat_features:
    for num_c in num_features:
        for t in ["mean"]:
            before, _ = get_type_feature_all(before, before, cat, num_c, t, "auto_1")
            # before["%s_%s_%s_rt" % (num_c, _, t)] = before[num_c] / (0.000 + before[_])
            gc.collect()



before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "1_total_fee", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","gender"], "1_total_fee", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "1_total_fee", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "1_total_fee", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "1_total_fee", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type","contract_time"], "1_total_fee", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ['is_mix_service',"service_type", "many_over_bill","contract_type"], "1_total_fee", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ['is_mix_service',"service_type", "many_over_bill","contract_type","contract_time"], "1_total_fee", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ['is_promise_low_consume','is_mix_service',"service_type", "many_over_bill","contract_type"], "1_total_fee", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ['is_promise_low_consume','is_mix_service',"service_type", "many_over_bill","contract_type","contract_time"], "1_total_fee", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ['net_service','is_promise_low_consume','is_mix_service',"service_type", "many_over_bill","contract_type"], "1_total_fee", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ['net_service','is_promise_low_consume','is_mix_service',"service_type", "many_over_bill","contract_type","contract_time"], "1_total_fee", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]
####
before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "1_total_fee", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","gender"], "1_total_fee", "count", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "1_total_fee", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "1_total_fee", "count", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "1_total_fee", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type","contract_time"], "1_total_fee", "count", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ['is_mix_service',"service_type", "many_over_bill","contract_type"], "1_total_fee", "count", "auto_1")
before, f2 = get_type_feature_all(before, before, ['is_mix_service',"service_type", "many_over_bill","contract_type","contract_time"], "1_total_fee", "count", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]


before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "month_traffic", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","gender"], "month_traffic", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "month_traffic", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "month_traffic", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "month_traffic", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type","contract_time"], "month_traffic", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "local_caller_time", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","gender"], "local_caller_time", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "local_caller_time", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "local_caller_time", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "local_caller_time", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type","contract_time"], "local_caller_time", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "2_total_fee", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","gender"], "2_total_fee", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill"], "2_total_fee", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "2_total_fee", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]

before, f1 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type"], "2_total_fee", "mean", "auto_1")
before, f2 = get_type_feature_all(before, before, ["service_type", "many_over_bill","contract_type","contract_time"], "2_total_fee", "mean", "auto_1")
before['%s_%s'%(f2,f1)] = before[f2] /before[f1]
del before[f2],before[f1]


#pred_features = pd.read_csv("../metafeatures/baseline_0.8516512885157108_0.8516508428757271.csv")
#before = before.merge(pred_features, on="user_id", how="left")
#before["pre_class"] = np.argmax(before[["pred_%s" % i for i in range(11)]].values, axis=1)


before["1_total_fee_rank"]=before["1_total_fee"].rank()
before["2_total_fee_rank"]=before["2_total_fee"].rank()
before["3_total_fee_rank"]=before["3_total_fee"].rank()
before["4_total_fee_rank"]=before["4_total_fee"].rank()

before["local_trafffic_month_rank"]=before["local_trafffic_month"].rank()
before["local_caller_time_rank"]=before["local_caller_time"].rank()
before["service1_caller_time_rank"]=before["service1_caller_time"].rank()
before["service2_caller_time_rank"]=before["service2_caller_time"].rank()

import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import f1_score
from datetime import datetime

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)
    f_score = np.square(f1_score(labels , preds,   average = 'macro'))
    return 'f1_score', f_score, True

# train_df = df4[df4['current_service'].notnull()]
# test_df = df4[df4['current_service'].isnull()]

train_df = before
test_df = df4
print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape,test_df.shape))
#del df
gc.collect()

SEED = 2018
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Create arrays and dataframes to store results
oof_preds_prob = np.zeros((train_df.shape[0],11))
oof_preds = np.zeros(train_df.shape[0])

sub_preds_prob = np.zeros((test_df.shape[0],11))
sub_preds = np.zeros(test_df.shape[0])

preds_prob = np.zeros((test_df.shape[0],11))

feature_importance_df = pd.DataFrame()
drop_features=['current_service','user_id','service_type',
               ]
feats = [f for f in train_df.columns if f not in drop_features]
print ('feats:' + str(len(feats)))

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['current_service'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['current_service'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['current_service'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
        'nthread': 32,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 128,
        'max_bin': 784,
        'learning_rate': 0.05,
        'subsample': 1,
        'colsample_bytree': 0.4,
        'verbose': 1,
        'num_class': 11,
    }

    if n_fold >= 0:
        evals_result = {}
        print ('Fold' + str(n_fold+1))
        dtrain = lgb.Dataset(
            train_x, label=train_y)
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain)
        bst = lgb.train(
            params, dtrain, num_boost_round=10000,
            valid_sets=[dval], early_stopping_rounds=300, verbose_eval=100,
            feval = evalerror)
        
#         new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
#         for item in new_list:
#             print (item) 

        oof_preds_prob[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        oof_preds[valid_idx] = oof_preds_prob[valid_idx].argmax(axis = 1)
        print("F1-Score is", np.square(f1_score(valid_y,  oof_preds[valid_idx] , average = 'macro')))
        
        sub_preds_prob += bst.predict(test_df[feats], num_iteration=bst.best_iteration)

# save prob file
oof_preds_prob.dump('oof_trans_' + str(SEED) + '.pkl')
sub_preds_prob.dump('pred_trans_' + str(SEED) + '.pkl')

preds  = sub_preds_prob.argmax(axis = 1)
print('Full F1-Score %.6f' % np.square(f1_score(train_df['current_service'],  oof_preds , average = 'macro'))) 

# oof4 = pd.DataFrame({"user_id":train["user_id"], "predict":oof_preds})
# oof4['predict'] = oof4['predict'].map(label2current_service)

# preds4 = pd.DataFrame({"user_id":test["user_id"], "predict":preds})
# preds4['predict'] = preds4['predict'].map(label2current_service)
