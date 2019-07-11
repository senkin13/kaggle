import pandas as pd
import numpy as np
import gc
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv('../input/oppo_round1_train_20180929.txt',sep='\t', header=None,\
                       names=['prefix', 'query_prediction', 'title', 'tag', 'label'])
valid = pd.read_csv('../input/oppo_round1_vali_20180929.txt',sep='\t', header=None,\
                       names=['prefix', 'query_prediction', 'title', 'tag', 'label'])
test = pd.read_csv('../input/oppo_round1_test_A_20180929.txt',sep='\t', header=None,\
                       names=['prefix', 'query_prediction', 'title', 'tag'])

df = pd.concat([train,valid,test],axis=0)
df = df[df['label']!='音乐']
df.label = pd.to_numeric(df.label)
print ('df shape:' + str(df.shape))

#split query_prediction to list
def split_prediction(text):
    if pd.isna(text): return []
    return [s.strip() for s in \
            text.replace("{", "").replace("}", "").split(",")]
df['pred_list'] = df['query_prediction'].apply(split_prediction)
df['pred_len'] = df['pred_list'].apply(len)

from tqdm import tqdm

# lable encoder 
def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = LabelEncoder()
            df[c] = lbl.fit_transform(df[c].astype(str))
            print(c)
    return df

#df_all = df[['prefix','tag','title','pred_len','label']]
df_all = df[['prefix','tag','title','label']]
df_all = df_lbl_enc(df_all)

df_all['dummy'] = 1

def agg(df,agg_cols):
    for c in tqdm(agg_cols):
        new_feature = '{}_{}_{}'.format('_'.join(c['groupby']), c['agg'], c['target'])
        gp = df.groupby(c['groupby'])[c['target']].agg(c['agg']).reset_index().rename(index=str, columns={c['target']:new_feature})
        df = df.merge(gp,on=c['groupby'],how='left')
    return df

agg_cols = [
    
    
    {'groupby': ['prefix'], 'target':'dummy', 'agg':'count'},   
    {'groupby': ['title'], 'target':'dummy', 'agg':'count'},    
    {'groupby': ['tag'], 'target':'dummy', 'agg':'count'},      
    {'groupby': ['prefix','title'], 'target':'dummy', 'agg':'count'},   
    {'groupby': ['prefix','tag'], 'target':'dummy', 'agg':'count'},    
    {'groupby': ['tag','title'], 'target':'dummy', 'agg':'count'},      
    {'groupby': ['prefix','tag','title'], 'target':'dummy', 'agg':'count'},          
]

df_all = agg(df_all,agg_cols)

import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
from sklearn.metrics import f1_score
from datetime import datetime
from tqdm import tqdm
import xlearn as xl
from sklearn.preprocessing import StandardScaler

train_df = df_all[df_all['label'].notnull()]
test_df = df_all[df_all['label'].isnull()]

print("Starting FM. Train shape: {}, test shape: {}".format(train_df.shape,test_df.shape))
#del df
gc.collect()


folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=9999)

oof_fm_preds = np.zeros(train_df.shape[0])
sub_fm_preds = np.zeros(test_df.shape[0])
#cat_features=['prefix', 'title', 'tag','"", ""','"prefix", "tag"','"tag", "title"']
cat_features = [
{'groupby': ['prefix']},   
{'groupby': ['title']},   
{'groupby': ['tag']},       
{'groupby': ['prefix','title']},
{'groupby': ['prefix','tag']},
{'groupby': ['tag','title']},
{'groupby': ['prefix','tag','title']},
]

# drop_list = ['query_prediction', 'label','dummy']
# feats = [f for f in train_df.columns if f not in drop_list]
#print ('feats:' + str(len(feats)))

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['label'])):
    # target encoding
    df_train = train_df.iloc[train_idx]
    train_df_new,test_df_new = train_df,test_df

    drop_list = ['query_prediction', 'label','dummy','prefix_title_same']
    feats = [f for f in train_df_new.columns if f not in drop_list]    
    print ('feats:' + str(len(feats)))
    
    train_x, train_y = train_df_new[feats].iloc[train_idx], train_df_new['label'].iloc[train_idx]
    valid_x, valid_y = train_df_new[feats].iloc[valid_idx], train_df_new['label'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    # FM features
    # Standardize input
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_x)
    X_val = scaler.transform(valid_x)
    X_test = scaler.transform(test_df_new[feats])
    
    fm_model = xl.FMModel(task='binary', init=0.1, 
                      epoch=50, k=4, lr=0.1, 
                      reg_lambda=0.01, opt='sgd', 
                      metric='acc')
    # Start to train
    fm_model.fit(X_train, 
             train_y, 
             eval_set=[X_val, valid_y])


    # Generate predictions
    oof_fm_preds[valid_idx] = fm_model.predict(X_val)
    sub_fm_preds += fm_model.predict(X_test) / folds.n_splits
