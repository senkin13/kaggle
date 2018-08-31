import numpy as np
import pandas as pd
import gc
import time
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('../input/application_train.csv', usecols=['SK_ID_CURR','TARGET'])
test_df = pd.read_csv('../input/application_test.csv', usecols=['SK_ID_CURR'])

#############blending##################
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline



weights = [0.11,0.11, 0.13,0.13, 0.08,0.08, 0.09,0.09, 0.09,0.09] # your weights for each model
files = [
    
'../blending/merge_lgb_all_v4_0.803198.csv', #0.808
'../blending/merge_lgb_all_v6_0.802716.csv', #0.808
'../blending/xuan_lgb_all_v4.10_cv0.800811.csv', #0.809
'../blending/xuan_lgb_all_v4.13_cv0.800871.csv', #0.809
'../blending/xuan_lgb_all_v4.08_cv0.800648.csv', #0.808
'../blending/xuan_lgb_all_v4.12_cv0.80063.csv', #0.808     
'../blending/yuanhao_lgb_all_0827-gbdt2_0.80122.csv', #0.806
'../blending/yuanhao_lgb_all_0829-9999_0.80115.csv', #0.806    
'../blending/neptune_lgb_all_v2.4_CV0.7952.csv', #0.804
'../blending/neptune_lgb_all_v2.6_CV0.7953.csv', #0.804    
] # your prediction files 


# Rank ensemble
finalRank = 0
for i in range(len(files)):
    temp_df = pd.read_csv(files[i])
    finalRank = finalRank + rankdata(temp_df.TARGET, method='ordinal') * weights[i]
finalRank = finalRank / (max(finalRank) + 1.0)

df = temp_df.copy()
df['TARGET'] = finalRank
df.to_csv('../blending/blend_666_9999_0.8036465711862508.csv', index = False)


from sklearn.metrics import roc_auc_score
y_true = train_df.TARGET
y_scores = \
get_rank(merge_8031_666_valid['TARGET'])*0.11 + get_rank(merge_8027_9999_valid['TARGET'])*0.11 + \
get_rank(xuan_8006_666_valid['TARGET'])*0.13 + get_rank(xuan_8006_9999_valid['TARGET'])*0.13 + \
get_rank(xuan_8008_666_valid['TARGET'])*0.08 + get_rank(xuan_8008_9999_valid['TARGET'])*0.08 + \
get_rank(yuanhao_8012_666_valid['TARGET'])*0.09 + get_rank(yuanhao_8011_9999_valid['TARGET'])*0.09 + \
get_rank(neptune_7952_666_valid['TARGET'])*0.09 + get_rank(neptune_7953_9999_valid['TARGET'])*0.09

roc_auc_score(y_true, y_scores)

#OOF Train
# Group1:gap(<0.3)
piupiu_lgb_all_v2_valid = pd.read_csv('../stacking/piupiu_lgb_all_v2.0__valid0.8006862440342553.csv').iloc[48744:].reset_index(drop=True)
plants_lgb_all_v1_valid = pd.read_csv('../stacking/plants_lgb_all_CV_0.7975910742409961.csv')
plants_lgb_all_v2_valid = pd.read_csv('../stacking/plants_lgb_all_CV_0.8019077588455712.csv')
kagglegogogo_lgb_all_v1_valid = pd.read_csv('../stacking/kagglegogogo_lgb_all_v1_0.7985_valid.csv')
yuanhao_lgb_all_v1_valid = pd.read_csv('../stacking/yuanhao_lgb_all_v31_0.79936_valid.csv')

# Group2:gap(>0.3 <0.6)
merge_lgb_all_v5_valid = pd.read_csv('../stacking/merge_lgb_all_v5_0.8027480848567339_valid.csv')
merge_lgb_all_v4_valid = pd.read_csv('../stacking/merge_lgb_all_v4_0.803198_valid.csv')
merge_lgb_all_v3_valid = pd.read_csv('../stacking/merge_lgb_all_v3_0.803763769677185_valid.csv')
merge_lgb_all_v1_valid = pd.read_csv('../stacking/merge_lgb_all_v1_0.8021677464786634_valid.csv')
merge_xgb_all_v1_valid = pd.read_csv('../stacking/xgb_0.801402672688144_valid.csv')
senkin_lgb_all_v1_valid = pd.read_csv('../stacking/senkin_lgb_all_v1_0.7989_valid.csv')
xuan_lgb_all_v2_valid = pd.read_csv('../stacking/xuan_lgb_all_v4.05_cv0.79993_valid.csv')
yuanhao_lgb_all_v2_valid = pd.read_csv('../stacking/yuanhao_lgb_all_0827-gbdt2_0.80122_valid.csv')
yuanhao_lgb_all_v3_valid = pd.read_csv('../stacking/yuanhao_lgb_all_v3_0.79967_valid.csv')
yuanhao_lgb_all_v4_valid = pd.read_csv('../stacking/yuanhao_lgb_all_0827-3_0.80042_valid.csv')

# Group3:gap(>0.6)
neptune_valid = pd.read_csv('../stacking/neptune_lgb_all_v2.4_valid_CV0.7952.csv') 
xuan_lgb_all_v1_valid = pd.read_csv('../stacking/xuan_lgb_all_v3.91_0.7971_valid.csv')
xuan_lgb_all_v3_valid = pd.read_csv('../stacking/xuan_lgb_all_v4.10_cv0.800811_valid.csv')
xuan_lgb_all_v4_valid = pd.read_csv('../stacking/xuan_lgb_all_v4.08_cv0.800648_valid.csv')



# OOF Test

# Group1:gap(<0.3)
piupiu_lgb_all_v2 = pd.read_csv('../stacking/piupiu_lgb_all_v2.0__valid0.8006862440342553.csv').iloc[:48744].reset_index(drop=True)
plants_lgb_all_v1 = pd.read_csv('../stacking/plants_lgb_all_0.7975910742409961.csv')
plants_lgb_all_v2 = pd.read_csv('../stacking/plants_lgb_all_0.8019077588455712.csv')
kagglegogogo_lgb_all_v1 = pd.read_csv('../stacking/kagglegogogo_lgb_all_v1_0.7985.csv')
yuanhao_lgb_all_v1 = pd.read_csv('../stacking/yuanhao_lgb_all_v31_0.79936.csv')

# Group2:gap(>0.3 <0.6)
merge_lgb_all_v5 = pd.read_csv('../stacking/merge_lgb_all_v5_0.8027480848567339.csv')
merge_lgb_all_v4 = pd.read_csv('../stacking/merge_lgb_all_v4_0.803198.csv')
merge_lgb_all_v3 = pd.read_csv('../stacking/merge_lgb_all_v3_0.803763769677185.csv')
merge_lgb_all_v1 = pd.read_csv('../stacking/merge_lgb_all_v1_0.8021677464786634.csv')
merge_xgb_all_v1 = pd.read_csv('../stacking/xgb_0.801402672688144.csv')
senkin_lgb_all_v1 = pd.read_csv('../stacking/senkin_lgb_all_v1_0.7989.csv')
xuan_lgb_all_v2 = pd.read_csv('../stacking/xuan_lgb_all_v4.05_cv0.79993.csv')
yuanhao_lgb_all_v2 = pd.read_csv('../stacking/yuanhao_lgb_all_0827-gbdt2_0.80122.csv')
yuanhao_lgb_all_v3 = pd.read_csv('../stacking/yuanhao_lgb_all_v3_0.79967.csv')
yuanhao_lgb_all_v4 = pd.read_csv('../stacking/yuanhao_lgb_all_0827-3_0.80042.csv')

# Group3:gap(>0.6)
neptune = pd.read_csv('../stacking/neptune_lgb_all_v2.4_CV0.7952.csv') 
xuan_lgb_all_v1 = pd.read_csv('../stacking/xuan_lgb_all_v3.91_0.7971.csv')
xuan_lgb_all_v3 = pd.read_csv('../stacking/xuan_lgb_all_v4.10_cv0.800811.csv')
xuan_lgb_all_v4 = pd.read_csv('../stacking/xuan_lgb_all_v4.08_cv0.800648.csv')


print ('Load All Table Finished!')

#OOF Train
lr_valid = pd.read_csv('../stacking/lr_0.7917_valid.csv')
lgb_valid = pd.read_csv('../stacking/lgb_single_0.7947313080300847_valid.csv')
rf_valid = pd.read_csv('../stacking/rf_single_0.7935997004252096_valid.csv')
xgb_valid = pd.read_csv('../stacking/xgb_single_0.7953397687365324_valid.csv')

# OOF Test
lr = pd.read_csv('../stacking/lr_0.7917.csv')
lgb = pd.read_csv('../stacking/lgb_single_0.7947313080300847.csv')
rf = pd.read_csv('../stacking/rf_single_0.7935997004252096.csv')
xgb = pd.read_csv('../stacking/xgb_single_0.7953397687365324.csv')

def get_rank(x):
    return pd.Series(x).rank(pct=True).values


# ====all table====
## train
train_df['piupiu_lgb_all_v2'] = get_rank(piupiu_lgb_all_v2_valid['piupiu_all_pred'])
train_df['plants_lgb_all_v1'] = get_rank(plants_lgb_all_v1_valid['PRED'])
train_df['plants_lgb_all_v2'] = get_rank(plants_lgb_all_v2_valid['PRED'])
train_df['kagglegogogo_lgb_all_v1'] = get_rank(kagglegogogo_lgb_all_v1_valid['TARGET'])

train_df['merge_lgb_all_v4'] = get_rank(merge_lgb_all_v4_valid['TARGET'])
train_df['merge_lgb_all_v3'] = get_rank(merge_lgb_all_v3_valid['TARGET'])
train_df['merge_lgb_all_v1'] = get_rank(merge_lgb_all_v1_valid['TARGET'])
train_df['merge_xgb_all_v1'] = get_rank(merge_xgb_all_v1_valid['TARGET'])
train_df['senkin_lgb_all_v1'] = get_rank(senkin_lgb_all_v1_valid['TARGET'])
train_df['xuan_lgb_all_v1'] = get_rank(xuan_lgb_all_v1_valid['TARGET'])
train_df['xuan_lgb_all_v2'] = get_rank(xuan_lgb_all_v2_valid['TARGET'])
#train_df['xuan_lgb_all_v3'] = get_rank(xuan_lgb_all_v3_valid['TARGET'])
#train_df['xuan_lgb_all_v4'] = get_rank(xuan_lgb_all_v4_valid['TARGET'])

#train_df['yuanhao_lgb_all_v1'] = get_rank(yuanhao_lgb_all_v1_valid['TARGET'])
train_df['yuanhao_lgb_all_v2'] = get_rank(yuanhao_lgb_all_v2_valid['TARGET'])
#train_df['yuanhao_lgb_all_v3'] = get_rank(yuanhao_lgb_all_v3_valid['TARGET'])
#train_df['yuanhao_lgb_all_v4'] = get_rank(yuanhao_lgb_all_v4_valid['TARGET'])


## test
test_df['piupiu_lgb_all_v2'] = get_rank(piupiu_lgb_all_v2['piupiu_all_pred'])
test_df['plants_lgb_all_v1'] = get_rank(plants_lgb_all_v1['PRED'])
test_df['plants_lgb_all_v2'] = get_rank(plants_lgb_all_v2['PRED'])
test_df['kagglegogogo_lgb_all_v1'] = get_rank(kagglegogogo_lgb_all_v1['TARGET'])

test_df['merge_lgb_all_v4'] = get_rank(merge_lgb_all_v4['TARGET'])
test_df['merge_lgb_all_v3'] = get_rank(merge_lgb_all_v3['TARGET'])
test_df['merge_lgb_all_v1'] = get_rank(merge_lgb_all_v1['TARGET'])
test_df['merge_xgb_all_v1'] = get_rank(merge_xgb_all_v1['TARGET'])
test_df['senkin_lgb_all_v1'] = get_rank(senkin_lgb_all_v1['TARGET'])
test_df['xuan_lgb_all_v1'] = get_rank(xuan_lgb_all_v1['TARGET'])
test_df['xuan_lgb_all_v2'] = get_rank(xuan_lgb_all_v2['TARGET'])
#test_df['xuan_lgb_all_v3'] = get_rank(xuan_lgb_all_v3['TARGET'])
#test_df['xuan_lgb_all_v4'] = get_rank(xuan_lgb_all_v4['TARGET'])

#test_df['yuanhao_lgb_all_v1'] = get_rank(yuanhao_lgb_all_v1['TARGET'])
test_df['yuanhao_lgb_all_v2'] = get_rank(yuanhao_lgb_all_v2['TARGET'])
#test_df['yuanhao_lgb_all_v3'] = get_rank(yuanhao_lgb_all_v3['TARGET'])
#test_df['yuanhao_lgb_all_v4'] = get_rank(yuanhao_lgb_all_v4['TARGET'])




##======all single table========
train_df['lr'] = get_rank(lr_valid['TARGET'])
train_df['lgb'] = get_rank(lgb_valid['TARGET'])
train_df['xgb'] = get_rank(xgb_valid['TARGET'])
train_df['rf'] = get_rank(rf_valid['TARGET'])

test_df['lr'] = get_rank(lr['TARGET'])
test_df['lgb'] = get_rank(lgb['TARGET'])
test_df['xgb'] = get_rank(xgb['TARGET'])
test_df['rf'] = get_rank(rf['TARGET'])

print ('Merge Finished!')


import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

model_list = ['senkin_lgb_all_v1','yuanhao_lgb_all_v2',
              'piupiu_lgb_all_v2','kagglegogogo_lgb_all_v1','plants_lgb_all_v1','plants_lgb_all_v2','merge_lgb_all_v3']
train_df_all = train_df[model_list]
train_df_all.corr()

# plt.figure(figsize = (8, 6))
# sns.heatmap(train_df_all, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
# plt.title('Correlation Heatmap');


%%time

from keras.layers import Dense, Input, Embedding, Dropout, Activation,GRU,Conv1D,GlobalMaxPool1D,MaxPooling1D,CuDNNGRU,TimeDistributed, Lambda, multiply,concatenate,CuDNNLSTM,Bidirectional
from keras.layers import SpatialDropout1D,GlobalMaxPool1D,GlobalAvgPool1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Nadam
from keras import backend as K
from sklearn.metrics import roc_auc_score
import copy

            
def get_model():
    input = Input(shape=(len(feats),))
    merged = Dense(1000)(input)
    relu = PReLU()(merged)
    #merged = Dropout(0.1)(relu)
    preds = Dense(1, activation='sigmoid')(relu)
    model = Model(inputs=[input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    # print(model.summary())
    return model

print("Starting Neural Network. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=666)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
#sub_preds=[]
roc_auc_score_mean = 0
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR'] ]

print ('features:' + str(len(feats)))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    model = get_model()
    best = [-1, 0, 0, 0]  # socre, epoch, model.copy , cv_result
    earlystop = 5
    for epoch in range(10000):
        model.fit(train_x,train_y,batch_size=512, epochs=1, verbose=0)
        r = model.predict(valid_x ,batch_size=512)
        r = np.reshape(r,(valid_x.shape[0]))
        oof_preds[valid_idx] = r
        s = roc_auc_score(valid_y,oof_preds[valid_idx])
        roc_auc_score_mean += s
        print('Fold %2d AUC : %.6f' % (n_fold + 1, s))
        #print(n_fold,epoch,s)
        if s > best[0]:# the bigger is better
            print("epoch " + str(epoch) + " improved from " + str(best[0]) + " to " + str(s))
            best = [s,epoch,copy.copy(model),r]
        if epoch-best[1]>earlystop:
            break

        p = model.predict(test_df[feats], batch_size=512)
        p = np.reshape(p,(test_df.shape[0]))
        sub_preds += p  / folds.n_splits
    
print('Full Mean AUC score:'+ str(roc_auc_score_mean/10)) 
print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

sub_preds['TARGET'] = get_rank(sub_preds['TARGET'] )

nn_oof = pd.DataFrame({"SK_ID_CURR":train_df["SK_ID_CURR"], "TARGET":oof_preds})
nn_preds = pd.DataFrame({"SK_ID_CURR":test_df["SK_ID_CURR"], "TARGET":sub_preds})

nn_oof.to_csv("../models/nn_" + str(roc_auc_score(train_df['TARGET'], oof_preds)) + "_valid.csv", index=False)
nn_preds.to_csv("../models/nn_" + str(roc_auc_score(train_df['TARGET'], oof_preds)) + ".csv", index=False)


%%time

from sklearn.ensemble import RandomForestClassifier

print("Starting RandomForest. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=666)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
roc_auc_score_mean = 0
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR'] ]
    
print ('features:' + str(len(feats)))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)   

    params = {
        'n_estimators': 300,
        'criterion': 'entropy',
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_leaf_nodes': 100,
        #'class_weight': 'balanced_subsample',
        'n_jobs': -1,
        'random_state': 16,
        'verbose': 1
    }    


    model= RandomForestClassifier(**params)
    model.fit(train_x, train_y)
    
    oof_preds[valid_idx] = model.predict_proba(valid_x)[:, 1]
    sub_preds += model.predict_proba(test_df[feats])[:, 1] / folds.n_splits

            
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    
    roc_auc_score_mean += roc_auc_score(valid_y, oof_preds[valid_idx])

print('Full Mean AUC score:'+ str(roc_auc_score_mean/10))    
print('Full Valid AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

rf_oof = pd.DataFrame({"SK_ID_CURR":train_df["SK_ID_CURR"], "TARGET":oof_preds})
rf_preds = pd.DataFrame({"SK_ID_CURR":test_df["SK_ID_CURR"], "TARGET":sub_preds})

rf_oof.to_csv("../models/rf_final_" + str(roc_auc_score(train_df['TARGET'], oof_preds)) + "_valid.csv", index=False)
rf_preds.to_csv("../models/rf_final_" + str(roc_auc_score(train_df['TARGET'], oof_preds)) + ".csv", index=False)


%%time

import lightgbm as lgb

print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=666)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
roc_auc_score_mean = 0
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR'] ]
    
print ('features:' + str(len(feats)))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
        'nthread': 32,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'auc',
        'num_leaves': 3,
        'learning_rate': 0.02,
        'max_depth': 3,  
        'subsample': 1,  
        'colsample_bytree': 1,        
        'verbose': 1,
        #'num_iterations': 250
    }    

    dtrain = lgb.Dataset(
        train_x, label=train_y)
    dval = lgb.Dataset(
        valid_x, label=valid_y, reference=dtrain)    
    bst = lgb.train(
        params, dtrain, num_boost_round=10000,
        #valid_sets=[dval], verbose_eval=100)
        valid_sets=[dval], early_stopping_rounds=200, verbose_eval=100)
        
    oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
    sub_preds += bst.predict(test_df[feats], num_iteration=bst.best_iteration) / folds.n_splits
    
    
    imp = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:30]
    for item in imp:
       print (item) 
            
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    
    roc_auc_score_mean += roc_auc_score(valid_y, oof_preds[valid_idx])
    del bst, train_x, train_y, valid_x, valid_y
    gc.collect()

print('Full Mean AUC score:'+ str(roc_auc_score_mean/10))    
print('Full Valid AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

lgb_oof = pd.DataFrame({"SK_ID_CURR":train_df["SK_ID_CURR"], "TARGET":oof_preds})
lgb_preds = pd.DataFrame({"SK_ID_CURR":test_df["SK_ID_CURR"], "TARGET":sub_preds})

lgb_oof.to_csv("../models/lgb_final_" + str(roc_auc_score(train_df['TARGET'], oof_preds)) + "_valid.csv", index=False)
lgb_preds.to_csv("../models/lgb_final_" + str(roc_auc_score(train_df['TARGET'], oof_preds)) + ".csv", index=False)

%%time

import xgboost as xgb

print("Starting Xgboost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=666)

params = {
    'objective': 'reg:logistic',
    'booster': 'gbtree',
    'eval_metric': 'auc',
    'eta': 0.02,
    'max_depth': 4,   
    'subsample': 0.9,  
    'colsample_bytree': 0.9,
}

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
roc_auc_score_mean = 0
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR'] ]
    
print ('features:' + str(len(feats)))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)


    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(valid_x, label=valid_y)   
    dtest = xgb.DMatrix(test_df[feats])    
    watchlist = [(dtrain, 'train'), (dval, 'valid')]
    bst = xgb.train(
        params, dtrain, 10000,
        watchlist, early_stopping_rounds=200, verbose_eval=100
    )    
        
    oof_preds[valid_idx] = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
    sub_preds += bst.predict(dtest, ntree_limit=bst.best_ntree_limit) / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    
    roc_auc_score_mean += roc_auc_score(valid_y, oof_preds[valid_idx])
    del bst, train_x, train_y, valid_x, valid_y
    gc.collect()

print('Full Mean AUC score:'+ str(roc_auc_score_mean/10))    
print('Full Valid AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

xgb_oof = pd.DataFrame({"SK_ID_CURR":train_df["SK_ID_CURR"], "TARGET":oof_preds})
xgb_preds = pd.DataFrame({"SK_ID_CURR":test_df["SK_ID_CURR"], "TARGET":sub_preds})

xgb_oof.to_csv("../models/xgb_final_" + str(roc_auc_score(train_df['TARGET'], oof_preds)) + "_valid.csv", index=False)
xgb_preds.to_csv("../models/xgb_final_" + str(roc_auc_score(train_df['TARGET'], oof_preds)) + ".csv", index=False)

#encoding=utf8
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from scipy import sparse
import xgboost
import lightgbm

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error,roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB


##############################################################分类####################################################
def stacking(clf,train_x,train_y,test_x,clf_name,class_num=1):
    train=np.zeros((train_x.shape[0],class_num))
    test=np.zeros((test_x.shape[0],class_num))
    test_pre=np.zeros((folds,test_x.shape[0],class_num))
    cv_scores=[]
    #for i,(train_index,test_index) in enumerate(kf):
    for i, (train_index, test_index) in enumerate(kf.split(train_df[feats], train_df['TARGET'])):

        tr_x=train_df[feats].iloc[train_index]
        tr_y=train_df['TARGET'].iloc[train_index]
        te_x=train_df[feats].iloc[test_index]
        te_y = train_df['TARGET'].iloc[test_index]
        if clf_name in ["rf","ada","gb","et","lr","knn","mnb","ovr","gnb"]:
            clf.fit(tr_x,tr_y)
            pre=clf.predict_proba(te_x)[:,1].reshape((te_x.shape[0],1))
            train[test_index]=pre
            test_pre[i,:]=clf.predict_proba(test_x)[:,1].reshape((test_x.shape[0],1))
            cv_scores.append(roc_auc_score(te_y, pre))
        elif clf_name in ["lsvc"]:
            clf.fit(tr_x,tr_y)
            pre=clf.decision_function(te_x)
            train[test_index]=pre
            test_pre[i,:]=clf.decision_function(test_x)
            cv_scores.append(roc_auc_score(te_y, pre))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 6,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'scale_pos_weight': 25,
                      'eta': 0.1,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 16,
                      "num_class": class_num
                      }

            num_round = 5000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,evals=watchlist,
                                  #early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(test_matrix,ntree_limit=model.best_ntree_limit).reshape((te_x.shape[0],1))
                train[test_index]=pre
                test_pre[i, :]= model.predict(z, ntree_limit=model.best_ntree_limit).reshape((test_x.shape[0],1))
                print(roc_auc_score(te_y, pre))
                cv_scores.append(roc_auc_score(te_y, pre))
        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            #z = clf.Dataset(test_x, label=te_y)
            #z=test_x
            params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'num_leaves': 2 ** 5 - 1,
            'min_data_in_leaf':500,
            'max_bin': 127,
            'subsample': .8,
            'subsample_freq': 1,
            'colsample_bytree': 0.7,
            'min_child_weight': 0,
            'scale_pos_weight': 25,
            'reg_alpha':1,
            'reg_lambda':1,
            'seed': 2018,
            'nthread': 16,
            'verbose': 0,
            }
            num_round = 2500
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                  #early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],1))
                train[test_index]=pre
                test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0],1))
                cv_scores.append(roc_auc_score(te_y, pre))
        elif clf_name in ["nn"]:
            from keras.layers import Dense, Dropout, BatchNormalization
            from keras.optimizers import SGD,RMSprop
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from keras.utils import np_utils
            from keras.regularizers import l2
            from keras.models import Sequential
            clf = Sequential()
            clf.add(Dense(1024, input_dim=tr_x.shape[1],activation="relu"))
            #clf.add(SReLU())
            clf.add(Dropout(0.6))
            clf.add(Dense(512,activation="relu"))
            #clf.add(SReLU())
            #clf.add(Dense(64, activation="relu", W_regularizer=l2()))
            clf.add(Dropout(0.3))
            clf.add(Dense(class_num, activation="sigmoid"))
            clf.summary()
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            reduce = ReduceLROnPlateau(min_lr=0.0002,factor=0.05)
            clf.compile(optimizer="rmsprop", loss="binary_crossentropy")
            clf.fit(tr_x, tr_y,
                      batch_size=12800,
                      nb_epoch=2,
                      validation_data=[te_x, te_y],
                      #callbacks=[early_stopping,reduce]
            )
            pre=clf.predict_proba(te_x)
            train[test_index]=pre
            test_pre[i,:]=clf.predict_proba(test_x)
            cv_scores.append(roc_auc_score(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:"%clf_name,cv_scores)
        with open("score.txt","a") as f:
            f.write("%s now score is:"%clf_name+str(cv_scores)+"\n")
    test[:]=test_pre.mean(axis=0)
    print("%s_score_list:"%clf_name,cv_scores)
    print("%s_score_mean:"%clf_name,np.mean(cv_scores))
    with open("score.txt", "a") as f:
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1,class_num),test.reshape(-1,class_num)

def rf(x_train, y_train, x_valid):
    where_are_inf = np.isinf(x_train)
    x_train[where_are_inf] = -1
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_inf] = -1
    randomforest = RandomForestClassifier(n_estimators=1200, max_depth=24, n_jobs=-1, random_state=2017, max_features="auto",verbose=1)
    rf_train, rf_test = stacking(randomforest, x_train, y_train, x_valid,"rf")
    return rf_train, rf_test,"rf"

def ada(x_train, y_train, x_valid):
    where_are_inf = np.isinf(x_train)
    x_train[where_are_inf] = -1
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_inf] = -1
    adaboost = AdaBoostClassifier(n_estimators=60, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking(adaboost, x_train, y_train, x_valid,"ada")
    return ada_train, ada_test,"ada"

def gb(x_train, y_train, x_valid):
    gbdt = GradientBoostingClassifier(learning_rate=0.06, n_estimators=100, subsample=0.8, random_state=2017,max_depth=5,verbose=1)
    gbdt_train, gbdt_test = stacking(gbdt, x_train, y_train, x_valid,"gb")
    return gbdt_train, gbdt_test,"gb"

def et(x_train, y_train, x_valid):
    where_are_inf = np.isinf(x_train)
    x_train[where_are_inf] = -1
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_inf] = -1
    extratree = ExtraTreesClassifier(n_estimators=1200, max_depth=24, max_features="auto", n_jobs=-1, random_state=2017,verbose=1)
    et_train, et_test = stacking(extratree, x_train, y_train, x_valid,"et")
    return et_train, et_test,"et"

def ovr(x_train, y_train, x_valid):
    est=RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=2017, max_features="auto",
                               verbose=1)
    ovr = OneVsRestClassifier(est,n_jobs=-1)
    ovr_train, ovr_test = stacking(ovr, x_train, y_train, x_valid,"ovr")
    return ovr_train, ovr_test,"ovr"

def xgb(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking(xgboost, x_train, y_train, x_valid,"xgb")
    return xgb_train, xgb_test,"xgb"

def lgb(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking(lightgbm, x_train, y_train, x_valid,"lgb")
    return xgb_train, xgb_test,"lgb"

def gnb(x_train, y_train, x_valid):
    x_train=np.log10(x_train+1)
    x_valid=np.log10(x_valid+1)
    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = -1
    x_train[where_are_inf] = -1
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = -1
    x_valid[where_are_inf] = -1

    gnb=GaussianNB()
    gnb_train, gnb_test = stacking(gnb, x_train, y_train, x_valid,"gnb")
    return gnb_train, gnb_test,"gnb"

def lr(x_train, y_train, x_valid):
    x_train=np.log10(x_train+1)
    x_valid=np.log10(x_valid+1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = -1
    x_train[where_are_inf] = -1
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = -1
    x_valid[where_are_inf] = -1

    scale=StandardScaler()
    #scale=MinMaxScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)

    logisticregression=LogisticRegression(n_jobs=-1,random_state=2017,C=0.1,max_iter=500)
    lr_train, lr_test = stacking(logisticregression, x_train, y_train, x_valid, "lr")
    return lr_train, lr_test, "lr"

def fm(x_train, y_train, x_valid):
    pass


def lsvc(x_train, y_train, x_valid):
    x_train=np.log10(x_train+1)
    x_valid=np.log10(x_valid+1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0

    scale=StandardScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)

    #linearsvc=SVC(probability=True,kernel="linear",random_state=2017,verbose=1)
    #linearsvc=SVC(probability=True,kernel="linear",random_state=2017,verbose=1)
    linearsvc=LinearSVC(random_state=2017)
    lsvc_train, lsvc_test = stacking(linearsvc, x_train, y_train, x_valid, "lsvc")
    return lsvc_train, lsvc_test, "lsvc"

def knn(x_train, y_train, x_valid):
    x_train=np.log10(x_train+1)
    x_valid=np.log10(x_valid+1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = -1
    x_train[where_are_inf] = -1
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = -1
    x_valid[where_are_inf] = -1

    scale=StandardScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)

    pca = PCA(n_components=10)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_valid = pca.transform(x_valid)

    kneighbors=KNeighborsClassifier(n_neighbors=6,n_jobs=-1)
    knn_train, knn_test = stacking(kneighbors, x_train, y_train, x_valid, "knn")
    return knn_train, knn_test, "knn"

def nn(x_train, y_train, x_valid):
    #x_train=np.log10(x_train+1)
    #x_valid=np.log10(x_valid+1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = -1
    x_train[where_are_inf] = -1
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = -1
    x_valid[where_are_inf] = -1

    scale=StandardScaler()
    #scale=MinMaxScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)

    nn_train, nn_test = stacking("", x_train, y_train, x_valid, "nn")
    return nn_train, nn_test, "nn"
###########################################################################################################

#####################################################回归##################################################

#####################################################获取数据##############################################

###########################################################################################################
def get_data():   #数据读取
    #x_train = np.array(pd.read_csv('xgb_fm/train_xgb_feature.csv'))#[:1000]
    #x_test = np.array(pd.read_csv('xgb_fm/test_xgb_feature.csv'))#[:100]
    #y_train = pd.read_csv('xgb_fm/label.csv')["is_attributed"].values
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR'] ]
    x_train = train_df[feats]#[:1000]
    x_test = test_df[feats]#[:100]
    y_train = train_df['TARGET']
    
    return x_train,y_train,x_test

if __name__=="__main__":
    np.random.seed(2018)
    x_train, y_train, x_valid= get_data()

    folds = 10
    seed = 666
    #kf = StratifiedKFold(x_train.shape[0], n_splits=folds, shuffle=True, random_state=seed)    #五折改成和nn一样的
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)    #五折改成和nn一样的
    
    #############################################选择模型###############################################
    #
    #
    #
    clf_list = [lr]   #分类器，防止内存不够最好一个个训练
    #
    #
    #
    column_list = []
    train_data_list=[]
    test_data_list=[]
    for clf in clf_list:
        train_data,test_data,clf_name=clf(x_train,y_train,x_valid)
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        column_list.append("%s_%s" % (clf_name, "_model"))

    train = np.concatenate(train_data_list, axis=1)
    test = np.concatenate(test_data_list, axis=1)

    result=test.copy()

    train = pd.DataFrame(train)
    train.columns = column_list

    test = pd.DataFrame(test)
    test.columns = column_list

    #数据输出
    train.to_csv("../models/lr_valid.csv", index=None)
    test.to_csv("../models/lr_.csv", index=None)


from sklearn.metrics import roc_auc_score
y_true = train_df.TARGET
y_scores = get_rank(rf_oof['TARGET'])*0.5 + get_rank(lgb_oof['TARGET'])*0.5
roc_auc_score(y_true, y_scores)

blend = pd.DataFrame()
blend['SK_ID_CURR'] = lgb_preds['SK_ID_CURR']
blend['lgb_stacking'] = get_rank(lgb_preds['TARGET'])
blend['rf_stacking'] = get_rank(rf_preds['TARGET'])
blend['TARGET'] = get_rank(blend['lgb_stacking'] * 0.5 + blend['rf_stacking'] * 0.5)
blend[['SK_ID_CURR','TARGET']].to_csv("../models/stacking_final_" + str(roc_auc_score(y_true, y_scores)) + ".csv", index=False)

