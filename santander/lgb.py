%%time
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
from scipy import stats
from scipy.signal import hann
from tqdm import tqdm_notebook
from scipy.signal import hilbert
from scipy.signal import convolve
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings('ignore')

# train = pd.read_csv('../input/train.csv')
# train['is_real'] = 1
# test = pd.read_csv('../input/test.csv')
  
# test_fake = pd.read_csv('../input/synthetic_samples_indexes.csv',header=None)
# test_fake.columns = ['ID_code']
# test_fake['ID_code'] = test_fake.ID_code.apply(lambda x: 'test_' + str(int(x)))
# test_fake['is_real'] = 0
# test = pd.merge(test, test_fake, on = ['ID_code'], how = 'left')
# test.is_real.fillna(1, inplace = True)

# df = pd.concat([train,test],axis=0)
# df.to_pickle('../input/df.pkl')

df= pd.read_pickle('../input/df.pkl')
train = df[df['target'].notnull()]
train['flag'] = 'train'
test = df[(df['target'].isnull()) & (df['is_real']==1)]
test['flag'] = 'test'
sub = pd.read_csv('../input/sample_submission.csv',usecols=['ID_code'])

drop_features = ['ID_code', 'target', 'is_real']
feats = [f for f in df.columns if f not in  drop_features]

for var in feats:
    print(var)
    data = pd.concat([train[['ID_code', 'flag', var]], test[['ID_code', 'flag', var]]])
    data['zscore_' + var]= (data[var]-data[var].mean())/data[var].std()*5    
    data['weight_' + var] = data[var].map(data.groupby([var])[var].count())   
    
    train['zscore_' + var] = data[data['flag']=='train']['zscore_' + var]
    test['zscore_' + var] = data[data['flag']=='test']['zscore_' + var]
    
    train['weight_' + var] = train[var].map(data.groupby([var])[var].count())
    test['weight_'+ var] = test[var].map(data.groupby([var])[var].count())

    train['minmax_weight_'+var]=(train['weight_'+var]-train['weight_'+var].min())/(train['weight_'+var].max()-train['weight_'+var].min())*8+1
    test['minmax_weight_'+var]=(test['weight_'+var]-test['weight_'+var].min())/(test['weight_'+var].max()-test['weight_'+var].min())*8+1
    

    train['binary_' + var] = train['weight_' + var].apply(lambda x: 1 if x > 1 else 0) * train[var]
    test['binary_' + var] = test['weight_' + var].apply(lambda x: 1 if x > 1 else 0) * test[var]
    train['double_' + var] = train['weight_' + var].apply(lambda x: 2 if x > 1 else 1) * train[var]
    test['double_' + var] = test['weight_' + var].apply(lambda x: 2 if x > 1 else 1) * test[var]
    train['value_double_' + var] = train[var] / train['weight_' + var].apply(lambda x: 2 if x > 1 else 1) 
    test['value_double_' + var] = test[var] / test['weight_' + var].apply(lambda x: 2 if x > 1 else 1) 
    train['value_count_' + var] = train[var].map(data.groupby([var])[var].count()) * train[var]
    test['value_count_'+ var] = test[var].map(data.groupby([var])[var].count()) * test[var]
    train['value_count2_' + var] = train[var].map(data.groupby([var])[var].count() ** 2) * train[var]
    test['value_count2_'+ var] = test[var].map(data.groupby([var])[var].count() ** 2) * test[var]
    train['log_' + var] = train[var].map(data.groupby([var])[var].count()) * np.log(train[var] - train[var].min() + 1)
    test['log_'+ var] = test[var].map(data.groupby([var])[var].count()) * np.log(test[var] - test[var].min() + 1)
    train['binary2_' + var] = train['weight_' + var].apply(lambda x: 1 if x > 2 else 0) * train[var]
    test['binary2_' + var] = test['weight_' + var].apply(lambda x: 1 if x > 2 else 0) * test[var]
    train['binary3_' + var] = train['weight_' + var].apply(lambda x: 1 if x > 3 else 0) * train[var]
    test['binary3_' + var] = test['weight_' + var].apply(lambda x: 1 if x > 3 else 0) * test[var]    
    train['binary4_' + var] = train['weight_' + var].apply(lambda x: 1 if x > 4 else 0) * train[var]
    test['binary4_' + var] = test['weight_' + var].apply(lambda x: 1 if x > 4 else 0) * test[var]

    train['transform_binary_' + var] = train['minmax_weight_' + var].apply(lambda x: 1 if x > 1 else 0) * train['zscore_' + var]
    test['transform_binary_' + var] = test['minmax_weight_' + var].apply(lambda x: 1 if x > 1 else 0) * test['zscore_' + var]
    train['transform_double_' + var] = train['minmax_weight_' + var].apply(lambda x: 2 if x > 1 else 1) * train['zscore_' + var]
    test['transform_double_' + var] = test['minmax_weight_' + var].apply(lambda x: 2 if x > 1 else 1) * test['zscore_' + var]
    train['transform_value_double_' + var] = train['zscore_' + var] / train['minmax_weight_' + var].apply(lambda x: 2 if x > 1 else 1) 
    test['transform_value_double_' + var] = test['zscore_' + var] / test['minmax_weight_' + var].apply(lambda x: 2 if x > 1 else 1) 
    train['transform_value_count_' + var] =  train['minmax_weight_' + var] * train['zscore_' + var]
    test['transform_value_count_'+ var] =  test['minmax_weight_' + var] * test['zscore_' + var]
    train['transform_value_count2_' + var] = (train['minmax_weight_' + var] ** 2) * train['zscore_' + var]
    test['transform_value_count2_'+ var] = (test['minmax_weight_' + var] ** 2) * test['zscore_' + var]
    train['transform_log_' + var] = train['minmax_weight_' + var] * np.log(train['zscore_' + var] - train['zscore_' + var].min() + 1)
    test['transform_log_'+ var] = test['minmax_weight_' + var] * np.log(test['zscore_' + var] - test['zscore_' + var].min() + 1)

    
print ('======Save to Train=======')  
for i in train.columns.values:
    if i not in feats:
        print (i)
        train[i].to_pickle('../feature/train/' + str(i))       

print ('======Save to Test=======')        
for i in test.columns.values:
    if i not in feats:
        print (i)
        test[i].to_pickle('../feature/test/' + str(i))  
        
import os
import glob
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings 
warnings.filterwarnings('ignore')

df= pd.read_pickle('../input/df.pkl')
train_df = df[df['target'].notnull()]
test_df = df[(df['target'].isnull()) & (df['is_real']==1)]

print ('Load Train')
for cat in ['weight_var','value_count_var','value_count2_var','log_var',
            'binary_var','double_var','binary2_var','binary3_var','binary4_var','value_double_var',
           'transform']:
    
    for fn in glob.glob('../feature/train/' + str(cat) + '_*'):
        tmp = pd.read_pickle(fn)
        train_df[os.path.basename(fn)] = tmp
        del tmp
        gc.collect()
        print (os.path.basename(fn))
  
print ('Load Test')
for cat in ['weight_var','value_count_var','value_count2_var','log_var',
            'binary_var','double_var','binary2_var','binary3_var','binary4_var','value_double_var',
           'transform']:
    for fn in glob.glob('../feature/test/' + str(cat) + '_*'):
        tmp = pd.read_pickle(fn)
        test_df[os.path.basename(fn)] = tmp
        del tmp
        gc.collect()
        print (os.path.basename(fn))    
        
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
from sklearn.metrics import mean_squared_error
from scipy import sparse
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.feature_selection import SelectFromModel, VarianceThreshold,RFE
from sklearn.metrics import roc_auc_score, roc_curve,log_loss
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import mixture


drop_features = ['ID_code', 'target','is_real',
                
]

value_minstd_var = [col for col in train_df if col.startswith('value_minstd_var')]
value_maxstd_var = [col for col in train_df if col.startswith('value_maxstd_var')]
value_meanstd_var = [col for col in train_df if col.startswith('value_meanstd_var')]
value_quantile25std_var = [col for col in train_df if col.startswith('value_quantile25std_var')]
std_count_var = [col for col in train_df if col.startswith('std_count_var')]
value_q_var = [col for col in train_df if col.startswith('value_q_var')]
drop7_features = [col for col in train_df if col.startswith('log_var')]


feats = [f for f in train_df.columns if f not in  drop_features 
        if f not in value_minstd_var
        if f not in value_maxstd_var
        if f not in value_meanstd_var
        if f not in value_quantile25std_var
        if f not in std_count_var
        if f not in value_q_var
        if f not in drop7_features]
#cat_features = [col for col in df if col.startswith('weight_')]
cat_features = []

n_splits= 10
SEED = 817
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
#folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
cv_list = []

feats = null_importance_feature #+ binary_var + double_var+ value_double_var
print ('feats:' + str(len(feats)))

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
    
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

    print('Train Index:',train_idx,'Val Index:',valid_idx)

    params = {'metric': 'auc',
        'learning_rate': 0.01,
        'nthread': -1,
        'max_depth':1,#1
        'max_bin': 500,#350,   
        'reg_lambda': 1.0,#0.0,        
        'objective': 'binary', 
        'bagging_freq': 5,
        'feature_fraction':0.05,
        'bagging_fraction':0.4,#0.4
        'min_data_in_leaf':50,
        'min_sum_hessian_in_leaf':10,
        'boost_from_average':False,
        'tree_learner':'serial',
        'num_leaves': 2,
        'boosting_type': 'gbdt'}

    if n_fold >= 0:
        evals_result = {}
        dtrain = lgb.Dataset(
           train_x, label=train_y,categorical_feature=cat_features)#
        dval = lgb.Dataset(
           valid_x, label=valid_y, reference=dtrain,)
        bst = lgb.train(
           params, dtrain, num_boost_round=100000,
           valid_sets=[dval], early_stopping_rounds=1000, verbose_eval=1000,)#feval = evalerror

        new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:20]
        for item in new_list:
            print (item)

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        oof_cv = roc_auc_score(valid_y,  oof_preds[valid_idx])
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += bst.predict(test_df[feats], num_iteration=bst.best_iteration) / folds.n_splits # test_df_new

cv = roc_auc_score(train_df['target'],  oof_preds)
print('Full OOF AUC %.6f' % cv)


# oof_df = pd.DataFrame()
# oof_df['card_id'] = train_df['card_id']
# oof_df['target'] = oof_preds
# oof_df[['card_id','target']].to_csv('../ensemble/lgb_v1_oof_' + str(cv) + '_' + str(SEED) + '.csv',index=False)

# test_df['target'] = sub_preds
# test_df[['card_id','target']].to_csv('../ensemble/lgb_v1_pred_' + str(cv) + '_' + str(SEED)  + '.csv',index=False)        
        
