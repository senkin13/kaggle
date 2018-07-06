##LGBM Model

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import sparse
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
import os
import glob

df_all = pickle.load(open('/tmp/basic_numerical_active.pkl','rb'))
df_train = df_all[df_all['deal_probability'].notnull()]
df_test = df_all[df_all['deal_probability'].isnull()].reset_index(drop=True)
y = df_all[df_all['deal_probability'].notnull()].deal_probability

# tfidf
ready_df_train = sparse.load_npz('/tmp/features/nlp/ready_df_train_200000_new.npz')
ready_df_test = sparse.load_npz('/tmp/features/nlp/ready_df_test_200000_new.npz')
tfvocab = pickle.load(open('/tmp/features/nlp/tfvocab_200000_new.pkl', 'rb'))

# image - put features to /tmp/features/image/train/ /data/features/image/test/
for fn in glob.glob('/tmp/features/image/train/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/image/test/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    
df_train['dullnessminuswhiteness'] = df_train['dullness'] - df_train['whiteness']
df_test['dullnessminuswhiteness'] = df_test['dullness'] - df_test['whiteness']

# tsvd
for fn in glob.glob('/tmp/features/tsvd/tmp/train/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/tsvd/tmp/test/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    
# text agg
for fn in glob.glob('/tmp/features/text_agg/train/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/text_agg/test/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))       
    
#number agg    
for fn in glob.glob('/tmp/features/number_agg/clean_train_active/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/number_agg/clean_test_active/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))     
    
# image_top_1 image_top_2
for fn in glob.glob('/tmp/features/number_agg/clean_train_image_top_1/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/number_agg/clean_test_image_top_1/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))     
    
# diff features
df_train['image_top_1_diff_price'] = df_train['price'] - df_train['image_top_1_median_price']
df_train['parent_category_name_diff_price'] = df_train['price'] - df_train['parent_category_name_mean_price']
df_train['category_name_diff_price'] = df_train['price'] - df_train['category_name_mean_price']
df_train['param_1_diff_price'] = df_train['price'] - df_train['param_1_mean_price']
df_train['param_2_diff_price'] = df_train['price'] - df_train['param_2_mean_price']
df_train['item_seq_number_diff_price'] = df_train['price'] - df_train['item_seq_number_mean_price']
df_train['user_id_diff_price'] = df_train['price'] - df_train['user_id_mean_price']
df_train['region_diff_price'] = df_train['price'] - df_train['region_mean_price']
df_train['city_diff_price'] = df_train['price'] - df_train['city_mean_price']

df_test['image_top_1_diff_price'] = df_test['price'] - df_test['image_top_1_median_price']
df_test['parent_category_name_diff_price'] = df_test['price'] - df_test['parent_category_name_mean_price']
df_test['category_name_diff_price'] = df_test['price'] - df_test['category_name_mean_price']
df_test['param_1_diff_price'] = df_test['price'] - df_test['param_1_mean_price']
df_test['param_2_diff_price'] = df_test['price'] - df_test['param_2_mean_price']
df_test['item_seq_number_diff_price'] = df_test['price'] - df_test['item_seq_number_mean_price']
df_test['user_id_diff_price'] = df_test['price'] - df_test['user_id_mean_price']
df_test['region_diff_price'] = df_test['price'] - df_test['region_mean_price']
df_test['city_diff_price'] = df_test['price'] - df_test['city_mean_price']

# drop_list
drop_list = [
    'param_123',
    'wday_region_mean_price',
    'wday_region_median_price',
    'wday_region_sum_price',
    'wday_region_max_price',   
    'wday_city_mean_price',
    'wday_city_median_price',
    'wday_city_sum_price',
    'wday_city_max_price', 
    'param_123_num_space',
    'param_123_num_pun',
    'title_num_pun',
    'title_num_space',

 ]

for d in drop_list:
    df_train.drop([d],axis=1,inplace=True)
    df_test.drop([d],axis=1,inplace=True)
    
# final feature    
from scipy.sparse import hstack, csr_matrix

df_train = df_train.drop([
                'deal_probability'],axis=1)   

df_test = df_test.drop([
                'deal_probability'],axis=1) 

X_tr = hstack([csr_matrix(df_train),ready_df_train]) # Sparse Matrix
X_test = hstack([csr_matrix(df_test),ready_df_test])

tfvocab = df_train.columns.tolist() + tfvocab

for shape in [X_tr,X_test]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))    

# train model and predict

from sklearn.model_selection import KFold

X = X_tr.tocsr()
#del X_tra
gc.collect()

test_pred = np.zeros(X_test.shape[0])
cat_features=['region','city','parent_category_name',
              'category_name',
              'user_type','image_top_1','param_1','param_2','param_3','wday']

params = {
    'boosting_type': 'gbdt',
    'objective': 'xentropy',
    'metric': 'rmse',
    'learning_rate': 0.015,
    'num_leaves': 600,  
    #'max_depth': 15,  
    'max_bin': 256,  
    'subsample': 1,  
    'colsample_bytree': 0.1,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
    'verbose': 1
    }

MAX_ROUNDS = 15000
NFOLDS = 5
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=228)
xval_err = 0

for i,(train_index,val_index) in enumerate(kfold.split(X,y)):
    print("Running fold {} / {}".format(i + 1, NFOLDS))
    print("Train Index:",train_index,",Val Index:",val_index)
    X_tra,X_val,y_tra,y_val = X[train_index, :], X[val_index, :], y[train_index], y[val_index]
    if i >=0:

        dtrain = lgb.Dataset(
            X_tra, label=y_tra, feature_name=tfvocab, categorical_feature=cat_features)
        dval = lgb.Dataset(
            X_val, label=y_val, reference=dtrain, feature_name=tfvocab, categorical_feature=cat_features)    
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dval], early_stopping_rounds=200, verbose_eval=200)
        val_pred = bst.predict(
            X_val, num_iteration=bst.best_iteration or MAX_ROUNDS)
        e = val_pred-y_val
        xval_err += np.dot(e,e)
        del dtrain,dval
        del X_tra,y_tra,y_val,X_val
        gc.collect()
        
        new_list = sorted(
            zip(tfvocab, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True)[:200]
        for item in new_list:
            print (item)  
            
        test_pred_current = bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
        test_pred += bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
        test_pred_current.dump('../models/kfold5_' + str(i) + '.pkl')
        del test_pred_current
        gc.collect()

print("Full Validation RMSE:", np.sqrt(xval_err/X.shape[0]))

test_pred /= NFOLDS

test = pd.read_csv('../input/test.csv', index_col = 'item_id', parse_dates = ['activation_date'])
testdex = test.index
sub = pd.DataFrame(test_pred,columns=["deal_probability"],index=testdex)
sub['deal_probability'] = sub['deal_probability'].clip(0.0, 1.0) # Between 0 and 1
sub.to_csv("../models/sub.csv",index=True,header=True)    
