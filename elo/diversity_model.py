## %%time
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cb
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import optimizers
from keras.optimizers import Adam
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.layers.advanced_activations import PReLU
from keras import layers
from keras.layers import LeakyReLU
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers import Dropout, Embedding, Permute, Concatenate, Flatten, Reshape
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D, PReLU
from keras.layers.advanced_activations import LeakyReLU, PReLU,ELU
from keras.models import Model
from keras.layers import CuDNNGRU, CuDNNLSTM, Bidirectional, Lambda
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,LearningRateScheduler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from keras.regularizers import l1, l2, l1_l2
from keras import optimizers
import tensorflow as tf
import random
import warnings 
warnings.filterwarnings('ignore')


############################################
# V2
# df = pd.read_pickle('../feature/df_v2_247.pkl')

# V5
# df = pd.read_pickle('../feature/df_v5_354.pkl')

# V6
# df = pd.read_pickle('../feature/df_v6_229.pkl')

# V8
# df = pd.read_pickle('../feature/df_v8_317.pkl')

# V9
# df = pd.read_pickle('../feature/df_v9_509.pkl')

# V10
#df = pd.read_pickle('../feature/df_v10_385.pkl')

############################################
# V2 nosubmodel
# df = pd.read_pickle('../feature/df_v2_nosubmodel.pkl')

# V5 nosubmodel
# df = pd.read_pickle('../feature/df_v5_nosubmodel.pkl')

# V6 nosubmodel
# df = pd.read_pickle('../feature/df_v6_nosubmodel.pkl')

# V8 nosubmodel
# df = pd.read_pickle('../feature/df_v8_nosubmodel.pkl')

# V9 nosubmodel
# df = pd.read_pickle('../feature/df_v9_nosubmodel.pkl')

# V10 nosubmodel
# df = pd.read_pickle('../feature/df_v10_nosubmodel.pkl')
############################################
def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

def preprocess(train_df,test_df,feats):
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(0) 

    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.fillna(0)
    
    scaler = StandardScaler()
    train_df[feats] = scaler.fit_transform(train_df[feats])
    test_df[feats] = scaler.transform(test_df[feats])
    
    return train_df[feats], test_df[feats]

def ann(input_shape):
    model = Sequential()
    model.add(Dense(2 ** 11, input_dim = input_shape, init='he_normal', activation='relu'))
    model.add(Dropout(0.2))    
    model.add(BatchNormalization())
    model.add(Dense(2 ** 9, init='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2)) 
    model.add(Dense(2 ** 7, init='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))      
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam') #Adam(lr=0.001, decay=0.0001)
    return model

def ann2(input_shape):
    model = Sequential()
    model.add(Dense(2 ** 9, input_dim = input_shape, init='random_uniform', activation='relu'))
    model.add(Dropout(0.25))    
    model.add(BatchNormalization())
    model.add(Dense(2 ** 7, init='random_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25)) 
    model.add(Dense(2 ** 5, init='random_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))      
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam') #Adam(lr=0.001, decay=0.0001)
    return model

def ann3(input_shape):
    model = Sequential()
    model.add(Dense(2 ** 10, input_dim = input_shape, init='random_uniform', activation='relu'))
    model.add(Dropout(0.25))    
    model.add(BatchNormalization())
    model.add(Dense(2 ** 9, init='random_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25)) 
    model.add(Dense(2 ** 6, init='random_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))     
    model.add(Dense(2 ** 5, init='random_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))      
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam') #Adam(lr=0.001, decay=0.0001)
    return model

def ffnn(input_shape):
    nn = Sequential()
    nn.add(Dense(units = 400 , kernel_initializer = 'he_normal', input_dim = input_shape))
    nn.add(PReLU())
    nn.add(Dropout(.3))
    nn.add(Dense(units = 200 , kernel_initializer = 'he_normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(units = 64 , kernel_initializer = 'he_normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(units = 32, kernel_initializer = 'he_normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(units = 16, kernel_initializer = 'he_normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(1))
    nn.compile(loss='mean_squared_error', optimizer='adam')
    return nn

def ffnn2(input_shape):
    nn = Sequential()
    nn.add(Dense(units = 1000 , kernel_initializer = 'random_uniform', input_dim = input_shape))
    nn.add(PReLU())
    nn.add(Dropout(.25))
    nn.add(Dense(units = 500 , kernel_initializer = 'random_uniform'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.25))
    nn.add(Dense(units = 300 , kernel_initializer = 'random_uniform'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.25))
    nn.add(Dense(units = 100, kernel_initializer = 'random_uniform'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.25))
    nn.add(Dense(units = 32, kernel_initializer = 'random_uniform'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.25))
    nn.add(Dense(1, kernel_initializer='random_uniform'))
    nn.compile(loss='mean_squared_error', optimizer='adam')
    return nn

def cv_cat(train_df,test_df,feats,seed,pkl):
    print ('feats:' + str(train_df[feats].shape[1] ))
    cat_features = [c for c in feats if 'feature_' in c]
    n_splits= 5
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    cv_list = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):

        trn_data = cb.Pool(train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx])
        val_data = cb.Pool(train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx])

        valid_y = train_df['target'].iloc[valid_idx]  
        
        print("Train Index:",train_idx,",Val Index:",valid_idx)
 
        num_round = 10000
        cb_model = cb.CatBoostRegressor(learning_rate=0.01, iterations=num_round, verbose=True, 
                              use_best_model=True, l2_leaf_reg=20, allow_writing_files=False, metric_period=50,
                              random_seed=seed, depth=10, loss_function='RMSE', od_wait=100, od_type='Iter')
        cb_model.fit(trn_data, verbose_eval = 100, eval_set = val_data)

        oof_preds[valid_idx] = cb_model.predict(train_df[feats].iloc[valid_idx])
        oof_cv = rmse(valid_y,  oof_preds[valid_idx])
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += cb_model.predict(test_df[feats]) / folds.n_splits 

    cv = rmse(train_df['target'],  oof_preds)
    print('Full OOF RMSE %.6f' % cv)  

    oof_df = pd.DataFrame()
    oof_df['card_id'] = train_df['card_id']
    oof_df['target'] = oof_preds
    oof_df[['card_id','target']].to_csv('../' + str(seed) + '/cat_' + str(pkl) + '_oof_' + str(cv) + '_' + str(seed) + '.csv',index=False)

    test_df['target'] = sub_preds
    test_df[['card_id','target']].to_csv('../' + str(seed) + '/cat_' + str(pkl) +'_pred_' + str(cv) + '_' + str(seed) + '.csv',index=False)
    
def cv_lgb(train_df,test_df,feats,seed,pkl):
    print ('feats:' + str(train_df[feats].shape[1] ))
    cat_features = [c for c in feats if 'feature_' in c]
    n_splits= 5
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    cv_list = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):

        train_x, train_y = train_df[feats].iloc[train_idx], df[df['target'].notnull()]['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], df[df['target'].notnull()]['target'].iloc[valid_idx]     
        print("Train Index:",train_idx,",Val Index:",valid_idx)
        params = {
               "objective" : "regression", 
               "boosting" : "gbdt", #
               "metric" : "rmse",  
               "max_depth": 9, #9
               "min_data_in_leaf": 70, #70
               "min_gain_to_split": 0.05,#0.05 
               "reg_alpha": 0.1, #0.1,
               "reg_lambda": 20, #20
               "num_leaves" : 120, #120
               "max_bin" : 350, #350
               "learning_rate" : 0.005, #0.005
               "bagging_fraction" : 1,
               "bagging_freq" : 1,
               "bagging_seed" : seed,
               "feature_fraction" : 0.2, #0.2
               "verbosity": -1,
               "random_state": seed,
        }
        
        print("Fold:" + str(n_fold))
        dtrain = lgb.Dataset(
            train_x, label=train_y,categorical_feature=cat_features,)#categorical_feature=cat_features
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain,categorical_feature=cat_features,) #weight=train_df.iloc[valid_idx]['outliers'] *  (-0.1) + 1
        bst = lgb.train(
            params, dtrain, num_boost_round=1780,
            valid_sets=[dval],verbose_eval=100,)# early_stopping_rounds=200, 
        
#         new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
#         for item in new_list:
#             print (item) 

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=1780)#bst.best_iteration
        oof_cv = rmse(valid_y,  oof_preds[valid_idx])
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += bst.predict(test_df[feats], num_iteration=1780) / folds.n_splits # test_df_new


    cv = rmse(train_df['target'],  oof_preds)
    print('Full OOF RMSE %.6f' % cv)  

    oof_df = pd.DataFrame()
    oof_df['card_id'] = train_df['card_id']
    oof_df['target'] = oof_preds
    oof_df[['card_id','target']].to_csv('../' + str(seed) + '/lgb_' + str(pkl) + '_oof_' + str(cv) + '_' + str(seed) + '.csv',index=False)

    test_df['target'] = sub_preds
    test_df[['card_id','target']].to_csv('../' + str(seed) + '/lgb_' + str(pkl) +'_pred_' + str(cv) + '_' + str(seed) + '.csv',index=False)

def cv_nn(train_df,test_df,feats,seed,network,pkl):
    print ('feats:' + str(train_df[feats].shape[1] ))
    n_splits= 5
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros((train_df.shape[0],1))
    sub_preds = np.zeros((test_df.shape[0],1))
    cv_list = []
    std_list = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):

        train_x, train_y = train_df[feats].iloc[train_idx], df[df['target'].notnull()]['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], df[df['target'].notnull()]['target'].iloc[valid_idx]     
        print("Train Index:",train_idx,",Val Index:",valid_idx)
    
    
        model = network(train_x.shape[1])
    
        filepath = str(n_fold) + "_nn_best_model.hdf5" 
        es = EarlyStopping(patience=5, mode='min', verbose=1) #monitor=root_mean_squared_error, 
        checkpoint = ModelCheckpoint(filepath=filepath, save_best_only=True,mode='auto') #monitor=root_mean_squared_error
        reduce_lr_loss = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=2, verbose=1, epsilon=1e-4, mode='min')

        hist = model.fit([ train_x], train_y, batch_size=128, epochs=30, validation_data=(valid_x, valid_y), callbacks=[es, checkpoint, reduce_lr_loss], verbose=1)

        model.load_weights(filepath)
        _oof_preds = model.predict(valid_x, batch_size=1024,verbose=1)
        oof_preds[valid_idx] = _oof_preds.reshape((-1,1))

        oof_cv = rmse(valid_y,  oof_preds[valid_idx])
        oof_std = np.std(oof_preds[valid_idx])
        cv_list.append(oof_cv)
        std_list.append(oof_std)
        print (cv_list)
        print (std_list)
        sub_preds += model.predict(test_df[feats] , batch_size=1024).reshape((-1,1)) / folds.n_splits # test_df_new

    cv = rmse(train_df['target'],  oof_preds)
    print('Full OOF RMSE %.6f' % cv)  
    std = np.std(oof_preds)
    print('Full OOF STD %.6f' % std)  

    oof_df = pd.DataFrame()
    oof_df['card_id'] = df[df['target'].notnull()]['card_id']
    oof_df['target'] = oof_preds
    oof_df[['card_id','target']].to_csv('../' + str(seed) + '/nn_' + str(pkl) + '_oof_' + str(cv) + '_' + str(seed) + '.csv',index=False)

    test_df['target'] = sub_preds
    test_df[['card_id','target']].to_csv('../' + str(seed) + '/nn_' + str(pkl) +'_pred_' + str(cv) + '_' + str(seed) + '.csv',index=False)
    
pkl = [ 'df_v10_385','df_v10_nosubmodel',
'df_v9_nosubmodel',       'df_v9_509',
      'df_v8_317', 'df_v8_nosubmodel',
      'df_v6_229',  'df_v6_nosubmodel',
      'df_v5_354', 'df_v5_nosubmodel',
      'df_v2_247', 'df_v2_nosubmodel',
      ]

for p in pkl:                                 
    df = pd.read_pickle('../feats/' + str(p) + '.pkl')
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]
    # outlier
    train_df['outliers'] = 0
    train_df.loc[train_df['target'] < -30, 'outliers'] = 1
    feats = [f for f in train_df.columns if f not in ['card_id','target', 'outliers']]
# CATBOOST    
    for seed in [4590,223,2020,28888,817,2233,111111,]: 
        print(p,seed)   
        cv_cat(train_df,test_df,feats,seed,p)

# LGB    
        for seed in [4590]: #111111,2020,223,28888,817,2233,
        print(p,seed)   
        cv_lgb(train_df,test_df,feats,seed,p)

# NN
    train_df[feats], test_df[feats] = preprocess(train_df,test_df,feats)
    for seed in [4590]:
        for network in [ann,ann2,ann3,ffnn,ffnn2]:
            print(p,seed,network)   
            cv_nn(train_df,test_df,feats,seed,network,p)
