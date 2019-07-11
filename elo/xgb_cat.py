%%time
import xgboost as xgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
from scipy import sparse
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.feature_selection import chi2, SelectPercentile

def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

drop_features=['card_id', 'target', 'outliers',
              ]


train_df = df[df['target'].notnull()]
test_df = df[df['target'].isnull()]



feats = [f for f in train_df.columns if f not in drop_features
        ]


# outlier tag
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1


cat_features = [c for c in feats if 'feature_' in c]
n_splits= 5

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4590)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
auc_list = []
logloss_list = []
print ('feats:' + str(len(feats)))
    
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
  
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['outliers'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['outliers'].iloc[valid_idx] 
    
#     train_x = pd.concat([train_x,pl[feats]],axis=0)
#     train_y = pd.concat([train_y,pl['target']],axis=0)
    
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
            'eta':0.005,
            'num_leaves': 350,
            'max_depth': 8,#6,
            'alpha': 1,
            'lambda': 20,
            'gamma': 0.1,
            'colsample_bytree': 0.2,#
            'colsample_bylevel': 0.3,
            'max_bin': 300,
            #'subsample': 0.85,#bagging
            #'subsample_freq': 1,#bagging_freq
            #'min_gain_to_split': 0.1,
            'scale_pos_weight': 10,
            'min_child_weight': 30,
            'task': 'train',
            'booster': 'gbtree',
            'eval_metric': 'auc',
            'objective': 'rank:pairwise'
    }  

    if n_fold >= 0:
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dval = xgb.DMatrix(valid_x, label=valid_y)   
        dtest = xgb.DMatrix(test_df[feats])
        watchlist = [(dtrain, 'train'), (dval, 'valid')]

        bst = xgb.train(params, dtrain, 10000, watchlist, early_stopping_rounds=200, verbose_eval=100)
        
        oof_preds[valid_idx] = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
        oof_logloss = log_loss(valid_y,  oof_preds[valid_idx])
        logloss_list.append(oof_logloss)
        print (logloss_list)        
        oof_auc = roc_auc_score(valid_y,  oof_preds[valid_idx])
        auc_list.append(oof_auc)
        print (auc_list)
        sub_preds += bst.predict(dtest, ntree_limit=bst.best_ntree_limit) / folds.n_splits

logloss = log_loss(train_df['outliers'], oof_preds)
auc = roc_auc_score(train_df['outliers'], oof_preds)
print('Full OOF LOGLOSS %.6f' % logloss) 
print('Full OOF AUC %.6f' % auc)  



%%time
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
import catboost as cb

def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

# outlier
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1

n_splits= 5

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4590)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
auc_list = []
logloss_list = []
print ('feats:' + str(len(feats)))

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
     
    trn_data = cb.Pool(train_df[feats].iloc[train_idx], train_df['outliers'].iloc[train_idx])
    val_data = cb.Pool(train_df[feats].iloc[valid_idx], train_df['outliers'].iloc[valid_idx])
    print("Train Index:",train_idx,",Val Index:",valid_idx)
    

    if n_fold >= 0:
        num_round = 10000
        cb_model = cb.CatBoostClassifier(learning_rate=0.005, iterations=num_round, verbose=True, #rsm=0.25,
                              use_best_model=True, l2_leaf_reg=20, allow_writing_files=False, metric_period=50,
                              random_seed=4590, depth=8,od_wait=100,od_type='Iter', loss_function='Logloss', eval_metric='AUC')#
        cb_model.fit(trn_data, verbose_eval = 100, eval_set = val_data)

        oof_preds[valid_idx] = cb_model.predict(train_df[feats].iloc[valid_idx])
#         oof_logloss = log_loss(valid_y,  oof_preds[valid_idx])
#         logloss_list.append(oof_logloss)
#         print (logloss_list)        
#         oof_auc = roc_auc_score(valid_y,  oof_preds[valid_idx])
#         auc_list.append(oof_auc)
#         print (auc_list)
        sub_preds += cb_model.predict(test_df[feats]) / folds.n_splits # test_df_new

logloss = log_loss(train_df['outliers'], oof_preds)
auc = roc_auc_score(train_df['outliers'], oof_preds)
print('Full OOF LOGLOSS %.6f' % logloss) 
print('Full OOF AUC %.6f' % auc)  
