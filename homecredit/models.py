import lightgbm as lgb

train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
#del df
gc.collect()
# Cross validation model

folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=666)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
feature_importance_df = pd.DataFrame()
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index'] 
]
    #        and f not in cols_to_drop
print ('feats:' + str(len(feats)))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
        'nthread': 32,
        'boosting_type': 'goss',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'num_leaves': 70,  
        'max_depth': 9,  
        'subsample': 1,  
        'colsample_bytree': 0.08,
        'min_split_gain': 0.09,
        'min_child_weight': 9.5,        
        'reg_alpha': 1,  
        'reg_lambda': 50,  
        'verbose': 1
    }    

    if n_fold >= 0:
        dtrain = lgb.Dataset(
            train_x, label=train_y)
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain)    
        bst = lgb.train(
            params, dtrain, num_boost_round=10000,
            valid_sets=[dval], early_stopping_rounds=300, verbose_eval=100)
        
        tmp_valid = bst.predict(valid_x, num_iteration=bst.best_iteration)
        tmp_valid.dump('../input/kfold_valid_' + str(n_fold) + '.pkl')
        
        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        
        
        tmp = bst.predict(test_df[feats], num_iteration=bst.best_iteration)
        tmp.dump('../input/kfold_' + str(n_fold) + '.pkl')
        sub_preds += bst.predict(test_df[feats], num_iteration=bst.best_iteration) / folds.n_splits
    
        # Make the feature importance dataframe
        gain = bst.feature_importance('gain')
        fold_importance_df = pd.DataFrame({'feature':bst.feature_name(),
                                        'split':bst.feature_importance('split'),
                                        'gain':100*gain/gain.sum(),
                                        'fold':n_fold,                        
                                        }).sort_values('gain',ascending=False)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    
        #new_list = sorted(
        #    zip(feats, clf.feature_importances_),
        #    key=lambda x: x[1], reverse=True)[:1]
        #for item in new_list:
        #    print (item) 
            
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del bst, train_x, train_y, valid_x, valid_y
        gc.collect()

print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

app_train = pd.read_csv('../input/application_train.csv', usecols=['SK_ID_CURR','TARGET'])
app_test = pd.read_csv('../input/application_test.csv', usecols=['SK_ID_CURR'])

oof = pd.DataFrame({"SK_ID_CURR":app_train["SK_ID_CURR"], "TARGET":oof_preds})
preds = pd.DataFrame({"SK_ID_CURR":app_test["SK_ID_CURR"], "TARGET":sub_preds})

oof.to_csv("../models/lgb_goss_" + str(roc_auc_score(app_train['TARGET'], oof_preds)) + "_valid.csv", index=False)
preds.to_csv("../models/lgb_goss_" + str(roc_auc_score(app_train['TARGET'], oof_preds)) + ".csv", index=False)


import xgboost as xgb

train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
#del df
gc.collect()
# Cross validation model

folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=666)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
feature_importance_df = pd.DataFrame()
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index'] 
]
    #        and f not in cols_to_drop
print ('feats:' + str(len(feats)))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
            'nthread': 32,
            'eta':0.05,#0.05,
            'num_leaves': 255,
            'max_depth': 8,#6,
            'alpha': 10,
            'lambda': 200,
            'gamma': 0.1,
            'colsample_bytree': 0.1,#0.109,
            #'colsample_bylevel': 0.6,
            'max_bin': 256,
            #'subsample': 0.85,#bagging
            #'subsample_freq': 1,#bagging_freq
            #'min_gain_to_split': 0.1,
            'min_child_weight': 30,
            #'silent': -1,
            #'verbose:' -1,
            'task': 'train',
            'booster': 'gbtree',
            'eval_metric': 'auc',
            'objective': 'binary:logistic'
    }  

    if n_fold >= 5:
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dval = xgb.DMatrix(valid_x, label=valid_y)   
        dtest = xgb.DMatrix(test_df[feats])
        watchlist = [(dtrain, 'train'), (dval, 'valid')]

        bst = xgb.train(params, dtrain, 10000, watchlist, early_stopping_rounds=200, verbose_eval=100)
        
        tmp_valid = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
        tmp_valid.dump('../input/xgb_kfold_valid_' + str(n_fold) + '.pkl')
        
        oof_preds[valid_idx] = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
        
        tmp = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
        tmp.dump('../input/xgb_kfold_' + str(n_fold) + '.pkl')

        sub_preds += bst.predict(dtest, ntree_limit=bst.best_ntree_limit) / folds.n_splits
        

            
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del bst, train_x, train_y, valid_x, valid_y
        gc.collect()

print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

from catboost import Pool, CatBoostClassifier

train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
#del df
gc.collect()
# Cross validation model

folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=666)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
feature_importance_df = pd.DataFrame()
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
print ('feats:' + str(len(feats)))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    clf = CatBoostClassifier(
        iterations=7500,
        learning_rate=0.02,
        depth=6,
        bootstrap_type='Bernoulli',
        l2_leaf_reg=50,
        #loss_function='auc',
        eval_metric='AUC',
        verbose=True,)

    train_pool = Pool(train_x, train_y)
    validate_pool = Pool(valid_x, valid_y)

    clf.fit(train_pool, use_best_model=True, eval_set=validate_pool)

    oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
    sub_preds += clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits
    

            
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del clf, train_x, train_y, valid_x, valid_y
    gc.collect()

print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

oof = pd.DataFrame({"SK_ID_CURR":train_df["SK_ID_CURR"], "TARGET":oof_preds})
preds = pd.DataFrame({"SK_ID_CURR":test_df["SK_ID_CURR"], "TARGET":sub_preds})

oof.to_csv("../stacking/cat_cv_valid.csv", index=False)
preds.to_csv("../stacking/cat_cv.csv", index=False)


####load oof dump file
import lightgbm as lgb

train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
del df,test_df
gc.collect()

folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=666)
# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
#sub_preds = np.zeros(test_df.shape[0])
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
         
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

       
    oof_preds[valid_idx] = pickle.load(open('../input/kfold_valid_' + str(n_fold) + '.pkl','rb'))
    del train_x, train_y, valid_x, valid_y
    gc.collect()    

print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))    

k0 = pickle.load(open('../input/kfold_0.pkl','rb'))
k1 = pickle.load(open('../input/kfold_1.pkl','rb'))
k2 = pickle.load(open('../input/kfold_2.pkl','rb'))
k3 = pickle.load(open('../input/kfold_3.pkl','rb'))
k4 = pickle.load(open('../input/kfold_4.pkl','rb'))
k5 = pickle.load(open('../input/kfold_5.pkl','rb'))
k6 = pickle.load(open('../input/kfold_6.pkl','rb'))
k7 = pickle.load(open('../input/kfold_7.pkl','rb'))
k8 = pickle.load(open('../input/kfold_8.pkl','rb'))
k9 = pickle.load(open('../input/kfold_9.pkl','rb'))

app_train = pd.read_csv('../input/application_train.csv', usecols=['SK_ID_CURR'])
app_test = pd.read_csv('../input/application_test.csv', usecols=['SK_ID_CURR'])

oof = pd.DataFrame({"SK_ID_CURR":app_train["SK_ID_CURR"], "TARGET":oof_preds})
oof.to_csv("../models/merge_lgb_all_v6_0.802195_valid.csv", index=False)

preds = pd.DataFrame({"SK_ID_CURR":app_test["SK_ID_CURR"], "k0":k0,"k1":k1,"k2":k2,"k3":k3,"k4":k4,
                     "k5":k5,"k6":k6,"k7":k7,"k8":k8,"k9":k9})
preds['TARGET'] = (preds.k0 + preds.k1 + preds.k2 + preds.k3 + preds.k4 + preds.k5 + preds.k6 + preds.k7 + preds.k8 + preds.k9)/10
preds[['SK_ID_CURR','TARGET']].to_csv("../models/merge_lgb_all_v6_0.802195.csv", index=False)
