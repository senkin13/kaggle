def cv_cb(train_df,test_df,feats,bag):
    train_df['label'] = np.sqrt(train_df['label'])
    print ('feats:' + str(train_df[feats].shape[1] ))
    n_splits= 5
    folds = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    cv_list = []
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['label'],groups=train_df[bag])):      
        trn_data = cb.Pool(train_df[feats].iloc[train_idx], train_df['label'].iloc[train_idx])
        val_data = cb.Pool(train_df[feats].iloc[valid_idx], train_df['label'].iloc[valid_idx])
        valid_y = train_df['label'].iloc[valid_idx]  

        print("Fold:" + str(n_fold))
        cb_model = cb.CatBoostRegressor(learning_rate=0.02, iterations=10000, verbose=100,min_data_in_leaf=5, 
                              use_best_model=True, l2_leaf_reg=10, task_type='GPU',allow_writing_files=False, metric_period=100,
                              random_seed=817, depth=10, loss_function='MAE', od_wait=100, od_type='Iter')
        cb_model.fit(trn_data, verbose_eval = 100, eval_set = val_data)
        
        oof_preds[valid_idx] = cb_model.predict(train_df[feats].iloc[valid_idx])  
        oof_cv = mean_absolute_error(np.power(valid_y,2),  np.power(oof_preds[valid_idx],2))
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += np.power(cb_model.predict(test_df[feats]) ,2) / folds.n_splits 

    cv = mean_absolute_error(np.power(train_df['label'],2),  np.power(oof_preds,2))
    print('Full OOF MAE %.6f' % cv)  

    oof_df['cb_' + str(bag)] = np.power(oof_preds,2)
    
    submission['cb_' + str(bag)] = sub_preds

# CatBoost
print ('===============CATBOOST==================')
for bag in ['bag0','bag1','bag2','bag3','bag4']: 
    print ('+++++++++++' + str(bag) + '+++++++++++' )
    train_df,test_df = train.copy(),test.copy()
    cv_cb(train_df,test_df,feats,bag) 
    
oof_df['cb_avg'] = (oof_df['cb_bag0'] + oof_df['cb_bag1'] + oof_df['cb_bag2'] + oof_df['cb_bag3'] + oof_df['cb_bag4']) / 5
print('CB Bag Average MAE %.6f' % mean_absolute_error(oof_df['label'],  oof_df['cb_avg'])) 
oof_df.to_csv('oof_cat.csv',index=False)

submission['cb_avg'] = (submission['cb_bag0'] + submission['cb_bag1'] + submission['cb_bag2'] + submission['cb_bag3'] + submission['cb_bag4']) / 5
submission['time_to_failure'] = submission['cb_avg']
submission[['seg_id','time_to_failure']].to_csv('submission.csv',index=False)
