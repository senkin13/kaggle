from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsRegressor
def cv_ridge(train_df,test_df,feats,bag):
    train_df['label'] = np.sqrt(train_df['label'])
    print ('feats:' + str(train_df[feats].shape[1] ))
    n_splits= 5
    folds = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    cv_list = []
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['label'],groups=train_df[bag])):      
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['label'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['label'].iloc[valid_idx] 
        print("Train Shape:",train_x.shape,",Val Shape:",valid_x.shape)
        print("Fold:" + str(n_fold))
        clf = Ridge(alpha=25,fit_intercept=True,normalize=False,copy_X=True,max_iter=None,tol=0.001,solver='auto')
        clf.fit(train_x.values,train_y)
        oof_preds[valid_idx] = clf.predict(valid_x.values) 
        oof_cv = mean_absolute_error(np.power(valid_y,2),  np.power(oof_preds[valid_idx],2))
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += np.power(clf.predict(test_df[feats].values) ,2) / folds.n_splits 

    cv = mean_absolute_error(np.power(train_df['label'],2),  np.power(oof_preds,2))
    print('Full OOF MAE %.6f' % cv)  

    oof_df['ridge'] = np.power(oof_preds,2)
    oof_df[['index','ridge']].to_csv('../ensemble/ridge_oof_' + str(bag) + '.csv',index=False)
    
    submission['ridge'] = sub_preds
    submission[['seg_id','ridge']].to_csv('../ensemble/ridge_pred_' + str(bag) + '.csv',index=False)

def cv_svm(train_df,test_df,feats,bag):
    train_df['label'] = np.sqrt(train_df['label'])
    print ('feats:' + str(train_df[feats].shape[1] ))
    n_splits= 5
    folds = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    cv_list = []
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['label'],groups=train_df[bag])):      
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['label'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['label'].iloc[valid_idx] 
        print("Train Shape:",train_x.shape,",Val Shape:",valid_x.shape)
        print("Fold:" + str(n_fold))
        clf = Ridge(alpha=25,fit_intercept=True,normalize=False,copy_X=True,max_iter=None,tol=0.001,solver='auto')
        clf.fit(train_x.values,train_y)
        oof_preds[valid_idx] = clf.predict(valid_x.values) 
        oof_cv = mean_absolute_error(np.power(valid_y,2),  np.power(oof_preds[valid_idx],2))
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += np.power(clf.predict(test_df[feats].values) ,2) / folds.n_splits 

    cv = mean_absolute_error(np.power(train_df['label'],2),  np.power(oof_preds,2))
    print('Full OOF MAE %.6f' % cv)  

    oof_df['ridge'] = np.power(oof_preds,2)
    oof_df[['index','ridge']].to_csv('../ensemble/ridge_oof_' + str(bag) + '.csv',index=False)
    
    submission['ridge'] = sub_preds
    submission[['seg_id','ridge']].to_csv('../ensemble/ridge_pred_' + str(bag) + '.csv',index=False)
    
# Ridge
train_df,test_df = train.copy(),test.copy()
#for bag in ['bag1','bag2','bag3','bag4','bag5']:
for bag in ['bag']:    
    cv_ridge(train_df,test_df,feats,bag)       
