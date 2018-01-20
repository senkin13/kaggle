import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

print ('Loading Data')
air_visit = pd.read_csv('input/air_visit_data.csv', parse_dates=['visit_date'], converters={'visitors': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0})
air_store = pd.read_csv('input/air_store_info.csv')
hpg_store = pd.read_csv('input/hpg_store_info.csv')
air_reserve = pd.read_csv('input/air_reserve.csv', parse_dates=['visit_datetime','reserve_datetime'])
hpg_reserve = pd.read_csv('input/hpg_reserve.csv', parse_dates=['visit_datetime','reserve_datetime'])
store_id = pd.read_csv('input/store_id_relation.csv')
date_info = pd.read_csv('input/date_info.csv', parse_dates=['calendar_date']).rename(columns={'calendar_date':'visit_date'})
sub = pd.read_csv('input/sample_submission.csv')

# train data
print ('Preprocessing Data')

air_visit['year'] = air_visit['visit_date'].dt.year
air_visit['month'] = air_visit['visit_date'].dt.month
air_visit['dow'] = air_visit['visit_date'].dt.dayofweek

prep_df  = pd.merge(air_visit, air_store,  how='inner', on='air_store_id')
prep_df  = pd.merge(prep_df , date_info, how='left',  on='visit_date')
prep_df  = prep_df.drop(['day_of_week'],axis=1)

tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].min().reset_index().rename(columns={'visitors':'min_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id','dow']) 
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].mean().reset_index().rename(columns={'visitors':'mean_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id','dow'])
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].median().reset_index().rename(columns={'visitors':'median_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id','dow'])
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].max().reset_index().rename(columns={'visitors':'max_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id','dow'])
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].count().reset_index().rename(columns={'visitors':'count_observations'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id','dow']) 
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].std().reset_index().rename(columns={'visitors':'std_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id','dow']) 
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].skew().reset_index().rename(columns={'visitors':'skew_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id','dow']) 



prep_df['var_max_lat'] = prep_df['latitude'].max() - prep_df['latitude']
prep_df['var_max_long'] = prep_df['longitude'].max() - prep_df['longitude']
prep_df['lon_plus_lat'] = prep_df['longitude'] + prep_df['latitude'] 

#prep_df['visitors'] = np.log1p(prep_df['visitors'])

# lable encoder for object
def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = LabelEncoder()
            df[c] = lbl.fit_transform(df[c])
            print(c)
    return df

prep_df = df_lbl_enc(prep_df)
prep_df_ohe = pd.get_dummies(prep_df, columns=['air_store_id','air_genre_name','air_area_name'], drop_first=True, sparse=True)
train = prep_df_ohe.drop(['visit_date','visitors'], axis=1)
label = prep_df_ohe['visitors'].values

# test data
sub['air_store_id'] = sub['id'].map(lambda x: '_'.join(x.split('_')[:2]))
sub['visit_date'] = pd.to_datetime(sub['id'].map(lambda x: str(x).split('_')[2]))

test_id = sub['id']
sub.drop('id', axis=1, inplace=True)
sub['year'] = sub['visit_date'].dt.year
sub['month'] = sub['visit_date'].dt.month
sub['dow'] = sub['visit_date'].dt.dayofweek

predict_data  = pd.merge(sub, air_store,  how='inner', on='air_store_id')
predict_data  = pd.merge(predict_data , date_info, how='left',  on='visit_date')
predict_data  = predict_data.drop(['day_of_week'],axis=1)

tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].min().reset_index().rename(columns={'visitors':'min_visitors'})
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id','dow']) 
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].mean().reset_index().rename(columns={'visitors':'mean_visitors'})
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id','dow'])
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].median().reset_index().rename(columns={'visitors':'median_visitors'})
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id','dow'])
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].max().reset_index().rename(columns={'visitors':'max_visitors'})
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id','dow'])
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].count().reset_index().rename(columns={'visitors':'count_observations'})
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id','dow']) 
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].std().reset_index().rename(columns={'visitors':'std_visitors'})
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id','dow']) 
tmp = air_visit.groupby(['air_store_id','dow'])['visitors'].skew().reset_index().rename(columns={'visitors':'skew_visitors'})
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id','dow']) 


predict_data['var_max_lat'] = predict_data['latitude'].max() - predict_data['latitude']
predict_data['var_max_long'] = predict_data['longitude'].max() - predict_data['longitude']
predict_data['lon_plus_lat'] = predict_data['longitude'] + predict_data['latitude']

predict_data = df_lbl_enc(predict_data)
predict_data_ohe = pd.get_dummies(predict_data, columns=['air_store_id','air_genre_name','air_area_name'], drop_first=True, sparse=True)
X_test = predict_data_ohe.drop(['visit_date','visitors'], axis=1)

print ('Preprocessing Finished')

from sklearn.cross_validation import train_test_split  
X_train, X_valid, y_train, y_valid = train_test_split(train, label, test_size=0.15, random_state=0)
import lightgbm as lgb

params = {
    'num_leaves': 65,
    'objective': 'regression',
    'min_data_in_leaf': 250,
    'max_bin': 256,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'rmse',
    'num_threads': 8   
}

MAX_ROUNDS = 40000

dtrain = lgb.Dataset(
        X_train, label=y_train
)
dval = lgb.Dataset(
        X_valid, label=y_valid, reference=dtrain)
bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100
)
val_pred = bst.predict(
        X_valid, num_iteration=bst.best_iteration or MAX_ROUNDS)
test_pred = bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True)))

from sklearn.metrics import mean_squared_error
def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5

score = RMSLE(y_valid, val_pred)
print('score:',score)

result = pd.DataFrame({"id": test_id, "visitors": test_pred})   
result['visitors'] = np.expm1(result['visitors'])
result.to_csv('LGB_sub.csv', index=False)
print('Done')

