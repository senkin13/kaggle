import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime


data = {
    'tra': pd.read_csv('input/air_visit_data.csv'),
    'as': pd.read_csv('input/air_store_info.csv'),
    'hs': pd.read_csv('input/hpg_store_info.csv'),
    'ar': pd.read_csv('input/air_reserve.csv'),
    'hr': pd.read_csv('input/hpg_reserve.csv'),
    'id': pd.read_csv('input/store_id_relation.csv'),
    'tes': pd.read_csv('input/sample_submission.csv'),
    'hol': pd.read_csv('input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }


data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

#sure it can be compressed...
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
# NEW FEATURES FROM Georgii Vyshnia
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
lbl = preprocessing.LabelEncoder()
for i in range(4):
    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_area_name' +str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

# NEW FEATURES FROM JMBULL
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

test_index = test['id']
#test_index_idx = test.index
train['visitors'] = np.log1p(train['visitors'].values)

#### preprocess steps to feed into the network #################
#################################################################

train['cat'] = 'train'
test['cat'] = 'test'
    
hot_enc_cols_cat = ['air_genre_name0','air_genre_name1','air_genre_name2','air_genre_name3',
                 'air_area_name0','air_area_name1','air_area_name2','air_area_name3',
                 'air_genre_name','air_area_name','day_of_week','dow','year','month']

full_df = pd.concat((train,test), axis=0, ignore_index=False)
    
df_index = full_df.index
    
full_df = pd.get_dummies(full_df, columns=hot_enc_cols_cat)

scale_cols = ['lon_plus_lat','var_max_long','var_max_lat','date_int','total_reserv_dt_diff_mean','total_reserv_mean',
             'total_reserv_sum','rs1_x','rv1_x','rs2_x','rv2_x','rs1_y','rs2_y','rv2_y','latitude','longitude',
             'count_observations','max_visitors','median_visitors','min_visitors','holiday_flg','rv1_y',
              'mean_visitors','air_store_id2','date_int','var_max_long']

full_df = full_df.fillna(0)

from scipy.special import erfinv
def rank_gauss(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

for coln in scale_cols:
    full_df[coln] = rank_gauss(np.array(full_df[coln]))

full_df.index = df_index    
        
train = full_df[full_df['cat']=='train']
test = full_df[full_df['cat']=='test']    
    
drop_cols = ['cat','id', 'air_store_id', 'visit_date','visitors','longitude','latitude']

targets = train['visitors']
train = train.drop(train[drop_cols],axis=1)
test = test.drop(test[drop_cols],axis=1)

print('Pre-processing done!')

print('train',train.shape)
print('test',test.shape)
print(targets.shape)

from sklearn.model_selection import train_test_split
train, valid, y_train, y_valid = train_test_split(train, targets, test_size=0.2, random_state=137)

##### 2 hidden layer network, relu activations, adam optimizer, mse loss function #####
#######################################################################################

np.random.seed(0)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
import h5py
from keras import backend
from tensorflow import set_random_seed
set_random_seed(99)
    
def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-05)
    
stop_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
filepath = 'best_wt_recruit_new.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_loss',save_best_only=True,mode='min')
    
dropt = .25

model = Sequential()
model.add(Dense(300,activation='relu',input_shape=(train.shape[1],)))
model.add(Dropout(dropt)) 
model.add(Dense(300,activation='relu'))
model.add(Dropout(dropt))    
model.add(Dense(1,activation='relu'))

model.compile(loss='mse', optimizer=adam)
# fit network
model.fit(np.array(train), np.array(y_train), epochs=100, batch_size=256, validation_data=(np.array(valid), np.array(y_valid)),
            verbose=2, callbacks=[stop_callback, checkpoint], shuffle=False)

model.load_weights('best_wt_recruit_new.hdf5')

# get validation score
pred = np.exp(model.predict(np.array(valid)))
score = rmsle(np.exp(np.array(y_valid)), pred)

print('score:',score)

# get predictions
prediction = np.exp(model.predict(np.array(test)))

nn_df = pd.DataFrame(prediction,columns=['visitors'],index=test_index)
print(nn_df.head())
nn_df.to_csv('nn.csv')

print('done')
