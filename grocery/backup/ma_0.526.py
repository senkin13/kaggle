# -*- coding: utf-8 -*-
import pandas as pd
from datetime import timedelta

dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32', 'onpromotion':'bool'}

train = pd.read_csv('../input/train.csv', usecols=[1,2,3,4,5], dtype=dtypes, parse_dates=['date'],
                    skiprows=range(1, 86672217) #Skip dates before 2016-08-01
                    )
                    
# adjustment for unit sales with promotion
Kpromo = 1.5

train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
train.loc[(train.onpromotion == True), 'unit_sales'] *= 1.0/Kpromo # reduce unit sales by 1/Kpromo
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
train['dow'] = train['date'].dt.dayofweek

#Days of Week Means
#By tarobxl: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/42948
ma_dw = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(
        ['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw').reset_index()
ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(
        ['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk').reset_index()

train.drop('dow',1,inplace=True)

# creating records for all items, in all markets on all dates
# for correct calculation of daily unit sales averages.
u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()
train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)
train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
).reset_index()

del u_dates, u_stores, u_items

train.loc[:, 'unit_sales'].fillna(0, inplace=True) # fill NaNs
lastdate = train.iloc[train.shape[0]-1].date

#Moving Averages
ma_is = train[['item_nbr','store_nbr','unit_sales']].groupby(
        ['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais')

for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')

del tmp,tmpg,train

ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)
ma_is.drop(list(ma_is.columns.values)[3:],1,inplace=True)

#Load test
test = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date'])
test['dow'] = test['date'].dt.dayofweek
test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])

del ma_is, ma_wk, ma_dw

#Forecasting Test
test['unit_sales'] = test.mais 
pos_idx = test['mawk'] > 0
test_pos = test.loc[pos_idx]
test.loc[pos_idx, 'unit_sales'] = test_pos['mais'] * test_pos['madw'] / test_pos['mawk']
test.loc[:, "unit_sales"].fillna(0, inplace=True)
test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1) # restoring unit values 

#50% more for promotion items
test.loc[test['onpromotion'] == True, 'unit_sales'] *= Kpromo

test[['id','unit_sales']].to_csv('ma.csv', index=False, float_format='%.4f')
