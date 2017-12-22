import gc
import numpy as np
import pandas as pd
from datetime import timedelta

dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32'}
train = pd.read_csv('input/train.csv', usecols=[1,2,3,4], dtype=dtypes, parse_dates=['date'],skiprows=range(1, 66458909) #Skip dates before 2016-08-01
train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
train['dow'] = train['date'].dt.dayofweek

u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()
train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)
train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
)

del u_dates, u_stores, u_items
gc.collect()

train.loc[:, 'unit_sales'].fillna(0, inplace=True) # fill NaNs
train.reset_index(inplace=True)
lastdate = train.iloc[train.shape[0]-1].date

ma_dw = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw')
ma_dw.reset_index(inplace=True)

ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk')
ma_wk.reset_index(inplace=True)

ma_is = train[['item_nbr','store_nbr','unit_sales']].groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais226')
for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')
   
ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)

test = pd.read_csv('input/test.csv', dtype=dtypes, parse_dates=['date'])
test['dow'] = test['date'].dt.dayofweek
test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])

#Forecasting Test
test['unit_sales'] = test.mais
pos_idx = test['mawk'] > 0
test_pos = test.loc[pos_idx]
test.loc[pos_idx, 'unit_sales'] = test_pos['mais'] * test_pos['madw'] / test_pos['mawk']
test.loc[:,'unit_sales'].fillna(0, inplace=True)
test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1)

holiday = pd.read_csv('input/holidays_events.csv', parse_dates=['date'])
holiday = holiday.loc[holiday['transferred'] == False]
test = pd.merge(test, holiday, how = 'left', on =['date'] )
test['transferred'].fillna(True, inplace=True)

test.loc[test['transferred'] == False, 'unit_sales'] *= 1.2
test.loc[test['onpromotion'] == True, 'unit_sales'] *= 1.5
test[['id','unit_sales']].to_csv('ma.csv', index=False, float_format='%.4f')
