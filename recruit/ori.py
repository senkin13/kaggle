import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime

print ('Loading Data')
air_visit = pd.read_csv('input/air_visit_data.csv', parse_dates=['visit_date'])
air_store = pd.read_csv('input/air_store_info.csv')
hpg_store = pd.read_csv('input/hpg_store_info.csv')
air_reserve = pd.read_csv('input/air_reserve.csv', parse_dates=['visit_datetime','reserve_datetime'])
hpg_reserve = pd.read_csv('input/hpg_reserve.csv', parse_dates=['visit_datetime','reserve_datetime'])
store_id = pd.read_csv('input/store_id_relation.csv')
date_info = pd.read_csv('input/date_info.csv', parse_dates=['calendar_date']).rename(columns={'calendar_date':'visit_date'})
sub = pd.read_csv('input/sample_submission.csv')

# merge hpg and air reserve
hpg_reserve = pd.merge(hpg_reserve, store_id, how='inner', on=['hpg_store_id'])
hpg_reserve.drop(['hpg_store_id'], axis=1, inplace=True)
air_reserve = air_reserve.append(hpg_reserve)
air_reserve['visit_date'] = air_reserve['visit_datetime'].dt.date
#air_reserve['reserve_diff'] = air_reserve.apply(lambda r:(r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
air_reserve.drop(['reserve_datetime', 'visit_datetime'], axis=1, inplace=True)
air_reserve = air_reserve.groupby(['air_store_id','visit_date'],as_index=False).sum().reset_index()

# test data
sub['air_store_id'] = sub['id'].map(lambda x: '_'.join(x.split('_')[:2]))
sub['visit_date'] = sub['id'].map(lambda x: str(x).split('_')[2])
test_id = sub['id']
sub.drop('id', axis=1, inplace=True)

print ('Loading Finished')
