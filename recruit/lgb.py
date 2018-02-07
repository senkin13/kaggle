import numpy as np
import pandas as pd
from sklearn import *
import datetime
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas.tseries.offsets as offsets
from scipy.stats import hmean
#, parse_dates=['visit_date']
print ('Loading Data')
air_visit = pd.read_csv('input/air_visit_data.csv', converters={'visitors': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0})  
air_store = pd.read_csv('input/air_store_info.csv')
hpg_store = pd.read_csv('input/hpg_store_info.csv')
air_reserve = pd.read_csv('input/air_reserve.csv', converters={'reserve_visitors': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0})
hpg_reserve = pd.read_csv('input/hpg_reserve.csv', converters={'reserve_visitors': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0})
store_id = pd.read_csv('input/store_id_relation.csv')
date_info = pd.read_csv('input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
sub = pd.read_csv('input/sample_submission.csv')
weather = pd.read_csv('input/weather.csv')
print ('Loading Finished')

# hpg reserve
hpg_reserve = pd.merge(hpg_reserve, store_id, how='inner', on=['hpg_store_id'])
hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'])
hpg_reserve['visit_datetime'] = hpg_reserve['visit_datetime'].dt.date
hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve['reserve_datetime'])
hpg_reserve['reserve_datetime'] = hpg_reserve['reserve_datetime'].dt.date
hpg_reserve['reserve_datetime_diff'] = hpg_reserve.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,axis=1)

tmp1 = hpg_reserve.groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff','reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'reserve_diff_sum_hpg', 'reserve_visitors':'reserve_visitors_sum_hpg'})
tmp2 = hpg_reserve.groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'reserve_diff_mean_hpg', 'reserve_visitors':'reserve_visitors_mean_hpg'})
hpg_reserve = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date']) 

hpg_reserve['visit_date'] = hpg_reserve['visit_date'].astype(str)

# air reserve
air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])
air_reserve['visit_datetime'] = air_reserve['visit_datetime'].dt.date
air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])
air_reserve['reserve_datetime'] = air_reserve['reserve_datetime'].dt.date
air_reserve['reserve_datetime_diff'] = air_reserve.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,axis=1)

tmp1 = air_reserve.groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff','reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'reserve_diff_sum_air', 'reserve_visitors':'reserve_visitors_sum_air'})
tmp2 = air_reserve.groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'reserve_diff_mean_air', 'reserve_visitors':'reserve_visitors_mean_air'})
air_reserve = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date']) 

air_reserve['visit_date'] = air_reserve['visit_date'].astype(str)

# weather
weather['visit_date'] = pd.to_datetime(weather[['year','month','day']]).astype(str)
#weather['mean_temperature'] = weather['max_temperature']*0.5 + weather['min_temperature']*0.5
weather['max_temperature_lev'] = weather['max_temperature'].map(lambda x:0 if x<15 else 2 if x>25 else 3)
#weather['min_temperature_lev'] = weather['min_temperature'].map(lambda x:0 if x<15 or x>25 else 1)
weather['precipitation_lev'] = weather['precipitation'].map(lambda x:1 if x<5 else 2 if x>=5 and x<30 else 3)
weather.drop(['max_temperature','min_temperature','precipitation','year','month','day'],axis=1, inplace=True)

# store
air_store['air_area_name_province'] = air_store['air_area_name'].map(lambda x: x.split(' ')[0])
air_store['air_area_name_city'] = air_store['air_area_name'].map(lambda x: x.split(' ')[1])
air_store['air_area_name_ward'] = air_store['air_area_name'].map(lambda x: x.split(' ')[2])
air_store.drop('air_area_name', axis=1, inplace=True)

# holiday
#date_info['holiday'] = date_info.apply((lambda x:1 if x.day_of_week=='Friday' or x.day_of_week=='Sunday' or x.day_of_week=='Saturday' else 2 if x.holiday_flg==1 else 0),axis=1).astype(int)
date_info['holiday'] = date_info.apply((lambda x:1 if x.day_of_week=='Sunday' or x.day_of_week=='Saturday' else 2 if x.holiday_flg==1 else 0),axis=1).astype(int)

date_info.drop('holiday_flg', axis=1, inplace=True)

# submission
seasons = [0,0,1,1,1,2,2,2,3,3,3,0] #feb is winter, then spring, summer, fall

sub['air_store_id'] = sub['id'].map(lambda x: '_'.join(x.split('_')[:2]))
sub['visit_date'] = sub['id'].map(lambda x: str(x).split('_')[2])
test_id = sub['id']
sub.drop('id', axis=1, inplace=True)
sub['visit_datetime'] = pd.to_datetime(sub['visit_date'])
sub['month'] = sub['visit_datetime'].dt.month
sub['woy'] = sub['visit_datetime'].dt.weekofyear
sub['doy'] = sub['visit_datetime'].dt.dayofyear
sub['dow'] = sub['visit_datetime'].dt.dayofweek
sub['day'] = sub['visit_datetime'].dt.day
#sub['season'] = sub['month'].map(lambda x: seasons[(x - 1)])
#sub['salary_day'] = sub['day'].map(lambda x:1 if x==10 or x==25 else 0)
sub['date_int'] = sub['visit_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
sub['air_id'] = sub['air_store_id']
sub['vis'] = np.nan
sub.drop('visitors', axis=1, inplace=True)

sub  = pd.merge(sub, air_store,  how='inner', on='air_store_id')
sub  = pd.merge(sub, date_info, how='left',  on='visit_date')
sub  = pd.merge(sub , weather, how='left',  on=['visit_date','air_area_name_province'])
sub = pd.merge(sub, hpg_reserve,  how='left', on=['air_store_id', 'visit_date'])
sub = pd.merge(sub, air_reserve,  how='left', on=['air_store_id', 'visit_date'])

sub['var_max_lat'] = sub['latitude'].max() - sub['latitude']
sub['var_max_long'] = sub['longitude'].max() - sub['longitude']
sub['lat_plus_long'] = sub['latitude'] + sub['longitude']


#sub.drop(['reserve_visitors_sum_hpg','reserve_visitors_mean_hpg','reserve_diff_sum_hpg','reserve_diff_mean_hpg'], axis=1, inplace=True)

# visit
air_visit['visit_datetime'] = pd.to_datetime(air_visit['visit_date'])
air_visit['month'] = air_visit['visit_datetime'].dt.month
air_visit['woy'] = air_visit['visit_datetime'].dt.weekofyear
air_visit['doy'] = air_visit['visit_datetime'].dt.dayofyear
air_visit['dow'] = air_visit['visit_datetime'].dt.dayofweek
air_visit['day'] = air_visit['visit_datetime'].dt.day
#air_visit['season'] = air_visit['month'].map(lambda x: seasons[(x - 1)])
#air_visit['salary_day'] = air_visit['day'].map(lambda x:1 if x==10 or x==25 else 0)
air_visit['date_int'] = air_visit['visit_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
air_visit['air_id'] = air_visit['air_store_id']
air_visit['vis'] = air_visit['visitors']
air_visit.drop('visitors', axis=1, inplace=True)

air_visit  = pd.merge(air_visit, air_store,  how='inner', on='air_store_id')
air_visit  = pd.merge(air_visit , date_info, how='left',  on='visit_date')
air_visit  = pd.merge(air_visit , weather, how='left',  on=['visit_date','air_area_name_province'])
air_visit = pd.merge(air_visit, hpg_reserve,  how='left', on=['air_store_id', 'visit_date'])
air_visit = pd.merge(air_visit, air_reserve,  how='left', on=['air_store_id', 'visit_date'])

air_visit['var_max_lat'] = air_visit['latitude'].max() - air_visit['latitude']
air_visit['var_max_long'] = air_visit['longitude'].max() - air_visit['longitude']
air_visit['lat_plus_long'] = air_visit['latitude'] + air_visit['longitude']

# concat train and test
air = air_visit.append(sub)
air['visit_date'] = pd.to_datetime(air['visit_date'])
#first_day = pd.to_datetime('2016-01-01')
#air['days_since_20160101'] = (air['visit_datetime'] - first_day).apply(lambda dt: dt.days)
def catStrFeatures(df, colname1, colname2, sep='_'):
    series = df[colname1].astype(str).str.cat(df[colname2].astype(str), sep=sep)
    return series
air['area_genre'] = catStrFeatures(air, 'air_area_name_ward', 'air_genre_name')
air['store_weekday'] = catStrFeatures(air, 'air_store_id', 'dow')
air['store_weekday_holiday'] = catStrFeatures(air, 'store_weekday', 'holiday')
#tmp = air[['air_store_id','dow','holiday','vis']].groupby(['air_store_id','dow','holiday'])[['vis']].sum().reset_index().rename(columns={'vis':'vis_dow_hol'})
#air = pd.merge(air,tmp,how='left', on=['air_store_id','dow','holiday'])

# reserve_visitors
reserve_visitors_mean_train = air.set_index(['air_store_id','visit_date'])[['reserve_visitors_mean_air']].unstack(
        level=-1)
reserve_visitors_mean_train.columns = reserve_visitors_mean_train.columns.get_level_values(1)
reserve_visitors_sum_train = air.set_index(['air_store_id','visit_date'])[['reserve_visitors_sum_air']].unstack(
        level=-1)
reserve_visitors_sum_train.columns = reserve_visitors_sum_train.columns.get_level_values(1)

# visitors
visitors_train = air.set_index(['air_store_id','visit_date'])[['vis']].unstack(
        level=-1)
visitors_train.columns = visitors_train.columns.get_level_values(1)
visitors_train_index = visitors_train.reset_index()[['air_store_id']]

# hol visitors
visitors_hol_train = air.set_index(['air_store_id','holiday','visit_date'])[['vis']].unstack(
        level=-1)
visitors_hol_train.columns = visitors_hol_train.columns.get_level_values(1)
visitors_hol_train_index = visitors_hol_train.reset_index()[['air_store_id','holiday']]

# labelencoder 
lbl = LabelEncoder()
air['air_id'] = lbl.fit_transform(air['air_id'])
air['air_genre_name'] = lbl.fit_transform(air['air_genre_name'])
air['air_area_name_province'] = lbl.fit_transform(air['air_area_name_province'])
air['air_area_name_city'] = lbl.fit_transform(air['air_area_name_city'])
air['air_area_name_ward'] = lbl.fit_transform(air['air_area_name_ward'])
air['area_genre'] = lbl.fit_transform(air['area_genre'])
air['store_weekday'] = lbl.fit_transform(air['store_weekday'])
air['store_weekday_holiday'] = lbl.fit_transform(air['store_weekday_holiday'])
#air = pd.get_dummies(air, columns=['air_genre_name','air_area_name_province','air_area_name_city','air_area_name_ward'], drop_first=True, sparse=False)

# ward genre combination
genre_ward_train = air.groupby(['visit_date','air_area_name_ward','air_genre_name'])['vis'].sum().reset_index()
genre_ward_train = genre_ward_train.set_index(['air_area_name_ward','air_genre_name','visit_date'])[['vis']].unstack(
        level=-1)
genre_ward_train.columns = genre_ward_train.columns.get_level_values(1)
genre_ward_train_index = genre_ward_train.reset_index()[['air_area_name_ward','air_genre_name']]


# air id
air_id_train = air.set_index(['air_store_id','visit_date'])[['air_id']].unstack(
        level=-1).fillna(-1)
air_id_train.columns = air_id_train.columns.get_level_values(1)
air_id_train['air_id'] = air_id_train.max(axis=1)
air_id_train = air_id_train['air_id']

# split to train and test
air_visit = air[air['visit_date']<'2017-04-23']
sub = air[air['visit_date']>='2017-04-23']

print ('Processing Finished')

from datetime import date, timedelta
def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(df,dt,tp):
    X = pd.DataFrame({
        "day1": get_timespan(df, dt, 1, 1).values.ravel() ,             
        "visit_date": dt,

        "ahead_diff_mean_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).diff(axis=1).mean(axis=1).values,            
        "ahead_mean_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).mean(axis=1).values,    
        "ahead_median_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).median(axis=1).values,  
        "ahead_max_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).max(axis=1).values,              
        "ahead_std_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).std(axis=1).values,              
        "ahead_skew_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).skew(axis=1).values,  
        "ahead_kurt_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).kurt(axis=1).values,              
        "ahead_count_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).count(axis=1).values,             
            
        "ahead_diff_mean_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).diff(axis=1).mean(axis=1).values,
        "ahead_mean_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).mean(axis=1).values,    
        "ahead_median_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).median(axis=1).values,  
        "ahead_max_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).max(axis=1).values,             
        "ahead_std_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).std(axis=1).values,              
        "ahead_skew_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).skew(axis=1).values,  
        "ahead_kurt_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).kurt(axis=1).values,              
        "ahead_count_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).count(axis=1).values, 

        "ahead_diff_mean_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).diff(axis=1).mean(axis=1).values,            
        "ahead_mean_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).mean(axis=1).values,    
        "ahead_median_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).median(axis=1).values,  
        "ahead_max_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).max(axis=1).values,                
        "ahead_std_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).std(axis=1).values,              
        "ahead_skew_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).skew(axis=1).values,  
        "ahead_kurt_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).kurt(axis=1).values,              
        "ahead_count_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).count(axis=1).values,         
        
        "diff_mean_273_{}".format(tp): get_timespan(df, dt, 273, 273).diff(axis=1).mean(axis=1).values,
        "mean_273_{}".format(tp): get_timespan(df, dt, 273, 273).mean(axis=1).values,    
        "median_273_{}".format(tp): get_timespan(df, dt, 273, 273).median(axis=1).values,  
        "max_273_{}".format(tp): get_timespan(df, dt, 273, 273).max(axis=1).values,               
        "std_273_{}".format(tp): get_timespan(df, dt, 273, 273).std(axis=1).values,              
        "skew_273_{}".format(tp): get_timespan(df, dt, 273, 273).skew(axis=1).values,  
        "kurt_273_{}".format(tp): get_timespan(df, dt, 273, 273).kurt(axis=1).values,              
        "count_273_{}".format(tp): get_timespan(df, dt, 273, 273).count(axis=1).values,  

        "diff_mean_182_{}".format(tp): get_timespan(df, dt, 182, 182).diff(axis=1).mean(axis=1).values,            
        "mean_182_{}".format(tp): get_timespan(df, dt, 182, 182).mean(axis=1).values,    
        "median_182_{}".format(tp): get_timespan(df, dt, 182, 182).median(axis=1).values,  
        "max_182_{}".format(tp): get_timespan(df, dt, 182, 182).max(axis=1).values,              
        "std_182_{}".format(tp): get_timespan(df, dt, 182, 182).std(axis=1).values,              
        "skew_182_{}".format(tp): get_timespan(df, dt, 182, 182).skew(axis=1).values,  
        "kurt_182_{}".format(tp): get_timespan(df, dt, 182, 182).kurt(axis=1).values,              
        "count_182_{}".format(tp): get_timespan(df, dt, 182, 182).count(axis=1).values,  

        "diff_mean_91_{}".format(tp): get_timespan(df, dt, 91, 91).diff(axis=1).mean(axis=1).values,            
        "mean_91_{}".format(tp): get_timespan(df, dt, 91, 91).mean(axis=1).values,    
        "median_91_{}".format(tp): get_timespan(df, dt, 91, 91).median(axis=1).values,  
        "max_91_{}".format(tp): get_timespan(df, dt, 91, 91).max(axis=1).values,            
        "std_91_{}".format(tp): get_timespan(df, dt, 91, 91).std(axis=1).values,              
        "skew_91_{}".format(tp): get_timespan(df, dt, 91, 91).skew(axis=1).values,  
        "kurt_91_{}".format(tp): get_timespan(df, dt, 91, 91).kurt(axis=1).values,              
        "count_91_{}".format(tp): get_timespan(df, dt, 91, 91).count(axis=1).values,   
            
        }) 

    X['diff_mean_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').diff(axis=1).mean(axis=1).values    
    X['mean_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').mean(axis=1).values
    X['median_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').median(axis=1).values
    X['max_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').max(axis=1).values
    X['min_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').min(axis=1).values
    X['std_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').std(axis=1).values
    X['skew_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').skew(axis=1).values
    X['kurt_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').kurt(axis=1).values
    X['count_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').count(axis=1).values 

    X['diff_mean_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').diff(axis=1).mean(axis=1).values    
    X['mean_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').mean(axis=1).values
    X['median_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').median(axis=1).values
    X['max_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').max(axis=1).values
    X['min_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').min(axis=1).values
    X['std_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').std(axis=1).values
    X['skew_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').skew(axis=1).values
    X['kurt_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').kurt(axis=1).values
    X['count_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').count(axis=1).values 

    X['diff_mean_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').diff(axis=1).mean(axis=1).values    
    X['mean_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').mean(axis=1).values
    X['median_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').median(axis=1).values
    X['max_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').max(axis=1).values
    X['min_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').min(axis=1).values
    X['std_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').std(axis=1).values
    X['skew_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').skew(axis=1).values
    X['kurt_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').kurt(axis=1).values
    X['count_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').count(axis=1).values 

# mean dow ratio    
    X['mean_91_dow_ratio_{}'.format(tp)] = X['mean_91_dow_{}'.format(tp)] / X['mean_91_{}'.format(tp)]
    X['mean_182_dow_ratio_{}'.format(tp)] = X['mean_182_dow_{}'.format(tp)] / X['mean_182_{}'.format(tp)]
    X['mean_273_dow_ratio_{}'.format(tp)] = X['mean_273_dow_{}'.format(tp)] / X['mean_273_{}'.format(tp)]
    
# diff between time window
    X['diff_mean_91_182_{}'.format(tp)] = X['mean_91_{}'.format(tp)] - X['mean_182_{}'.format(tp)] 
    X['diff_mean_182_273_{}'.format(tp)] = X['mean_182_{}'.format(tp)] - X['mean_273_{}'.format(tp)]     
    X['diff_ahead_mean_91_182_{}'.format(tp)] = X['ahead_mean_91_{}'.format(tp)] - X['ahead_mean_182_{}'.format(tp)] 
    X['diff_ahead_mean_182_273_{}'.format(tp)] = X['ahead_mean_182_{}'.format(tp)] - X['ahead_mean_273_{}'.format(tp)]  
    
    if tp == 0:
        X['air_id'] = visitors_train_index
        X['reserve_sum_3'] = get_timespan(reserve_visitors_sum_train, dt, 3, 3).sum(axis=1).values  
    if tp == 1:
        X[['air_area_name_ward','air_genre_name']] = genre_ward_train_index[['air_area_name_ward','air_genre_name']]
    
    return X

def prepare_dataset2(df,dt,tp):
    X = pd.DataFrame({
        "day1": get_timespan(df, dt, 1, 1).values.ravel() ,             
        "visit_date": dt,

        "ahead_diff_mean_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).diff(axis=1).mean(axis=1).values,             
        "ahead_mean_273_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 273, 273).mean(axis=1).values,    
   
        "ahead_diff_mean_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).diff(axis=1).mean(axis=1).values,    
        "ahead_mean_182_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 182, 182).mean(axis=1).values,    
          
        "ahead_diff_mean_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).diff(axis=1).mean(axis=1).values,    
        "ahead_mean_91_{}".format(tp): get_timespan(df, dt + timedelta(days=-7), 91, 91).mean(axis=1).values,    
      
        "diff_mean_273_{}".format(tp): get_timespan(df, dt, 273, 273).diff(axis=1).mean(axis=1).values,
        "mean_273_{}".format(tp): get_timespan(df, dt, 273, 273).mean(axis=1).values,    

        "diff_mean_182_{}".format(tp): get_timespan(df, dt, 182, 182).diff(axis=1).mean(axis=1).values,    
        "mean_182_{}".format(tp): get_timespan(df, dt, 182, 182).mean(axis=1).values,    
       
        "diff_mean_91_{}".format(tp): get_timespan(df, dt, 91, 91).diff(axis=1).mean(axis=1).values,    
        "mean_91_{}".format(tp): get_timespan(df, dt, 91, 91).mean(axis=1).values,    
            
        }) 

    X['diff_mean_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').diff(axis=1).mean(axis=1).values
    X['mean_273_dow_{}'.format(tp)] = get_timespan(df, dt, 273, 39, freq='7D').mean(axis=1).values
    
    X['diff_mean_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').diff(axis=1).mean(axis=1).values
    X['mean_182_dow_{}'.format(tp)] = get_timespan(df, dt, 182, 26, freq='7D').mean(axis=1).values
    
    X['diff_mean_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').diff(axis=1).mean(axis=1).values
    X['mean_91_dow_{}'.format(tp)] = get_timespan(df, dt, 91, 13, freq='7D').mean(axis=1).values

# mean dow ratio    
    X['mean_91_dow_ratio_{}'.format(tp)] = X['mean_91_dow_{}'.format(tp)] / X['mean_91_{}'.format(tp)]
    X['mean_182_dow_ratio_{}'.format(tp)] = X['mean_182_dow_{}'.format(tp)] / X['mean_182_{}'.format(tp)]
    X['mean_273_dow_ratio_{}'.format(tp)] = X['mean_273_dow_{}'.format(tp)] / X['mean_273_{}'.format(tp)]
    
# diff between time window
    X['diff_mean_91_182_{}'.format(tp)] = X['mean_91_{}'.format(tp)] - X['mean_182_{}'.format(tp)] 
    X['diff_mean_182_273_{}'.format(tp)] = X['mean_182_{}'.format(tp)] - X['mean_273_{}'.format(tp)]     
    X['diff_ahead_mean_91_182_{}'.format(tp)] = X['ahead_mean_91_{}'.format(tp)] - X['ahead_mean_182_{}'.format(tp)] 
    X['diff_ahead_mean_182_273_{}'.format(tp)] = X['ahead_mean_182_{}'.format(tp)] - X['ahead_mean_273_{}'.format(tp)]  
    
    if tp == 0:
        X['air_id'] = visitors_train_index
        X['reserve_sum_3'] = get_timespan(reserve_visitors_sum_train, dt, 3, 3).sum(axis=1).values  
    if tp == 1:
        X[['air_area_name_ward','air_genre_name']] = genre_ward_train_index[['air_area_name_ward','air_genre_name']]
    
    return X


dt = date(2016, 10, 7)
X_l, y_l = [], []
for i in range(237):
    delta = timedelta(days=i)
    X_tmp = prepare_dataset(
        visitors_train,dt + delta,0
    )
    X_l.append(X_tmp)
all = pd.concat(X_l, axis=0)
all.drop(['day1'],axis=1,inplace=True)
del X_l

for i in range(237):
    delta = timedelta(days=i)
    y_tmp = prepare_dataset2(
        genre_ward_train,dt + delta,1
    )
    y_l.append(y_tmp)
all2 = pd.concat(y_l, axis=0)
all2.drop(['day1'],axis=1,inplace=True)
del y_l


# train
all['air_id'] = lbl.fit_transform(all['air_id'])
air_visit2017 = air_visit[air_visit['visit_date']>'2016-10-6']
air_visit2017['air_id'] = air_visit2017['air_id'].astype(float)
all['visit_date'] = pd.to_datetime(all['visit_date'])
all2['visit_date'] = pd.to_datetime(all2['visit_date'])
#all3['visit_date'] = pd.to_datetime(all3['visit_date'])
air2017 = pd.merge(air_visit2017, all, how='left', on=['visit_date','air_id']) 
air2017 = pd.merge(air2017, all2, how='left', on=['visit_date','air_area_name_ward','air_genre_name']) 
#air2017 = pd.merge(air2017, all3, how='left', on=['visit_date','air_area_name_city','air_genre_name']) 

air2017['ahead_mean_273_genre_ward_ratio'] = air2017['ahead_mean_273_0'] / air2017['ahead_mean_273_1']
air2017['ahead_mean_182_genre_ward_ratio'] = air2017['ahead_mean_182_0'] / air2017['ahead_mean_182_1']
air2017['ahead_mean_91_genre_ward_ratio'] = air2017['ahead_mean_91_0'] / air2017['ahead_mean_91_1']
air2017['mean_273_genre_ward_ratio'] = air2017['mean_273_0'] / air2017['mean_273_1']
air2017['mean_182_genre_ward_ratio'] = air2017['mean_182_0'] / air2017['mean_182_1']
air2017['mean_91_genre_ward_ratio'] = air2017['mean_91_0'] / air2017['mean_91_1']
air2017['mean_273_dow_genre_ward_ratio'] = air2017['mean_273_dow_0'] / air2017['mean_273_dow_1']
air2017['mean_182_dow_genre_ward_ratio'] = air2017['mean_182_dow_0'] / air2017['mean_182_dow_1']
air2017['mean_91_dow_genre_ward_ratio'] = air2017['mean_91_dow_0'] / air2017['mean_91_dow_1']

#air2017['ahead_mean_273_genre_city_ratio'] = air2017['ahead_mean_273_0'] / air2017['ahead_mean_273_2']
#air2017['ahead_mean_182_genre_city_ratio'] = air2017['ahead_mean_182_0'] / air2017['ahead_mean_182_2']
#air2017['ahead_mean_91_genre_city_ratio'] = air2017['ahead_mean_91_0'] / air2017['ahead_mean_91_2']
#air2017['mean_273_genre_city_ratio'] = air2017['mean_273_0'] / air2017['mean_273_2']
#air2017['mean_182_genre_city_ratio'] = air2017['mean_182_0'] / air2017['mean_182_2']
#air2017['mean_91_genre_city_ratio'] = air2017['mean_91_0'] / air2017['mean_91_2']
#air2017['mean_273_dow_genre_city_ratio'] = air2017['mean_273_dow_0'] / air2017['mean_273_dow_2']
#air2017['mean_182_dow_genre_city_ratio'] = air2017['mean_182_dow_0'] / air2017['mean_182_dow_2']
#air2017['mean_91_dow_genre_city_ratio'] = air2017['mean_91_dow_0'] / air2017['mean_91_dow_2']

train = air2017.drop(['air_store_id','visit_date','visit_datetime','vis','day_of_week'], axis=1)
label = air2017['vis'].values

# test
sub['air_id'] = sub['air_id'].astype(float)
X_test = pd.merge(sub, all, how='left', on=['visit_date','air_id']) 
X_test = pd.merge(X_test, all2, how='left', on=['visit_date','air_area_name_ward','air_genre_name']) 
#X_test = pd.merge(X_test, all3, how='left', on=['visit_date','air_area_name_city','air_genre_name']) 

X_test['ahead_mean_273_genre_ward_ratio'] = X_test['ahead_mean_273_0'] / X_test['ahead_mean_273_1']
X_test['ahead_mean_182_genre_ward_ratio'] = X_test['ahead_mean_182_0'] / X_test['ahead_mean_182_1']
X_test['ahead_mean_91_genre_ward_ratio'] = X_test['ahead_mean_91_0'] / X_test['ahead_mean_91_1']
X_test['mean_273_genre_ward_ratio'] = X_test['mean_273_0'] / X_test['mean_273_1']
X_test['mean_182_genre_ward_ratio'] = X_test['mean_182_0'] / X_test['mean_182_1']
X_test['mean_91_genre_ward_ratio'] = X_test['mean_91_0'] / X_test['mean_91_1']
X_test['mean_273_dow_genre_ward_ratio'] = X_test['mean_273_dow_0'] / X_test['mean_273_dow_1']
X_test['mean_182_dow_genre_ward_ratio'] = X_test['mean_182_dow_0'] / X_test['mean_182_dow_1']
X_test['mean_91_dow_genre_ward_ratio'] = X_test['mean_91_dow_0'] / X_test['mean_91_dow_1']

#X_test['ahead_mean_273_genre_city_ratio'] = X_test['ahead_mean_273_0'] / X_test['ahead_mean_273_2']
#X_test['ahead_mean_182_genre_city_ratio'] = X_test['ahead_mean_182_0'] / X_test['ahead_mean_182_2']
#X_test['ahead_mean_91_genre_city_ratio'] = X_test['ahead_mean_91_0'] / X_test['ahead_mean_91_2']
#X_test['mean_273_genre_city_ratio'] = X_test['mean_273_0'] / X_test['mean_273_2']
#X_test['mean_182_genre_city_ratio'] = X_test['mean_182_0'] / X_test['mean_182_2']
#X_test['mean_91_genre_city_ratio'] = X_test['mean_91_0'] / X_test['mean_91_2']
#X_test['mean_273_dow_genre_city_ratio'] = X_test['mean_273_dow_0'] / X_test['mean_273_dow_2']
#X_test['mean_182_dow_genre_city_ratio'] = X_test['mean_182_dow_0'] / X_test['mean_182_dow_2']
#X_test['mean_91_dow_genre_city_ratio'] = X_test['mean_91_dow_0'] / X_test['mean_91_dow_2']

X_test = X_test.drop(['air_store_id','visit_date','visit_datetime','vis','day_of_week'], axis=1)

print ('Kfold 9m Prepairing Finished')

from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5

params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'max_bin': 256,
    'learning_rate': 0.003,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'rmse',
    'num_threads': 8   
}

MAX_ROUNDS = 30000

NFOLDS = 10
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=228)
X = train.as_matrix()

train_pred = np.zeros(len(label))
test_pred = np.zeros(len(test_id))

for train_index,val_index in kfold.split(X):
    print("Train Index:",train_index,",Val Index:",val_index)
    X_train,X_valid = X[train_index],X[val_index]
    y_train,y_valid = label[train_index],label[val_index]
    print (X_train.shape)
    print (X_valid.shape)
    print (y_train.shape)
    print (y_valid.shape)
    
    dtrain = lgb.Dataset(
        X_train, label=y_train)
    dval = lgb.Dataset(
        X_valid, label=y_valid, reference=dtrain)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100)
    train_pred[val_index] += bst.predict(
        X_valid, num_iteration=bst.best_iteration or MAX_ROUNDS)
    val_pred = train_pred[val_index]
    test_pred += bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
    
test_pred /= NFOLDS    

result = pd.DataFrame({"id": test_id, "visitors": test_pred})   
result['visitors'] = np.expm1(result['visitors'])
result.to_csv('lgb_kfold_9m.csv', index=False)
print('Kfold 9m Done')
