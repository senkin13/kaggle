import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import pickle
import gensim

%%time

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        }
X = pd.read_csv('../input/full.csv',parse_dates=['click_time'],dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time','is_attributed'])
X['day'] = X['click_time'].dt.day
X['hour'] = X['click_time'].dt.hour

%%time

#two
X1 = X.copy()
gp = X1[['ip','day','os']].groupby(by=['ip','day'])[['os']].nunique().reset_index().rename(index=str, columns={'os': 'ip_day_os_unique'})
X1 = X1.merge(gp, on=['ip','day'], how='left')
X1['ip_day_os_unique'] = X1['ip_day_os_unique'].astype(np.float32)
X1['ip_day_os_unique'].to_pickle('/data/ip_day_os_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['ip','day','hour']].groupby(by=['ip','day'])[['hour']].nunique().reset_index().rename(index=str, columns={'hour': 'ip_day_hour_unique'})
X1 = X1.merge(gp, on=['ip','day'], how='left')
X1['ip_day_hour_unique'] = X1['ip_day_hour_unique'].astype(np.float32)
X1['ip_day_hour_unique'].to_pickle('/data/ip_day_hour_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['ip','day','app']].groupby(by=['ip','day'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'ip_day_app_unique'})
X1 = X1.merge(gp, on=['ip','day'], how='left')
X1['ip_day_app_unique'] = X1['ip_day_app_unique'].astype(np.float32)
X1['ip_day_app_unique'].to_pickle('/data/ip_day_app_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['ip','day','device']].groupby(by=['ip','day'])[['device']].nunique().reset_index().rename(index=str, columns={'device': 'ip_day_device_unique'})
X1 = X1.merge(gp, on=['ip','day'], how='left')
X1['ip_day_device_unique'] = X1['ip_day_device_unique'].astype(np.float32)
X1['ip_day_device_unique'].to_pickle('/data/ip_day_device_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['ip','day','channel']].groupby(by=['ip','day'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_day_channel_unique'})
X1 = X1.merge(gp, on=['ip','day'], how='left')
X1['ip_day_channel_unique'] = X1['ip_day_channel_unique'].astype(np.float32)
X1['ip_day_channel_unique'].to_pickle('/data/ip_day_channel_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['channel','day','app']].groupby(by=['channel','day'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'channel_day_app_unique'})
X1 = X1.merge(gp, on=['channel','day'], how='left')
X1['channel_day_app_unique'] = X1['channel_day_app_unique'].astype(np.float32)
X1['channel_day_app_unique'].to_pickle('/data/channel_day_app_unique.pkl')
del gp,X1
gc.collect()


#three
X1 = X.copy()
gp = X1[['ip','app','day','channel']].groupby(by=['ip','app','day'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_app_day_channel_unique'})
X1 = X1.merge(gp, on=['ip','app','day'], how='left')
X1['ip_app_day_channel_unique'] = X1['ip_app_day_channel_unique'].astype(np.float32)
X1['ip_app_day_channel_unique'].to_pickle('/data/ip_app_day_channel_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['ip','app','day','os']].groupby(by=['ip','app','day'])[['os']].nunique().reset_index().rename(index=str, columns={'os': 'ip_app_day_os_unique'})
X1 = X1.merge(gp, on=['ip','app','day'], how='left')
X1['ip_app_day_os_unique'] = X1['ip_app_day_os_unique'].astype(np.float32)
X1['ip_app_day_os_unique'].to_pickle('/data/ip_app_day_os_unique.pkl')
del gp,X1
gc.collect()



#four
X1 = X.copy()
gp = X1[['ip','app','os','day','channel']].groupby(by=['ip','app','os','day'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_app_os_day_channel_unique'})
X1 = X1.merge(gp, on=['ip','app','os','day'], how='left')
X1['ip_app_os_day_channel_unique'] = X1['ip_app_os_day_channel_unique'].astype(np.float32)
X1['ip_app_os_day_channel_unique'].to_pickle('/data/ip_app_os_day_channel_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['ip','device','os','day','app']].groupby(by=['ip','device','os','day'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'ip_device_os_day_app_unique'})
X1 = X1.merge(gp, on=['ip','device','os','day'], how='left')
X1['ip_device_os_day_app_unique'] = X1['ip_device_os_day_app_unique'].astype(np.float32)
X1['ip_device_os_day_app_unique'].to_pickle('/data/ip_device_os_day_app_unique.pkl')
del gp,X1
gc.collect()

#five

X1 = X.copy()
gp = X1[['ip','device','os','channel','day','app']].groupby(by=['ip','device','os','channel','day'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'ip_device_os_channel_day_app_unique'})
X1 = X1.merge(gp, on=['ip','device','os','channel','day'], how='left')
X1['ip_device_os_channel_day_app_unique'] = X1['ip_device_os_channel_day_app_unique'].astype(np.float32)
X1['ip_device_os_channel_day_app_unique'].to_pickle('/data/ip_device_os_channel_day_app_unique.pkl')
del gp,X1
gc.collect()


#######################Average clicks on app by distinct users###########################
X1 = X.copy()
gp = X1[['app','day','ip']].groupby(by=['app','day'])[['ip']].agg(lambda x: float(len(x)) / len(x.unique())).reset_index().rename(index=str, columns={'ip': 'app_day_AvgViewPerDistinct'})
X1 = X1.merge(gp, on=['app','day'], how='left')
X1['app_day_AvgViewPerDistinct'] = X1['app_day_AvgViewPerDistinct'].astype(np.float32)
X1['app_day_AvgViewPerDistinct'].to_pickle('/data/app_day_AvgViewPerDistinct.pkl')
del gp,X1
gc.collect()

#######################Average clicks on channel by distinct users###########################
X1 = X.copy()
gp = X1[['channel','day','ip']].groupby(by=['channel','day'])[['ip']].agg(lambda x: float(len(x)) / len(x.unique())).reset_index().rename(index=str, columns={'ip': 'channel_day_AvgViewPerDistinct'})
X1 = X1.merge(gp, on=['channel','day'], how='left')
X1['channel_day_AvgViewPerDistinct'] = X1['channel_day_AvgViewPerDistinct'].astype(np.float32)
X1['channel_day_AvgViewPerDistinct'].to_pickle('/data/channel_day_AvgViewPerDistinct.pkl')
del gp,X1
gc.collect()

#######################Average clicks on app/channel by distinct users###########################
X1 = X.copy()
gp = X1[['app','channel','day','ip']].groupby(by=['app','channel','day'])[['ip']].agg(lambda x: float(len(x)) / len(x.unique())).reset_index().rename(index=str, columns={'ip': 'app_channel_day_AvgViewPerDistinct'})
X1 = X1.merge(gp, on=['app','channel','day'], how='left')
X1['app_channel_day_AvgViewPerDistinct'] = X1['app_channel_day_AvgViewPerDistinct'].astype(np.float32)
X1['app_channel_day_AvgViewPerDistinct'].to_pickle('/data/app_channel_day_AvgViewPerDistinct.pkl')
del gp,X1
gc.collect()


##################################################next click time#######################################################
X['ip_os_pre_interval'] = X[['ip','os','click_time']].groupby(['ip','os'])[['click_time']].transform(lambda x: x.diff().shift(1).dt.seconds)
X['ip_os_pre_interval'].to_pickle('/data/ip_os_pre_interval.pkl')

###daily cumcount###
X1 = X.copy()
X1['pre_ip_app_day_cumcount'] = X[['ip','app','day']].groupby(['ip','app','day']).cumcount().rename('pre_ip_app_day_cumcount')
X1['pre_ip_app_day_cumcount'] = X1['pre_ip_app_day_cumcount'].astype(np.float32)
X1['pre_ip_app_day_cumcount'].to_pickle('/data/timelag/pre_ip_app_day_cumcount.pkl')

X1['fu_ip_app_day_cumcount'] = X[['ip','app','day']].iloc[::-1].groupby(['ip','app','day']).cumcount().rename('fu_ip_app_day_cumcount').iloc[::-1]
X1['fu_ip_app_day_cumcount'] = X1['fu_ip_app_day_cumcount'].astype(np.float32)
X1['fu_ip_app_day_cumcount'].to_pickle('/data/timelag/fu_ip_app_day_cumcount.pkl')

del X1
gc.collect()


X1 = X.copy()
X1['pre_ip_os_day_cumcount'] = X[['ip','os','day']].groupby(['ip','os','day']).cumcount().rename('pre_ip_os_day_cumcount')
X1['pre_ip_os_day_cumcount'] = X1['pre_ip_os_day_cumcount'].astype(np.float32)
X1['pre_ip_os_day_cumcount'].to_pickle('/data/timelag/pre_ip_os_day_cumcount.pkl')

X1['fu_ip_os_day_cumcount'] = X[['ip','os','day']].iloc[::-1].groupby(['ip','os','day']).cumcount().rename('fu_ip_os_day_cumcount').iloc[::-1]
X1['fu_ip_os_day_cumcount'] = X1['fu_ip_os_day_cumcount'].astype(np.float32)
X1['fu_ip_os_day_cumcount'].to_pickle('/data/timelag/fu_ip_os_day_cumcount.pkl')

del X1
gc.collect()

X1 = X.copy()
X1['pre_ip_device_os_app_day_cumcount'] = X[['ip','device','os','app','day']].groupby(['ip','device','os','app','day']).cumcount().rename('pre_ip_device_os_app_day_cumcount')
X1['pre_ip_device_os_app_day_cumcount'] = X1['pre_ip_device_os_app_day_cumcount'].astype(np.float32)
X1['pre_ip_device_os_app_day_cumcount'].to_pickle('/data/timelag/pre_ip_device_os_app_day_cumcount.pkl')

X1['fu_ip_device_os_app_day_cumcount'] = X[['ip','device','os','app','day']].iloc[::-1].groupby(['ip','device','os','app','day']).cumcount().rename('fu_ip_device_os_app_day_cumcount').iloc[::-1]
X1['fu_ip_device_os_app_day_cumcount'] = X1['fu_ip_device_os_app_day_cumcount'].astype(np.float32)
X1['fu_ip_device_os_app_day_cumcount'].to_pickle('/data/timelag/fu_ip_device_os_app_day_cumcount.pkl')

del X1
gc.collect()

X1 = X.copy()
X1['pre_ip_device_os_app_channel_day_cumcount'] = X[['ip','device','os','app','channel','day']].groupby(['ip','device','os','app','channel','day']).cumcount().rename('pre_ip_device_os_app_channel_day_cumcount')
X1['pre_ip_device_os_app_channel_day_cumcount'] = X1['pre_ip_device_os_app_channel_day_cumcount'].astype(np.float32)
X1['pre_ip_device_os_app_channel_day_cumcount'].to_pickle('/data/timelag/pre_ip_device_os_app_channel_day_cumcount.pkl')

X1['fu_ip_device_os_app_channel_day_cumcount'] = X[['ip','device','os','app','channel','day']].iloc[::-1].groupby(['ip','device','os','app','channel','day']).cumcount().rename('fu_ip_device_os_app_channel_day_cumcount').iloc[::-1]
X1['fu_ip_device_os_app_channel_day_cumcount'] = X1['fu_ip_device_os_app_channel_day_cumcount'].astype(np.float32)
X1['fu_ip_device_os_app_channel_day_cumcount'].to_pickle('/data/timelag/fu_ip_device_os_app_channel_day_cumcount.pkl')

del X1
gc.collect()

###addtional cumcount###
X1 = X.copy()
X1['pre_ip_app_cumcount'] = X[['ip','app']].groupby(['ip','app']).cumcount().rename('pre_ip_app_cumcount')
X1['pre_ip_app_cumcount'] = X1['pre_ip_app_cumcount'].astype(np.float32)
X1['pre_ip_app_cumcount'].to_pickle('/data/pre_ip_app_cumcount.pkl')

X1['fu_ip_app_cumcount'] = X[['ip','app']].iloc[::-1].groupby(['ip','app']).cumcount().rename('fu_ip_app_cumcount').iloc[::-1]
X1['fu_ip_app_cumcount'] = X1['fu_ip_app_cumcount'].astype(np.float32)
X1['fu_ip_app_cumcount'].to_pickle('/data/fu_ip_app_cumcount.pkl')

del X1
gc.collect()

###addtional unique###
X1 = X.copy()
gp = X1[['channel','app']].groupby(by=['channel'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'channel_app_unique'})
X1 = X1.merge(gp, on=['channel'], how='left')
X1['channel_app_unique'] = X1['channel_app_unique'].astype(np.float32)
X1['channel_app_unique'].to_pickle('/data/channel_app_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['device','app']].groupby(by=['device'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'device_app_unique'})
X1 = X1.merge(gp, on=['device'], how='left')
X1['device_app_unique'] = X1['device_app_unique'].astype(np.float32)
X1['device_app_unique'].to_pickle('/data/device_app_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['ip','device','os','channel']].groupby(by=['ip','device','os'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_device_os_channel_unique'})
X1 = X1.merge(gp, on=['ip','device','os'], how='left')
X1['ip_device_os_channel_unique'] = X1['ip_device_os_channel_unique'].astype(np.float32)
X1['ip_device_os_channel_unique'].to_pickle('/data/ip_device_os_channel_unique.pkl')
del gp,X1
gc.collect()

X1 = X.copy()
gp = X1[['ip','device','os','app','channel']].groupby(by=['ip','device','os','app'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_device_os_app_channel_unique'})
X1 = X1.merge(gp, on=['ip','device','os','app'], how='left')
X1['ip_device_os_app_channel_unique'] = X1['ip_device_os_app_channel_unique'].astype(np.float32)
X1['ip_device_os_app_channel_unique'].to_pickle('/data/ip_device_os_app_channel_unique.pkl')
del gp,X1
gc.collect()

ip_day_click_time_min = X.groupby(['ip','day'])['click_time'].min().reset_index().rename(columns={'click_time': 'ip_day_click_time_min'})
X = X.merge(ip_day_click_time_min, on=['ip','day'], how='left')
X['ip_day_since_click_time_start'] = (X['click_time'] - X['ip_day_click_time_min']).dt.seconds.astype(np.float32)
X['ip_day_since_click_time_start'].reset_index(drop=True).to_pickle('/data/ip_day_since_click_time_start.pkl')
del ip_day_click_time_min
gc.collect()

ip_day_click_time_max = X.groupby(['ip','day'])['click_time'].max().reset_index().rename(columns={'click_time': 'ip_day_click_time_max'})
X = X.merge(ip_day_click_time_max, on=['ip','day'], how='left')
X['ip_day_util_click_time_end'] = (X['ip_day_click_time_max'] - X['click_time']).dt.seconds.astype(np.float32)
X['ip_day_util_click_time_end'].reset_index(drop=True).to_pickle('/data/ip_day_util_click_time_end.pkl')
del ip_day_click_time_max
gc.collect()

ip_day_hour_click_time_min = X.groupby(['ip','day','hour'])['click_time'].min().reset_index().rename(columns={'click_time': 'ip_day_hour_click_time_min'})
X = X.merge(ip_day_hour_click_time_min, on=['ip','day','hour'], how='left')
X['ip_day_hour_since_click_time_start'] = (X['click_time'] - X['ip_day_hour_click_time_min']).dt.seconds.astype(np.float32)
X['ip_day_hour_since_click_time_start'].reset_index(drop=True).to_pickle('/data/ip_day_hour_since_click_time_start.pkl')
del ip_day_hour_click_time_min
gc.collect()

ip_day_hour_click_time_max = X.groupby(['ip','day','hour'])['click_time'].max().reset_index().rename(columns={'click_time': 'ip_day_hour_click_time_max'})
X = X.merge(ip_day_hour_click_time_max, on=['ip','day','hour'], how='left')
X['ip_day_hour_util_click_time_end'] = (X['ip_day_hour_click_time_max'] - X['click_time']).dt.seconds.astype(np.float32)
X['ip_day_hour_util_click_time_end'].reset_index(drop=True).to_pickle('/data/ip_day_hour_util_click_time_end.pkl')
del ip_day_hour_click_time_max
gc.collect()

del X

ip_click_time_min = X.groupby(['ip'])['click_time'].min().reset_index().rename(columns={'click_time': 'ip_click_time_min'})
X = X.merge(ip_click_time_min, on=['ip'], how='left')
X['ip_since_click_time_start'] = (X['click_time'] - X['ip_click_time_min']).dt.seconds.astype(np.float32)
X['ip_since_click_time_start'].reset_index(drop=True).to_pickle('/data/ip_since_click_time_start.pkl')
del ip_click_time_min
gc.collect()

ip_click_time_max = X.groupby(['ip'])['click_time'].max().reset_index().rename(columns={'click_time': 'ip_click_time_max'})
X = X.merge(ip_click_time_max, on=['ip'], how='left')
X['ip_util_click_time_end'] = (X['ip_click_time_max'] - X['click_time']).dt.seconds.astype(np.float32)
X['ip_util_click_time_end'].reset_index(drop=True).to_pickle('/data/ip_util_click_time_end.pkl')
del ip_click_time_max
gc.collect()

ip_device_click_time_min = X.groupby(['ip','device'])['click_time'].min().reset_index().rename(columns={'click_time': 'ip_device_click_time_min'})
X = X.merge(ip_device_click_time_min, on=['ip','device'], how='left')
X['ip_device_since_click_time_start'] = (X['click_time'] - X['ip_device_click_time_min']).dt.seconds.astype(np.float32)
X['ip_device_since_click_time_start'].reset_index(drop=True).to_pickle('/data/ip_device_since_click_time_start.pkl')
del ip_device_click_time_min
gc.collect()

ip_device_click_time_max = X.groupby(['ip','device'])['click_time'].max().reset_index().rename(columns={'click_time': 'ip_device_click_time_max'})
X = X.merge(ip_device_click_time_max, on=['ip','device'], how='left')
X['ip_device_util_click_time_end'] = (X['ip_device_click_time_max'] - X['click_time']).dt.seconds.astype(np.float32)
X['ip_device_util_click_time_end'].reset_index(drop=True).to_pickle('/data/ip_device_util_click_time_end.pkl')
del ip_device_click_time_max
gc.collect()





#======ip_app_actived=======
%%time
index_start6 = 0
index_end6 = len(X[X['day']==6]) - 1

index_start7 = index_end6 + 1
index_end7 = index_end6 + len(X[X['day']==7])

index_start8 = index_end7 + 1
index_end8 = index_end7 + len(X[X['day']==8])

index_start9 = index_end8 + 1
index_end9 = index_end8 + len(X[X['day']==9])

index_start10 = index_end9 + 1
index_end10 = index_end9 + len(X[X['day']==10])

index_start11 = index_end10 + 1
index_end11 = index_end10 + len(X[X['day']==11])

X6 = X.loc[index_start6:index_end6]
X7 = X.loc[index_start7:index_end7]
X8 = X.loc[index_start8:index_end8]
X9 = X.loc[index_start9:index_end9]
X10 = X.loc[index_start10:index_end10]
X11 = X.loc[index_start11:index_end11]

X6['ip_app_actived'] = np.nan
X7['ip_app_actived'] = np.nan
X11['ip_app_actived'] = np.nan

X7_ip_app_actived = pd.DataFrame()
X7_ip_app_actived = X7[X7['is_attributed']==1][['ip','app','is_attributed']].drop_duplicates(subset=['ip','app']).rename(columns={'is_attributed': 'ip_app_actived'})
X8 = X8.merge(X7_ip_app_actived, on=['ip','app'], how='left')
del X7_ip_app_actived
gc.collect()

X8_ip_app_actived = pd.DataFrame()
X8_ip_app_actived = X8[X8['is_attributed']==1][['ip','app','is_attributed']].drop_duplicates(subset=['ip','app']).rename(columns={'is_attributed': 'ip_app_actived'})
X9 = X9.merge(X8_ip_app_actived, on=['ip','app'], how='left')
del X8_ip_app_actived
gc.collect()

X9_ip_app_actived = pd.DataFrame()
X9_ip_app_actived = X9[X9['is_attributed']==1][['ip','app','is_attributed']].drop_duplicates(subset=['ip','app']).rename(columns={'is_attributed': 'ip_app_actived'})
X10 = X10.merge(X9_ip_app_actived, on=['ip','app'], how='left')
del X9_ip_app_actived
gc.collect()

#=======concat===========
X2 = pd.concat([X6,X7,X8,X9,X10,X11])
X2['ip_actived','ip_device_actived','ip_app_actived'].to_pickle('/data/actived.pkl')


X7_ip = pd.DataFrame()
X7_ip['ip'] = X7.ip.drop_duplicates()
X7_ip['clicked_ip'] = 1
X8 = pd.merge(X8,X7_ip, on=['ip'], how='left')

X8_ip = pd.DataFrame()
X8_ip['ip'] = X8.ip.drop_duplicates()
X8_ip['clicked_ip'] = 1
X9 = pd.merge(X9,X8_ip, on=['ip'], how='left')

X9_ip = pd.DataFrame()
X9_ip['ip'] = X9.ip.drop_duplicates()
X9_ip['clicked_ip'] = 1
X10 = pd.merge(X10,X9_ip, on=['ip'], how='left')

X2 = pd.concat([X6,X7,X8,X9,X10,X11])
X2['clicked_ip'].to_pickle('/data/clicked_ip.pkl')


##################################################var#######################################################
gp = X[['channel','day','hour']].groupby(by=['channel','day'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'channel_day_hour_val'})
gp.to_pickle('/data/channel_day_hour_val.pkl')
del gp
gc.collect()

gp = X[['device','day','hour']].groupby(by=['device','day'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'device_day_hour_val'})
gp.to_pickle('/data/device_day_hour_val.pkl')
del gp
gc.collect()

gp = X[['device','day']].groupby(by=['device'])[['day']].var().reset_index().rename(index=str, columns={'day': 'device_day_val'})
gp.to_pickle('/data/device_day_val.pkl')
del gp
gc.collect()

gp = X[['ip','app','os','hour']].groupby(by=['ip','app','os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_hour_val'})
gp.to_pickle('../features/train/ip_app_os_hour_val.pkl')
del gp
gc.collect()

gp = X[['ip','app','channel','day']].groupby(by=['ip','app','channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_day_val'})
gp.to_pickle('../features/train/ip_app_channel_day_val.pkl')
del gp
gc.collect()

gp = X[['ip','day','channel','hour']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_day_channel_hour_val'})
gp.to_pickle('../features/train/ip_day_channel_hour_val.pkl')
del gp
gc.collect()

##################################################mean#######################################################
gp = X[['ip','app','channel','hour']].groupby(by=['ip','app','channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_hour_mean'})
gp.to_pickle('../features/train/ip_app_channel_hour_mean.pkl')
del gp
gc.collect()


##################################################cumcount#######################################################

X['pre_ip_app_device_os_channel_cumcount'] = X[['ip','app','device','os','channel']].groupby(['ip','app','device','os','channel']).cumcount().rename('pre_ip_app_device_os_channel_cumcount')
X['pre_ip_app_device_os_channel_cumcount'].to_pickle('../features/train/pre_ip_app_device_os_channel_cumcount09.pkl')
X['fu_ip_app_device_os_channel_cumcount'] = X[['ip','app','device','os','channel']].iloc[::-1].groupby(['ip','app','device','os','channel']).cumcount().rename('fu_ip_app_device_os_channel_cumcount').iloc[::-1]
X['fu_ip_app_device_os_channel_cumcount'].to_pickle('../features/train/fu_ip_app_device_os_channel_cumcount09.pkl')


X['pre_ip_os_cumcount'] = X[['ip','os']].groupby(['ip','os']).cumcount().rename('pre_ip_os_cumcount')
X['pre_ip_os_cumcount'].to_pickle('../features/train/pre_ip_os_cumcount.pkl')
X['fu_ip_os_cumcount'] = X[['ip','os']].iloc[::-1].groupby(['ip','os']).cumcount().rename('fu_ip_os_cumcount').iloc[::-1]
X['fu_ip_os_cumcount'].to_pickle('../features/train/fu_ip_os_cumcount.pkl')

X['pre_ip_device_os_app_cumcount'] = X[['ip','device','os','app']].groupby(['ip','device','os','app']).cumcount().rename('fu_ip_device_os_app_cumcount')
X['pre_ip_device_os_app_cumcount'].to_pickle('../features/train/pre_ip_device_os_app_cumcount.pkl')
X['fu_ip_device_os_app_cumcount'] = X[['ip','device','os','app']].iloc[::-1].groupby(['ip','device','os','app']).cumcount().rename('fu_ip_device_os_app_cumcount').iloc[::-1]
X['fu_ip_device_os_app_cumcount'].to_pickle('../features/train/fu_ip_device_os_app_cumcount.pkl')


##################################################next click time#######################################################
X['ip_app_interval'] = X[['ip','app','click_time']].groupby(['ip','app'])[['click_time']].transform(lambda x: x.diff().shift(-1).dt.seconds)
X['ip_app_interval'].to_pickle('../features/train/ip_app_interval.pkl')


##################################################next click time#######################################################
X['ip_os_device_interval'] = X[['ip','os','device','click_time']].groupby(['ip','os','device'])[['click_time']].transform(lambda x: x.diff().shift(-1).dt.seconds)
X['ip_os_device_interval'].to_pickle('../features/train/ip_os_device_interval.pkl')

X['ip_os_device_pre_interval'] = X[['ip','os','device','click_time']].groupby(['ip','os','device'])[['click_time']].transform(lambda x: x.diff().shift(1).dt.seconds)
X['ip_os_device_pre_interval'].to_pickle('../features/train/ip_os_device_pre_interval.pkl')


##################################################unique counts#######################################################

##################one group##################
gp = X[['ip','app']].groupby(by=['ip'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'ip_app_unique'})
gp.to_pickle('../features/train/ip_app_unique.pkl')
del gp
gc.collect()
gp = X[['ip','channel']].groupby(by=['ip'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_channel_unique'})
gp.to_pickle('../features/train/ip_channel_unique.pkl')
del gp
gc.collect()
gp = X[['ip','device']].groupby(by=['ip'])[['device']].nunique().reset_index().rename(index=str, columns={'device': 'ip_device_unique'})
gp.to_pickle('../features/train/ip_device_unique.pkl')
del gp
gc.collect()
gp = X[['app','channel']].groupby(by=['app'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'app_channel_unique'})
gp.to_pickle('../features/train/app_channel_unique.pkl')
del gp
gc.collect()

##################two group##################
gp = X[['ip','day','hour']].groupby(by=['ip','day'])[['hour']].nunique().reset_index().rename(index=str, columns={'hour': 'ip_day_hour_unique'})
gp.to_pickle('../features/train/ip_day_hour_unique.pkl')
del gp
gc.collect()
gp = X[['ip','app','os']].groupby(by=['ip','app'])[['os']].nunique().reset_index().rename(index=str, columns={'os': 'ip_app_os_unique'})
gp.to_pickle('../features/train/ip_app_os_unique.pkl')
del gp
gc.collect()
##################three group##################
gp = X[['ip','device','os','app']].groupby(by=['ip','device','os'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'ip_device_os_app_unique'})
gp.to_pickle('../features/train/ip_device_os_app_unique.pkl')
del gp
gc.collect()



#######################group by click count###########################
##################one group##################
########ip_count########
gp = X[['ip','channel']].groupby(by=['ip'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_count'})
X = X.merge(gp, on=['ip'], how='left')
del gp
gc.collect()

##################two group##################
########ip_app_count########
gp = X[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
X = X.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()
########ip_device_count########
gp = X[['ip','device', 'channel']].groupby(by=['ip', 'device'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_count'})
X = X.merge(gp, on=['ip','device'], how='left')
del gp
gc.collect()
########ip_os_count########
gp = X[['ip','os', 'channel']].groupby(by=['ip', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_os_count'})
X = X.merge(gp, on=['ip','os'], how='left')
del gp
gc.collect()
########ip_hour_count########
gp = X[['ip','hour', 'channel']].groupby(by=['ip', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_hour_count'})
X = X.merge(gp, on=['ip','hour'], how='left')
del gp
gc.collect()
########ip_day_count########
gp = X[['ip','day', 'channel']].groupby(by=['ip', 'day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_count'})
X = X.merge(gp, on=['ip','day'], how='left')
del gp
gc.collect()
########device_app_count########
gp = X[['device','app', 'channel']].groupby(by=['device', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'device_app_count'})
X = X.merge(gp, on=['device','app'], how='left')
del gp
gc.collect()
########app_channel_count########
gp = X[['app', 'channel','device']].groupby(by=['app', 'channel'])[['device']].count().reset_index().rename(index=str, columns={'device': 'app_channel_count'})
X = X.merge(gp, on=['app','channel'], how='left')
del gp
gc.collect()

##################three group##################
########ip_device_os_count########
gp = X[['ip','device', 'os', 'channel']].groupby(by=['ip', 'device', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_count'})
X = X.merge(gp, on=['ip','device', 'os'], how='left')
del gp
gc.collect()

########ip_device_hour_count########
gp = X[['ip','device', 'hour', 'channel']].groupby(by=['ip', 'device', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_hour_count'})
X = X.merge(gp, on=['ip','device', 'hour'], how='left')
del gp
gc.collect()
########ip_device_day_count########
gp = X[['ip','device', 'day', 'channel']].groupby(by=['ip', 'device', 'day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_day_count'})
X = X.merge(gp, on=['ip','device', 'day'], how='left')
del gp
gc.collect()
########ip_day_hour_count########
gp = X[['ip','day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_hour_count'})
X = X.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()
########ip_device_app_count########
gp = X[['ip','device', 'app', 'channel']].groupby(by=['ip', 'device', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_app_count'})
X = X.merge(gp, on=['ip','device', 'app'], how='left')
del gp
gc.collect()

##################four group##################
########ip_device_os_app_count########
gp = X[['ip','device', 'os', 'app', 'channel']].groupby(by=['ip', 'device', 'os', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_app_count'})
X = X.merge(gp, on=['ip','device', 'os', 'app'], how='left')
del gp
gc.collect()
########ip_device_os_hour_count########
gp = X[['ip','device', 'os', 'hour', 'channel']].groupby(by=['ip', 'device', 'os', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_hour_count'})
X = X.merge(gp, on=['ip','device', 'os', 'hour'], how='left')
del gp
gc.collect()
########ip_device_os_day_count########
gp = X[['ip','device', 'os', 'day', 'channel']].groupby(by=['ip', 'device', 'os', 'day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_day_count'})
X = X.merge(gp, on=['ip','device', 'os', 'day'], how='left')
del gp
gc.collect()
########ip_os_day_hour_count########
gp = X[['ip', 'os', 'day', 'hour','channel']].groupby(by=['ip','os','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_os_day_hour_count'})
X = X.merge(gp, on=['ip', 'os', 'day','hour'], how='left')
del gp
gc.collect()
########ip_app_day_hour_count########
gp = X[['ip', 'app', 'day', 'hour','channel']].groupby(by=['ip','app','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_day_hour_count'})
X = X.merge(gp, on=['ip', 'app', 'day','hour'], how='left')
del gp
gc.collect()

##################five group##################
########ip_device_os_app_hour_count########
gp = X[['ip','device', 'os', 'app', 'hour', 'channel']].groupby(by=['ip', 'device', 'os', 'app', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_app_hour_count'})
X = X.merge(gp, on=['ip','device', 'os', 'app', 'hour'], how='left')
del gp
gc.collect()
########ip_device_os_app_day_count########
gp = X[['ip','device', 'os', 'app', 'day', 'channel']].groupby(by=['ip', 'device', 'os', 'app', 'day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_app_day_count'})
X = X.merge(gp, on=['ip','device', 'os', 'app', 'day'], how='left')
del gp
gc.collect()
########ip_device_os_day_hour_count########
gp = X[['ip','device', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'device', 'os', 'day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_day_hour_count'})
X = X.merge(gp, on=['ip','device', 'os', 'day', 'hour'], how='left')
del gp
gc.collect()

##################################################Final Features#######################################################
X.to_pickle('../features/train/count.pkl')



##timelag
%%time

####daily count####

##################one group##################

########ip_day_count########
X1 = X.copy()
gp = X1[['ip','day', 'channel']].groupby(by=['ip','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_count'})
X1 = X1.merge(gp, on=['ip','day'], how='left')
X1['ip_day_count'] = X1['ip_day_count'].astype(np.float32)
X1['ip_day_count'].to_pickle('/data/timelag/ip_day_count.pkl')
del gp,X1
gc.collect()
########app_day_count########
X1 = X.copy()
gp = X1[['app','day', 'channel']].groupby(by=['app','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'app_day_count'})
X1 = X1.merge(gp, on=['app','day'], how='left')
X1['app_day_count'] = X1['app_day_count'].astype(np.float32)
X1['app_day_count'].to_pickle('/data/timelag/app_day_count.pkl')
del gp,X1
gc.collect()
########channel_day_count########
X1 = X.copy()
gp = X1[['channel','day', 'app']].groupby(by=['channel','day'])[['app']].count().reset_index().rename(index=str, columns={'app': 'channel_day_count'})
X1 = X1.merge(gp, on=['channel','day'], how='left')
X1['channel_day_count'] = X1['channel_day_count'].astype(np.float32)
X1['channel_day_count'].to_pickle('/data/timelag/channel_day_count.pkl')
del gp,X1
gc.collect()


##################two group##################
########ip_app_day_count########
X1 = X.copy()
gp = X1[['ip','app','day', 'channel']].groupby(by=['ip','app','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_day_count'})
X1 = X1.merge(gp, on=['ip','app','day'], how='left')
X1['ip_app_day_count'] = X1['ip_app_day_count'].astype(np.float32)
X1['ip_app_day_count'].to_pickle('/data/timelag/ip_app_day_count.pkl')
del gp,X1
gc.collect()
########ip_device_day_count########
X1 = X.copy()
gp = X1[['ip','device','day', 'channel']].groupby(by=['ip','device','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_day_count'})
X1 = X1.merge(gp, on=['ip','device','day'], how='left')
X1['ip_device_day_count'] = X1['ip_device_day_count'].astype(np.float32)
X1['ip_device_day_count'].to_pickle('/data/timelag/ip_device_day_count.pkl')
del gp,X1
gc.collect()
########ip_os_day_count########
X1 = X.copy()
gp = X1[['ip','os','day', 'channel']].groupby(by=['ip','os','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_os_day_count'})
X1 = X1.merge(gp, on=['ip','os','day'], how='left')
X1['ip_os_day_count'] = X1['ip_os_day_count'].astype(np.float32)
X1['ip_os_day_count'].to_pickle('/data/timelag/ip_os_day_count.pkl')
del gp,X1
gc.collect()
########ip_hour_day_count########
X1 = X.copy()
gp = X1[['ip','hour','day', 'channel']].groupby(by=['ip','hour','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_hour_day_count'})
X1 = X1.merge(gp, on=['ip','hour','day'], how='left')
X1['ip_hour_day_count'] = X1['ip_hour_day_count'].astype(np.float32)
X1['ip_hour_day_count'].to_pickle('/data/timelag/ip_hour_day_count.pkl')
del gp,X1
gc.collect()
########device_app_day_count########
X1 = X.copy()
gp = X1[['device','app','day', 'channel']].groupby(by=['device','app','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'device_app_day_count'})
X1 = X1.merge(gp, on=['device','app','day'], how='left')
X1['device_app_day_count'] = X1['device_app_day_count'].astype(np.float32)
X1['device_app_day_count'].to_pickle('/data/timelag/device_app_day_count.pkl')
del gp,X1
gc.collect()
########app_channel_day_count########
X1 = X.copy()
gp = X1[['app','channel','day', 'device']].groupby(by=['app', 'channel','day'])[['device']].count().reset_index().rename(index=str, columns={'device': 'app_channel_day_count'})
X1 = X1.merge(gp, on=['app','channel','day'], how='left')
X1['app_channel_day_count'] = X1['app_channel_day_count'].astype(np.float32)
X1['app_channel_day_count'].to_pickle('/data/timelag/app_channel_day_count.pkl')
del gp,X1
gc.collect()
########app_hour_day_count########
X1 = X.copy()
gp = X1[['app','hour','day', 'device']].groupby(by=['app', 'hour','day'])[['device']].count().reset_index().rename(index=str, columns={'device': 'app_hour_day_count'})
X1 = X1.merge(gp, on=['app','hour','day'], how='left')
X1['app_hour_day_count'] = X1['app_hour_day_count'].astype(np.float32)
X1['app_hour_day_count'].to_pickle('/data/timelag/app_hour_day_count.pkl')
del gp,X1
gc.collect()
########os_device_day_count########
X1 = X.copy()
gp = X1[['os','device','day', 'channel']].groupby(by=['os','device','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'os_device_day_count'})
X1 = X1.merge(gp, on=['os','device','day'], how='left')
X1['os_device_day_count'] = X1['os_device_day_count'].astype(np.float32)
X1['os_device_day_count'].to_pickle('/data/timelag/os_device_day_count.pkl')
del gp,X1
gc.collect()


##################three group##################
########ip_device_os_day_count########
X1 = X.copy()
gp = X1[['ip','device','os','day', 'channel']].groupby(by=['ip','device','os','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_day_count'})
X1 = X1.merge(gp, on=['ip','device','os','day'], how='left')
X1['ip_device_os_day_count'] = X1['ip_device_os_day_count'].astype(np.float32)
X1['ip_device_os_day_count'].to_pickle('/data/timelag/ip_device_os_day_count.pkl')
del gp,X1
gc.collect()
########ip_device_hour_day_count########
X1 = X.copy()
gp = X1[['ip','device','hour','day','channel']].groupby(by=['ip','device', 'hour','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_hour_day_count'})
X1 = X1.merge(gp, on=['ip','device','hour','day'], how='left')
X1['ip_device_hour_day_count'] = X1['ip_device_hour_day_count'].astype(np.float32)
X1['ip_device_hour_day_count'].to_pickle('/data/timelag/ip_device_hour_day_count.pkl')
del gp,X1
gc.collect()
########ip_os_hour_day_count########
X1 = X.copy()
gp = X1[['ip','os','hour','day','channel']].groupby(by=['ip','os', 'hour','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_os_hour_day_count'})
X1 = X1.merge(gp, on=['ip','os','hour','day'], how='left')
X1['ip_os_hour_day_count'] = X1['ip_os_hour_day_count'].astype(np.float32)
X1['ip_os_hour_day_count'].to_pickle('/data/timelag/ip_os_hour_day_count.pkl')
del gp,X1
gc.collect()
########ip_app_hour_day_count########
X1 = X.copy()
gp = X1[['ip','app','hour','day','channel']].groupby(by=['ip','app', 'hour','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_hour_day_count'})
X1 = X1.merge(gp, on=['ip','app','hour','day'], how='left')
X1['ip_app_hour_day_count'] = X1['ip_app_hour_day_count'].astype(np.float32)
X1['ip_app_hour_day_count'].to_pickle('/data/timelag/ip_app_hour_day_count.pkl')
del gp,X1
gc.collect()
########ip_device_app_day_count########
X1 = X.copy()
gp = X1[['ip','device','app','day','channel']].groupby(by=['ip', 'device', 'app','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_app_day_count'})
X1 = X1.merge(gp, on=['ip','device','app','day'], how='left')
X1['ip_device_app_day_count'] = X1['ip_device_app_day_count'].astype(np.float32)
X1['ip_device_app_day_count'].to_pickle('/data/timelag/ip_device_app_day_count.pkl')
del gp,X1
gc.collect()

########os_app_channel_day_count########
X1 = X.copy()
gp = X1[['os', 'app', 'channel','day','device']].groupby(by=['os', 'app','channel','day'])[['device']].count().reset_index().rename(index=str, columns={'device': 'os_app_channel_day_count'})
X1 = X1.merge(gp, on=['os', 'app', 'channel','day'], how='left')
X1['os_app_channel_day_count'] = X1['os_app_channel_day_count'].astype(np.float32)
X1['os_app_channel_day_count'].to_pickle('/data/timelag/os_app_channel_day_count.pkl')
del gp,X1
gc.collect()

##################four group##################

########ip_device_os_app_day_count########
X1 = X.copy()
gp = X1[['ip','device', 'os', 'app','day','channel']].groupby(by=['ip', 'device', 'os', 'app','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_app_day_count'})
X1 = X1.merge(gp, on=['ip','device', 'os','app','day'], how='left')
X1['ip_device_os_app_day_count'] = X1['ip_device_os_app_day_count'].astype(np.float32)
X1['ip_device_os_app_day_count'].to_pickle('/data/timelag/ip_device_os_app_day_count.pkl')
del gp,X1
gc.collect()      
########ip_device_os_channel_day_count########
X1 = X.copy()
gp = X1[['ip','device', 'os', 'channel','day','app']].groupby(by=['ip', 'device', 'os', 'channel','day'])[['app']].count().reset_index().rename(index=str, columns={'app': 'ip_device_os_channel_day_count'})
X1 = X1.merge(gp, on=['ip','device', 'os','channel','day'], how='left')
X1['ip_device_os_channel_day_count'] = X1['ip_device_os_channel_day_count'].astype(np.float32)
X1['ip_device_os_channel_day_count'].to_pickle('/data/timelag/ip_device_os_channel_day_count.pkl')
del gp,X1
gc.collect()     


##################five group##################
########ip_device_os_app_hour_day_count########
X1 = X.copy()         
gp = X1[['ip','device', 'os', 'app', 'hour','day','channel']].groupby(by=['ip', 'device', 'os', 'app', 'hour','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_app_hour_day_count'})
X1 = X1.merge(gp, on=['ip','device', 'os', 'app', 'hour','day'], how='left')
X1['ip_device_os_app_hour_day_count'] = X1['ip_device_os_app_hour_day_count'].astype(np.float32)
X1['ip_device_os_app_hour_day_count'].to_pickle('/data/timelag/ip_device_os_app_hour_day_count.pkl')
del gp,X1
gc.collect()


########ip_device_os_app_channel_day_count########
X1 = X.copy()         
gp = X1[['ip','device', 'os', 'app', 'channel','day','hour']].groupby(by=['ip', 'device', 'os', 'app', 'channel','day'])[['hour']].count().reset_index().rename(index=str, columns={'hour': 'ip_device_os_app_channel_day_count'})
X1 = X1.merge(gp, on=['ip','device', 'os', 'app', 'channel','day'], how='left')
X1['ip_device_os_app_channel_day_count'] = X1['ip_device_os_app_channel_day_count'].astype(np.float32)
X1['ip_device_os_app_channel_day_count'].to_pickle('/data/timelag/ip_device_os_app_channel_day_count.pkl')
del gp,X1
gc.collect()

