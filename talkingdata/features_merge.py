import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import pickle

X = pickle.load(open('/data/X.pkl','rb'))

#app_channel_lastday_count
app_channel_lastday_count = pickle.load(open('/data/app_channel_lastday_count.pkl','rb'))
app_channel_lastday_count = app_channel_lastday_count.reset_index(drop=True)
X['app_channel_lastday_count'] = app_channel_lastday_count
del app_channel_lastday_count
gc.collect()

#/data/app_lastday_count.pkl
app_lastday_count = pickle.load(open('/data/app_lastday_count.pkl','rb'))
app_lastday_count = app_lastday_count.reset_index(drop=True)
X['app_lastday_count'] = app_lastday_count
del app_lastday_count
gc.collect()

#/data/device_app_lastday_count.pkl
device_app_lastday_count = pickle.load(open('/data/device_app_lastday_count.pkl','rb'))
device_app_lastday_count = device_app_lastday_count.reset_index(drop=True)
X['device_app_lastday_count'] = device_app_lastday_count
del device_app_lastday_count
gc.collect()

#/data/ip_app_lastday_count.pkl
ip_app_lastday_count = pickle.load(open('/data/ip_app_lastday_count.pkl','rb'))
ip_app_lastday_count = ip_app_lastday_count.reset_index(drop=True)
X['ip_app_lastday_count'] = ip_app_lastday_count
del ip_app_lastday_count
gc.collect()


#/data/ip_device_app_lastday_count.pkl
ip_device_app_lastday_count = pickle.load(open('/data/ip_device_app_lastday_count.pkl','rb'))
ip_device_app_lastday_count = ip_device_app_lastday_count.reset_index(drop=True)
X['ip_device_app_lastday_count'] = ip_device_app_lastday_count
del ip_device_app_lastday_count
gc.collect()

#/data/ip_device_lastday_count.pkl
ip_device_lastday_count = pickle.load(open('/data/ip_device_lastday_count.pkl','rb'))
ip_device_lastday_count = ip_device_lastday_count.reset_index(drop=True)
X['ip_device_lastday_count'] = ip_device_lastday_count
del ip_device_lastday_count
gc.collect()

#/data/ip_device_os_app_lastday_count.pkl
ip_device_os_app_lastday_count = pickle.load(open('/data/ip_device_os_app_lastday_count.pkl','rb'))
ip_device_os_app_lastday_count = ip_device_os_app_lastday_count.reset_index(drop=True)
X['ip_device_os_app_lastday_count'] = ip_device_os_app_lastday_count
del ip_device_os_app_lastday_count
gc.collect()

#/data/ip_device_os_lastday_count.pkl
ip_device_os_lastday_count = pickle.load(open('/data/ip_device_os_lastday_count.pkl','rb'))
ip_device_os_lastday_count = ip_device_os_lastday_count.reset_index(drop=True)
X['ip_device_os_lastday_count'] = ip_device_os_lastday_count
del ip_device_os_lastday_count
gc.collect()

#/data/ip_hour_lastday_count.pkl
ip_hour_lastday_count = pickle.load(open('/data/ip_hour_lastday_count.pkl','rb'))
ip_hour_lastday_count = ip_hour_lastday_count.reset_index(drop=True)
X['ip_hour_lastday_count'] = ip_hour_lastday_count
del ip_hour_lastday_count
gc.collect()

#/data/ip_lastday_count.pkl
ip_lastday_count = pickle.load(open('/data/ip_lastday_count.pkl','rb'))
ip_lastday_count = ip_lastday_count.reset_index(drop=True)
X['ip_lastday_count'] = ip_lastday_count
del ip_lastday_count
gc.collect()

#/data/ip_os_hour_lastday_count.pkl
ip_os_hour_lastday_count = pickle.load(open('/data/ip_os_hour_lastday_count.pkl','rb'))
ip_os_hour_lastday_count = ip_os_hour_lastday_count.reset_index(drop=True)
X['ip_os_hour_lastday_count'] = ip_os_hour_lastday_count
del ip_os_hour_lastday_count
gc.collect()

#/data/ip_os_lastday_count.pkl
ip_os_lastday_count = pickle.load(open('/data/ip_os_lastday_count.pkl','rb'))
ip_os_lastday_count = ip_os_lastday_count.reset_index(drop=True)
X['ip_os_lastday_count'] = ip_os_lastday_count
del ip_os_lastday_count
gc.collect()

app_day_confRate = pickle.load(open('/data/app_day_confRate.pkl','rb'))
app_day_confRate = app_day_confRate.reset_index(drop=True)
X['app_day_confRate'] = app_day_confRate
del app_day_confRate
gc.collect()


os_day_confRate = pickle.load(open('/data/os_day_confRate.pkl','rb'))
os_day_confRate = os_day_confRate.reset_index(drop=True)
X['os_day_confRate'] = os_day_confRate
del os_day_confRate
gc.collect()


device_day_confRate = pickle.load(open('/data/device_day_confRate.pkl','rb'))
device_day_confRate = device_day_confRate.reset_index(drop=True)
X['device_day_confRate'] = device_day_confRate
del device_day_confRate
gc.collect()


channel_day_confRate = pickle.load(open('/data/channel_day_confRate.pkl','rb'))
channel_day_confRate = channel_day_confRate.reset_index(drop=True)
X['channel_day_confRate'] = channel_day_confRate
del channel_day_confRate
gc.collect()

ip_day_confRate = pickle.load(open('/data/ip_day_confRate.pkl','rb'))
ip_day_confRate = ip_day_confRate.reset_index(drop=True)
X['ip_day_confRate'] = ip_day_confRate
del ip_day_confRate
gc.collect()

X_actived = pickle.load(open('/data/actived.pkl','rb'))
X['ip_app_actived'] = X_actived['ip_app_actived'].reset_index(drop=True)


