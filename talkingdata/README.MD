# Prepocessing
```
Use hive spark for dealing bigdata
1. transform click_time from utc to china time zone.
2. remove data of 6th and 11th (6th:500, 7th 8th 9th 10th : 60000000, 11th:500)
3. sort all the data by click_time
```

# Postprocessing
```
modify duplicated data's result to 0 except last row(public learboard + 0.0001, private learboard - 0.0001)
```

# Feature Engineering
```
[category] 
app
device
os
channel
hour

[full data count] 
ip_count
app_count
ip_app_count
ip_device_count
ip_os_count
ip_hour_count
ip_day_count
device_app_count
app_channel_count
ip_device_os_count
ip_device_hour_count
ip_device_day_count
ip_day_hour_count
ip_device_app_count
ip_device_os_app_count
ip_device_os_hour_count
ip_device_os_day_count
ip_os_day_hour_count
ip_app_day_hour_count
ip_device_os_app_hour_count
ip_device_os_app_day_count
ip_device_os_day_hour_count

[next/pre click time] 
ip_device_os_app_interval
ip_device_app_interval
ip_os_interval
ip_app_interval
ip_device_interval
ip_channel_interval
ip_device_os_app_channel_interval
ip_device_os_app_pre_interval
ip_os_device_interval
ip_device_os_app_channel_pre_interval

[full data unique count] 
ip_os_unique
ip_hour_unique
ip_app_channel_unique
ip_app_os_channel_unique
ip_device_os_channel_app_unique
app_channel_unique
ip_app_unique
ip_device_unique
ip_channel_unique
ip_day_hour_unique
ip_app_os_unique
ip_device_os_app_unique

[full data var] 
ip_app_channel_day_val
ip_app_os_hour_val
ip_day_channel_hour_val
channel_day_hour_val
device_day_hour_val
device_day_val

[full data mean] 
ip_app_channel_hour_mean

[AvgViewPerDistinct] 
app_AvgViewPerDistinct
channel_AvgViewPerDistinct
app_channel_AvgViewPerDistinct

[cumcount] 
fu_ip_app_device_os_channel_cumcount
fu_ip_device_os_app_cumcount
fu_ip_os_cumcount
pre_ip_app_device_os_channel_cumcount
pre_ip_device_os_app_cumcount
pre_ip_os_cumcount

[time lagged conversion rate with smoothing]
app_channel_day_confRate
app_os_day_confRate
app_day_confRate
app_device_day_confRate
channel_os_day_confRate
ip_day_confRate
channel_day_confRate
os_device_day_confRate
channel_device_day_confRate
os_day_confRate
device_day_confRate

[time lagged lastday count]
ip_device_os_lastday_count
ip_device_lastday_count
ip_os_lastday_count
ip_hour_day_count
ip_os_hour_lastday_count
ip_lastday_count
ip_device_os_app_lastday_count
ip_device_app_lastday_count
device_app_lastday_count
ip_app_lastday_count
app_lastday_count

[daily count]
ip_day_count
os_device_day_count
ip_device_os_day_count
ip_os_hour_day_count
ip_device_day_count
ip_device_os_app_hour_day_count
ip_app_day_count
ip_device_hour_day_count
os_app_channel_day_count
app_hour_day_count
ip_os_day_count
app_channel_day_count
ip_device_os_app_day_count
app_day_count
ip_device_os_app_channel_day_count
ip_app_hour_day_count

[daily unique count]
channel_day_app_unique
ip_app_os_day_channel_unique
ip_app_day_os_unique
ip_device_os_day_app_unique
ip_day_device_unique
ip_app_day_channel_unique
ip_day_hour_unique
ip_day_os_unique
ip_day_channel_unique
ip_day_app_unique
```

# MODEL
```
LGBM model1: use fulldata(train+test_supplement) features
LGBM model2: use daily+time-lagged(train 8th,9th without 7th ) features
```

## Ensembling
```
Final model1: kfold5 + kfold10 + 4 different seeds with 0.95 random split 
Final model2: kfold5 

Ensembling: 0.6*model1 + 0.4*model2
```

## parameter
```
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.2,
    'num_leaves': 255,  
    'max_depth': 8,  
    'min_child_samples': 100,  
    'max_bin': 512,  
    'subsample': 0.8,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.7,  
    'min_child_weight': 0,  
    'subsample_for_bin': 200000,  
    'min_split_gain': 0,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
    'verbose': 1,
    'scale_pos_weight':99.7
    }
```


## Model1 Features Importance
```
AvgViewPerDistinct: 1064434898.20
app: 288114538.85
channel: 230323927.88
ip_device_os_app_interval: 63081150.68
os: 39880287.50
ip_day_count: 35519653.87
hour: 21087197.50
app_count: 16035460.52
app_channel_count: 15725471.22
ip_app_unique: 15279595.57
ip_channel_unique: 13200176.97
ip_device_os_app_channel_interval: 13115941.11
ip_day_hour_unique: 9994154.98
ip_os_interval: 8489880.82
ip_device_os_count: 8408598.97
ip_device_unique: 7917279.61
ip_count: 7562795.10
device: 7072340.97
ip_app_interval: 6642276.86
ip_device_os_app_unique: 6238522.84
device_app_count: 5623707.73
ip_app_count: 5193923.89
ip_device_app_interval: 5109227.75
ip_device_os_hour_count: 4049344.42
app_channel_AvgViewPerDistinct: 4029501.96
ip_device_interval: 3951805.77
ip_os_count: 3858614.17
ip_hour_count: 3855379.73
ip_device_os_day_count: 3746720.46
ip_device_count: 3615113.84
ip_channel_interval: 3379538.61
ip_device_hour_count: 2986476.79
ip_day_hour_count: 2659574.17
ip_device_os_app_day_count: 2407219.98
ip_os_device_interval: 2374831.55
ip_device_os_day_hour_count: 2342606.83
ip_app_os_hour_val: 2335979.25
fu_ip_os_cumcount: 2326211.37
ip_os_day_hour_count: 2323880.99
ip_day_channel_hour_val: 2204448.97
ip_app_channel_hour_mean: 2189282.66
ip_device_day_count: 2156196.32
ip_device_app_count: 2152771.87
channel_AvgViewPerDistinct: 2053314.57
ip_device_os_app_pre_interval: 2011082.09
ip_device_os_app_channel_pre_interval: 1984448.38
ip_app_channel_day_val: 1932419.38
app_channel_unique: 1694746.68
ip_app_os_unique: 1690145.90
fu_ip_device_os_app_cumcount: 1571961.67
ip_device_os_app_count: 1549787.68
pre_ip_device_os_app_cumcount: 1520544.78
pre_ip_os_cumcount: 1457642.90
ip_app_day_hour_count: 1456536.06
ip_device_os_app_hour_count: 1348046.55
fu_ip_app_device_os_channel_cumcount: 1099012.88
channel_day_hour_val: 1063818.91
device_day_hour_val: 1008783.71
pre_ip_app_device_os_channel_cumcount: 925293.26
device_day_val: 524397.96
```


       
## Model2 Features Importance
```
app_channel_day_confRate: 307386690.01
app_os_day_confRate: 45150592.18
app_day_confRate: 18663941.82
ip_device_os_app_interval: 15872974.44
app: 11937979.15
channel: 8231264.10
ip_day_app_unique: 4758790.04
app_device_day_confRate: 4511819.78
hour: 4409729.92
os: 4155300.60
ip_day_count: 3797865.52
ip_device_os_app_channel_interval: 2690668.52
ip_day_channel_unique: 2564297.91
ip_day_os_unique: 2417356.68
channel_os_day_confRate: 1786649.87
ip_day_hour_unique: 1391647.51
ip_app_interval: 1332453.29
os_device_day_count: 1314436.66
ip_device_os_lastday_count: 1263692.71
ip_app_actived: 1239764.62
ip_app_day_channel_unique: 1207281.28
ip_channel_interval: 1020483.38
ip_day_confRate: 998582.11
ip_os_hour_day_count: 987099.66
ip_day_device_unique: 950227.65
ip_device_os_day_count: 737583.60
ip_device_os_app_pre_interval: 718463.26
fu_ip_os_day_cumcount: 714092.95
pre_ip_os_day_cumcount: 680397.72
ip_hour_lastday_count: 559473.56
ip_device_lastday_count: 555626.53
fu_ip_device_os_app_day_cumcount: 546697.67
ip_os_lastday_count: 519789.74
ip_device_app_interval: 514176.81
ip_hour_day_count: 511609.16
ip_os_interval: 491808.53
ip_device_interval: 431397.36
ip_device_day_count: 430193.94
ip_os_hour_lastday_count: 421602.30
ip_device_os_app_hour_day_count: 419821.60
ip_lastday_count: 415953.47
ip_app_day_count: 412231.47
fu_ip_app_day_cumcount: 411672.95
ip_device_os_app_channel_pre_interval: 366186.89
ip_device_hour_day_count: 354160.72
ip_device_os_app_lastday_count: 352607.67
os_app_channel_day_count: 338735.49
app_hour_day_count: 338046.80
pre_ip_device_os_app_day_cumcount: 299755.63
channel_day_confRate: 278338.34
os_device_day_confRate: 264116.35
fu_ip_device_os_app_channel_day_cumcount: 257177.07
ip_os_day_count: 231578.57
pre_ip_app_day_cumcount: 217029.46
app_channel_day_count: 212336.93
ip_device_app_lastday_count: 207487.04
ip_device_os_day_app_unique: 200250.29
ip_device_os_app_day_count: 199123.92
device_app_lastday_count: 196530.51
pre_ip_device_os_app_channel_day_cumcount: 185043.02
channel_device_day_confRate: 184098.77
ip_app_hour_day_count: 178982.31
ip_app_day_os_unique: 167040.88
ip_app_lastday_count: 152210.10
ip_os_device_interval: 137740.95
device_app_day_count: 136537.54
ip_device_os_channel_day_count: 125617.70
ip_device_app_day_count: 124575.44
app_day_count: 112572.01
app_channel_lastday_count: 109559.49
device: 104819.16
os_day_confRate: 99134.64
ip_device_os_app_channel_day_count: 94211.15
app_lastday_count: 92324.60
ip_app_os_day_channel_unique: 85043.77
device_day_confRate: 67359.84
channel_day_count: 58204.31
channel_day_app_unique: 23715.17
ip_device_os_channel_day_app_unique: 21900.99
```
