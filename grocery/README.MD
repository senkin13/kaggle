# Summary
predict 2017.8.16 - 2017.8.30 (16 days) unit_sales of each store's each item 

# Prepocessing
1. only use 2017 data
2. fill unit_sales's nan to 0,fill promotion's nan to false
3. transform train and test data to special format,one row is one item's info, column is date's unit_sales or onpromotion
  
# Feature engineering
## unit_sales's and promotion's statistics variable 
mean_7_2017: 5692166.96
mean_14_2017: 3324304.48
ahead7_mean_5: 1935866.46
ahead7_mean_6: 1582501.86
mean_7_2017_01: 614152.16
mean_14_2017_01: 584798.62
promo_0: 403271.37
ahead0_mean_5: 390026.87
mean_30_2017: 303733.60
per75_7_2017: 295517.47
median_8_dow0_2017: 294408.53
day_1_2017: 226402.73
ahead0_mean_2: 213642.94
median_16_dow0_2017: 145506.98
promo_7_2017: 115919.45
mean_8_dow0_2017: 107758.61
mean_30_2017_01: 104410.44
per25_7_2017: 100356.55
mean_20_dow0_2017: 81016.42
median_20_dow0_2017: 69484.45
bf_unpromo_0: 66632.18
median_14_2017: 60260.34
bf_unpromo_7: 55808.38
mean_60_2017: 51064.37
median_4_dow0_2017: 47903.91
median_30_2017: 38745.79
aft_promo_014: 37940.31
mean_16_dow0_2017: 30257.03
std_30_2017: 29830.08
mean_3_2017_32: 29162.48
std_14_2017: 28297.71
bf_unpromo_14: 27425.93
per25_3_2017: 26551.60
min_3_2017: 24524.46
aft_promo_0140: 21661.79
mean_4_dow0_2017: 21389.10
aft_promo_060: 18870.57
day_1_2017_01: 18253.50
ahead7_mean_3: 17934.06
var_14_2017: 17806.53
median_8_dow2_2017: 15997.74
max_3_2017: 13335.69
ahead0_mean_3: 11654.48
mean_3_2017_12: 11630.29
mean_8_dow2_2017: 11622.90
std_7_2017: 11293.82
unpromo_16aftsum_2017: 11054.41
max_7_2017: 10806.12
mean_3_2017: 10537.88
median_7_2017: 10297.30
mean_20_dow4_2017: 9875.45
mean_60_2017_32: 9145.11
mean_60_2017_01: 8966.44
day_1_2017_11: 8565.69
mean_60_2017_21: 8182.31
ahead0_mean_1: 8173.42
mean_30_2017_32: 7823.07
per75_3_2017: 7784.65
mean_140_2017: 7735.18
var_7_2017: 7139.18
var_30_2017: 6842.22
max_14_2017: 6634.32
mean_30_2017_21: 6331.10
promo_14: 6166.70
promo_60_2017: 5927.47
mean_3_2017_41: 5869.05
min_7_2017: 5735.97
promo_14_2017: 5687.56
median_20_dow4_2017: 5603.22
promo_7: 5570.81
mean_30_2017_22: 5404.41
mean_30_2017_12: 5240.22
mean_20_dow2_2017: 5171.16
mean_3_2017_51: 5138.52
mean_7_2017_51: 5134.67
median_60_2017: 4774.94
mean_3_2017_21: 4686.89
mean_16_dow4_2017: 4597.17
max_30_2017: 4530.39
mean_3_2017_31: 4518.99
aft_promo_914: 4510.16
mean_7_2017_11: 4440.63
median_20_dow2_2017: 4293.98
mean_7_2017_21: 4259.64
mean_30_2017_11: 4139.12
ahead7_mean_2: 4101.46
mean_3_2017_61: 4002.80
mean_7_2017_12: 3984.52
mean_7_2017_31: 3945.25
range_30_2017: 3850.11
range_7_2017: 3831.84
median_3_2017: 3712.32
mean_7_2017_22: 3678.40
mean_30_2017_02: 3676.92
day_1_2017_61: 3663.11
mean_7_2017_41: 3613.34
day_1_2017_12: 3592.20
range_14_2017: 3434.61
range_3_2017: 3392.02
ahead0_mean_6: 3382.50
aft_promo_2140: 3348.94
mean_4_dow6_2017: 3331.86
bf_unpromo_2: 3310.61
mean_3_2017_22: 3228.19
day_1_2017_51: 3212.38
mean_3_2017_11: 3083.99
day_1_2017_41: 3082.21
mean_20_dow3_2017: 3054.93
mean_3_2017_02: 3053.18
min_30_2017: 2920.27
mean_60_2017_22: 2865.35
promo_3: 2783.90
min_14_2017: 2765.38
day_1_2017_31: 2722.13
mean_7_2017_61: 2693.60
mean_16_dow6_2017: 2674.13
median_16_dow2_2017: 2630.62
median_16_dow6_2017: 2582.87
day_1_2017_52: 2578.22
day_1_2017_21: 2573.53
mean_8_dow4_2017: 2558.41
aft_promo_214: 2557.48
mean_140_2017_01: 2540.10
mean_4_dow5_2017: 2451.74
median_8_dow4_2017: 2432.07
mean_60_2017_31: 2394.61
mean_3_2017_62: 2351.98
mean_20_dow6_2017: 2322.14
aft_promo_6140: 2300.60
median_20_dow6_2017: 2280.69
day_1_2017_62: 2259.44
mean_14_2017_31: 2245.36
mean_14_2017_11: 2215.93
day_1_2017_02: 2210.49
median_4_dow5_2017: 2205.77
mean_14_2017_52: 2205.14
day_1_2017_32: 2178.14
aft_promo_960: 2144.12
mean_8_dow1_2017: 2129.29
mean_14_2017_02: 2111.43
median_8_dow3_2017: 2109.60
mean_16_dow2_2017: 2073.95
mean_20_dow1_2017: 2059.91
mean_3_2017_42: 2032.08
median_8_dow1_2017: 2028.29
median_16_dow4_2017: 2018.96
mean_7_2017_02: 1997.39
median_8_dow6_2017: 1993.75
aft_promo_14140: 1984.28
mean_7_2017_52: 1983.70
mean_20_dow5_2017: 1976.55
mean_14_2017_62: 1966.22
mean_60_2017_12: 1947.40
mean_3_2017_52: 1946.03
promo_140_2017: 1932.40
mean_16_dow3_2017: 1927.76
mean_7_2017_42: 1911.28
mean_4_dow2_2017: 1908.38
mean_8_dow6_2017: 1871.96
day_1_2017_42: 1870.34
aft_promo_160: 1868.66
aft_promo_1460: 1859.60
ahead7_mean_1: 1840.36
median_4_dow6_2017: 1825.59
mean_14_2017_12: 1823.19
mean_7_2017_62: 1822.80
mean_14_2017_22: 1819.55
aft_promo_1360: 1813.16
day_1_2017_22: 1807.07
mean_8_dow5_2017: 1799.44
aft_promo_1140: 1791.21
aft_promo_714: 1791.20
median_4_dow2_2017: 1791.08
mean_140_2017_32: 1790.30
mean_4_dow4_2017: 1786.72
median_8_dow5_2017: 1776.84
median_4_dow4_2017: 1767.57
mean_14_2017_21: 1744.52
mean_4_dow3_2017: 1740.73
mean_14_2017_42: 1738.74
median_20_dow1_2017: 1732.76
mean_16_dow5_2017: 1716.55
mean_16_dow1_2017: 1715.38
mean_14_2017_41: 1707.31
median_4_dow1_2017: 1681.89
aft_promo_260: 1623.23
median_16_dow1_2017: 1604.75
mean_7_2017_32: 1600.68
aft_promo_9140: 1593.16
mean_8_dow3_2017: 1588.16
mean_140_2017_12: 1581.76
aft_promo_114: 1565.32
median_4_dow3_2017: 1564.53
mean_4_dow1_2017: 1539.14
aft_promo_1414: 1535.20
median_20_dow3_2017: 1534.87
median_140_2017: 1489.66
median_16_dow5_2017: 1474.62
aft_promo_5140: 1473.39
median_20_dow5_2017: 1365.11
mean_14_2017_61: 1358.85
mean_140_2017_42: 1347.29
median_16_dow3_2017: 1342.59
ahead7_mean_4: 1251.03
mean_14_2017_32: 1168.70
aft_promo_13140: 1164.78
aft_promo_3140: 1114.97
mean_3_2017_01: 1075.45
mean_140_2017_11: 1074.92
mean_14_2017_51: 1070.90
aft_promo_4140: 1070.68
mean_60_2017_62: 1017.75
aft_promo_760: 1012.78
mean_140_2017_22: 1007.52
mean_30_2017_52: 984.39
aft_promo_7140: 962.74
bf_unpromo_9: 910.98
mean_30_2017_31: 906.77
aft_promo_614: 870.90
mean_30_2017_51: 852.13
aft_promo_1314: 845.88
aft_promo_360: 834.42
mean_30_2017_41: 743.42
aft_promo_414: 735.04
aft_promo_660: 724.69
aft_promo_1514: 724.59
promo_4: 704.72
aft_promo_1160: 703.79
aft_promo_15140: 700.73
aft_promo_560: 686.15
promo_5: 684.69
aft_promo_1260: 675.70
aft_promo_1560: 657.77
aft_promo_8140: 654.64
mean_30_2017_42: 652.42
mean_60_2017_02: 623.28
aft_promo_860: 617.19
mean_60_2017_11: 581.31
aft_promo_11140: 580.25
aft_promo_460: 571.83
bf_unpromo_4: 567.71
bf_unpromo_1: 545.24
aft_promo_12140: 523.58
mean_60_2017_61: 507.89
aft_promo_10140: 495.20
mean_30_2017_62: 491.41
bf_unpromo_3: 448.12
bf_unpromo_6: 446.80
mean_30_2017_61: 439.89
aft_promo_1114: 435.50
mean_60_2017_52: 425.56
promo_2: 418.68
mean_60_2017_42: 394.61
aft_promo_1060: 391.61
mean_140_2017_61: 356.25
mean_140_2017_41: 348.16
mean_140_2017_21: 344.07
mean_60_2017_51: 339.06
promo_1: 335.07
aft_promo_314: 318.38
mean_140_2017_31: 304.74
mean_60_2017_41: 290.18
mean_140_2017_52: 274.38
mean_140_2017_02: 274.17
mean_140_2017_51: 263.46
bf_unpromo_11: 259.64
bf_unpromo_5: 258.75
mean_140_2017_62: 243.02
aft_promo_814: 242.64
promo_9: 231.41
aft_promo_1214: 218.73
promo_6: 194.90
aft_promo_514: 184.57
bf_unpromo_10: 183.86
promo_15: 179.61
bf_unpromo_13: 172.14
promo_8: 155.05
aft_promo_1014: 140.72
bf_unpromo_8: 82.90
ahead0_mean_4: 77.95
bf_unpromo_12: 71.04
promo_13: 41.27
promo_12: 37.27
promo_11: 34.69
promo_10: 17.81
bf_unpromo_15: 0.00

# Model Selection
1. MA
2. LSTM
3. LightGBM 
4. TODO(Xgboost catboost nn)

# Model Evaluation
## Train
2017.6.14, 2017.6.21, 2017.6.28, 2017.7.5, 2017.7.12, 2017.7.19

## Validation
### public Validation
2017.7.26 - 2017.7.30 (5days)
### private Validation
2017.7.31 - 2017.8.10 (11days)

## Test
### public leaderboard
2017.8.16 - 2017.8.20 (5 days)
### private leaderboard
2017.8.21 - 2017.8.31 (11 days)

# Hyper parameter tuning
TODO

# Bagging
TODO

# Model Ensemble
## Now
### fist layer
lgb2*0.4 + lstm*0.3 + ma*0.3
### sencond layer
layer1*0.6 + lgb3*0.2 + lgb4*0.2 
### third layer
layer2*0.8 + 0.2*lgb5 
### fourth layer
layer3*0.5 + 0.2*lgb8 

## Optimized
TODO
