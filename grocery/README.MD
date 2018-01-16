# Summary
predict 2017.8.16 - 2017.8.30 (16 days) unit_sales of each store's each item 

# Prepocessing
1. only use 2017 data
2. fill unit_sales's nan to 0,fill promotion's nan to false
3. transform train and test data to special format,one row is one item's info, column is date's unit_sales or onpromotion
4. add item,store metadata catagorial features, using one hot encoder
5. use future 16 days oil price as feature
  
# Feature engineering
## unit_sales's and promotion's statistics variable 


# Model Selection
1. MA 0.526
2. RNN(LSTM) 0.516
3. LightGBM 0.510

# Model Evaluation
## Train
1)2017.6.14, 2017.6.21, 2017.6.28, 2017.7.5, 2017.7.12, 2017.7.19
2)2017.6.7, 2017.6.14, 2017.6.21, 2017.6.28, 2017.7.5, 2017.7.12
3)2017.5.31, 2017.6.7, 2017.6.14, 2017.6.21, 2017.6.28, 2017.7.5


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
```
params = {
    'num_leaves': 65,
    'objective': 'regression',
    'min_data_in_leaf': 250,
    'max_bin': 256,
    'learning_rate': 0.015,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2_root',
    'num_threads': 8   
}
```

# Bagging
## change train and val dataset dynamicly by best local cv scores


# Model Ensemble
## 

```
lgb['unit_sales']*0.6 + ma['unit_sales']*0.2 + lstm['unit_sales']*0.2
```

