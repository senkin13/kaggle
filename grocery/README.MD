# Summary
predict 2017.8.16 - 2017.8.30 (16 days) unit_sales of each store's each item 

# Prepocessing
1. only use 2017 data
2. fill unit_sales's nan to 0,fill promotion's nan to false
3. transform train and test data to special format,one row is one item's info, column is date's unit_sales or onpromotion
4. add item,store metadata catagorial features, using one hot encoder
  
# Feature engineering
## unit_sales's and promotion's statistics variable 


# Model Selection
1. MA 0.526
2. LSTM 0.516
3. LightGBM 0.515, 0.514, 0.514, 0.511, 0.510
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
## change train and val dataset
2017.6.7, 2017.6.14, 2017.6.21, 2017.6.28, 2017.7.5, 2017.7.12 - 2017.7.19
2017.5.31, 2017.6.7, 2017.6.14, 2017.6.21, 2017.6.28, 2017.7.5 - 2017.7.12

# Model Ensemble
## Now
### fist layer - 0.513
```
lgb2*0.4 + lstm*0.3 + ma*0.3
```
### sencond layer - 0.513+
```
layer1*0.6 + lgb3*0.2 + lgb4*0.2 
```
### third layer - 0.511
```
layer2*0.8 + lgb5*0.2
```
### fourth layer - 0.510
```
layer3*0.5 + lgb8*0.5 
```
## Optimized
TODO
