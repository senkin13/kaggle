# Prepocessing

# Feature engineering
## unit_sales

## promotion

# Model Selection
1. MA
2. LSTM
3. LightGBM
4. TODO(Xgboost catboost nn)

# Hyper parameter tuning

# Bagging

# Model Ensemble
## fist layer
lgb2*0.4 + lstm*0.3 + ma*0.3
## sencond layer
layer1*0.6 + lgb3*0.2 + lgb4*0.2 
## third layer
layer2*0.8 + 0.2*lgb5 
