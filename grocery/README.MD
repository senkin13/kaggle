# Summary
predict unit_sales of each store's each item 

# Prepocessing
1. only use 2017 data
2. fill unit_sales's nan to 0,fill promotion's nan to false
3. transform train and test data to ,one row, one item including 
                    date
store	  item        2017-01-01	2017-01-02	2017-01-03
1	      96995		    0.000000	0.000000	0.000000
        103665	    0.000000	0.000000	1.386294	
        105574		  0.000000	1.791759	2.564949	
# Feature engineering
## unit_sales

## promotion

# Model Selection
1. MA
2. LSTM
3. LightGBM ()
4. TODO(Xgboost catboost nn)

# Model Evaluation
## Train
2017.6.14, 2017.6.21, 2017.6.28, 2017.7.5, 2017.7.12, 2017.7.19

## Validation
### public Validation
2017.7.26
### private Validation
2017.7

## Test
2017.8.16

# Hyper parameter tuning

# Bagging

# Model Ensemble
## fist layer
lgb2*0.4 + lstm*0.3 + ma*0.3
## sencond layer
layer1*0.6 + lgb3*0.2 + lgb4*0.2 
## third layer
layer2*0.8 + 0.2*lgb5 
