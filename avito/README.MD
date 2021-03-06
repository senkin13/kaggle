# 9th Rank Of Kaggle Avito Demand Prediction Challenge

# Prepocessing
```
df_all['price'].fillna(0, inplace=True)
df_all['price'] = np.log1p(df_all['price'])
df_all['city'] = df_all['city'] + "_" + df_all['region']
df_all['param_123'] = (df_all['param_1'].fillna('') + ' ' + df_all['param_2'].fillna('') + ' ' + df_all['param_3'].fillna('')).astype(str)
df_all['text'] = df_all['description'].fillna('').astype(str) + ' ' + df_all['title'].fillna('').astype(str) 
```

# Feature Engineering
```
[category] 
[numerical]
[aggregation] 
[text aggregation] 
[text tfidf]
[word2vec]
[fasttext]
[image meta]
[vgg16,resnet,xception output]
```

# MODEL
```
Lightgbm
Xgboost
Catboost
Ridge Regression
Random Forest

Gru-Attention-Dense-Dense
Bigru-Cnn
Dpcnn

```

## Ensembling
```
Level1: 60 singles models
Level2: stacking of 60 models,kernel models,blend models,single models with diversity
Level3: quiz blending(https://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf)
```

## Magic feature
```I’ve designed 3 nn models trained on active data to predict log1p(price), renewed and log1p(total_duration). These models had 2 rnn branches for title and description and also used category embeddings. The difference between actual log1p(price) and predicted one was an extremely important feature.```

## Best nn
https://www.kaggle.com/c/avito-demand-prediction/discussion/59880

## Best Stacking
https://www.kaggle.com/c/avito-demand-prediction/discussion/59871

