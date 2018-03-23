
# Preprocessing
- stopwords
- lower
- fillna
- bad words
- good words
- spell correct 
- translation

# Feature Engineering
pretrained word embedding
concatenation of pretrained word embedding
self-trained word embedding

# Pretrained Vectors
- word2vec
- fasttext
- glove

# Train and Validation
Kfold10
StratifiedKFold

# Model
## Traditional 
Logistic Regression
Lightgbm
Wordbatch
SVM

## Neural Network
rdnn 
capsule net 
rnn1(1 layer GRU)
rnn(2 layer GRU) - https://github.com/PavelOstyakov/toxic/blob/master/toxic/model.py
rcnn

## Blending
### 1st layer traditional models
none = (Logistic Regression + Lightgbm + Wordbatch +SVM) /4
### 2rd layer
(rdnn[col].values)*40 +
              (rnn1[col].values)*25 +
              (cap[col].values)*25 +
              (rnn2[col].values)*20 +
              (none[col].values)*20 +
              (w2v[col].values)*20 
             ) / 150
