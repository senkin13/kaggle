# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


train = pd.read_csv('kfold/train_clean_english.csv')

from sklearn.model_selection import StratifiedKFold
NFOLDS = 10
kfold = StratifiedKFold(n_splits=NFOLDS)

#x_train_0 = train[train['threat']==1][['comment_text','threat']]
#x_0 = x_train_0.comment_text
#x_train_0.to_csv('../input/train_threat_1.csv', index=False)

#x_train_0 = train[train['identity_hate']==1][['comment_text','identity_hate']]
#x_0 = x_train_0.comment_text
#x_train_0.to_csv('../input/train_identity_hate_1.csv', index=False)

#label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
label_cols = ['threat']
for i in label_cols:
    x_train_0 = train[train[i]==0][['comment_text',i]]
    x_0 = x_train_0.comment_text
    y_0 = x_train_0[i]
    x_train_1 = train[train[i]==1][['comment_text',i]]
    x_1 = x_train_1.comment_text
    y_1 = x_train_1[i]
    
    for ii,(train_index,val_index) in enumerate(kfold.split(x_0,y_0)):
        print("Running fold {} / {} / {}".format(i, ii + 1, NFOLDS))
        print("Train Index:",train_index,",Val Index:",val_index)
        X_tra,X_val = x_0[train_index],x_0[val_index]
        y_tra,y_val = y_0[train_index],y_0[val_index]
        X_tra.to_csv('kfold/train_' + str(i) + '_0_' + str(ii) + '.csv', index=False)
        X_val.to_csv('kfold/validation_' + str(i) + '_0_' + str(ii) + '.csv', index=False)
        
    for ii,(train_index,val_index) in enumerate(kfold.split(x_1,y_1)):
        print("Running fold {} / {} / {}".format(i, ii + 1, NFOLDS))
        print("Train Index:",train_index,",Val Index:",val_index)
        X_tra,X_val = x_1[train_index],x_1[val_index]
        y_tra,y_val = y_1[train_index],y_1[val_index]
        X_tra.to_csv('kfold/train_' + str(i) + '_1_' + str(ii) + '.csv', index=False)
        X_val.to_csv('kfold/validation_' + str(i) + '_1_' + str(ii) + '.csv', index=False) 
