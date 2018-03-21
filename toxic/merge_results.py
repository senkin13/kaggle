import pandas as pd
import numpy as np

submission = pd.read_csv("./input/sample_submission.csv")

df1 = pd.read_csv("./models/rcnn_w2v_kfold_1.csv")
df2 = pd.read_csv("./models/rcnn_w2v_kfold_2.csv")
df3 = pd.read_csv("./models/rcnn_w2v_kfold_3.csv")
df4 = pd.read_csv("./models/rcnn_w2v_kfold_4.csv")
df5 = pd.read_csv("./models/rcnn_w2v_kfold_5.csv")
df6 = pd.read_csv("./models/rcnn_w2v_kfold_6.csv")
df7 = pd.read_csv("./models/rcnn_w2v_kfold_7.csv")
df8 = pd.read_csv("./models/rcnn_w2v_kfold_8.csv")
df9 = pd.read_csv("./models/rcnn_w2v_kfold_9.csv")
df10 = pd.read_csv("./models/rcnn_w2v_kfold_10.csv")

for i in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
    submission[i] = (df1[i] + df2[i] + df3[i] + df4[i] + df5[i] + df6[i] + df7[i] + df8[i] + df9[i] + df10[i]) / 10
submission.to_csv("./sub/rcnn_w2v_k10.csv", index=False)

train = pd.read_csv("./models/rcnn/rcnn_glove_kfold_1toxic.csv")
submission = pd.read_csv("./input/sample_submission.csv")
df = np.zeros(train.shape[0],)

#for i in ['toxic']:
for i in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
    df0 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_0" + str(i) + ".csv", names=[i])
    df1 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_1" + str(i) + ".csv", names=[i])
    df2 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_2" + str(i) + ".csv", names=[i])
    df3 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_3" + str(i) + ".csv", names=[i])
    df4 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_4" + str(i) + ".csv", names=[i])
    df5 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_5" + str(i) + ".csv", names=[i])
    df6 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_6" + str(i) + ".csv", names=[i])
    df7 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_7" + str(i) + ".csv", names=[i])
    df8 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_8" + str(i) + ".csv", names=[i])
    df9 = pd.read_csv("./models/rcnn/rcnn_glove_kfold_9" + str(i) + ".csv", names=[i])
    submission[i] = (df0[i] + df1[i] + df2[i] + df3[i] + df4[i] + df5[i] + df6[i] + df7[i] + df8[i] + df9[i]) / 10
    submission.to_csv("./sub/gru_glove_k10.csv", index=False)
    
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for i in label_cols:
    for j in range(10):
        x_train_0 = pd.read_csv('kfold/train_' + str(i) + '_0_' + str(j) + '.csv', names='comment_text')
        x_train_1 = pd.read_csv('kfold/train_' + str(i) + '_1_' + str(j) + '.csv', names='comment_text')
        x_train_0[i] = 0
        x_train_1[i] = 1
        x_train = pd.concat([x_train_0,x_train_1])
        x_train.to_csv('kfolds/train_' + str(i) + str(j) + '.csv')
 
        x_validation_0 = pd.read_csv('kfold/validation_' + str(i) + '_0_' + str(j) + '.csv', names='comment_text')
        x_validation_1 = pd.read_csv('kfold/validation_' + str(i) + '_1_' + str(j) + '.csv', names='comment_text')
        x_validation_0[i] = 0
        x_validation_1[i] = 1
        x_validation = pd.concat([x_validation_0,x_validation_1])
        x_validation.to_csv('kfolds/validation_' + str(i) + str(j) + '.csv')    
