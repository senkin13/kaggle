import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

rdnn = pd.read_csv("../sub/rdnn_k10.csv") #0.9864
cap = pd.read_csv("../sub/capsule_k10.csv") #0.9854
gru = pd.read_csv("../sub/gru_k10.csv") #0.9853
none = pd.read_csv("../sub/none_nn_0.9837.csv") #0.9837
nn = pd.read_csv("../sub/nn_k10.csv") #0.9836
w2v = pd.read_csv("../sub/rcnn_w2v_k10.csv") #0.9832



blend = gru.copy()
col = blend.columns

col = col.tolist()
col.remove('id')


blend[col] = ((rdnn[col].values)*40 +
              (gru[col].values)*25 +
              (cap[col].values)*25 +
              (nn[col].values)*20 +
              (none[col].values)*20 +
              (w2v[col].values)*20 
             ) / 150

blend.to_csv("../sub/ensemble_m.csv", index=False)

glove = pd.read_csv("gru_k10.csv") #0.9858
rdnn = pd.read_csv("rdnn.csv") #0.9851
#rcnn = pd.read_csv("rcnn_glove_k10.csv") #0.9846
rcnn = pd.read_csv("rcnn.csv") #0.9841
nn = pd.read_csv("nn.csv") #0.9836
nbk = pd.read_csv("nb_kfold.csv") #0.9816
wordbatch = pd.read_csv("wordbatch.csv") #0.9805
lgbk = pd.read_csv("lgbk.csv") #0.9804
lgbn = pd.read_csv("lgbn.csv") #0.9800
lr = pd.read_csv("lr.csv") #0.9792
#glm2 = pd.read_csv("glm.csv") #0.9785
glm = pd.read_csv("glm.csv") #0.9771
word2vec = pd.read_csv("word2vec.csv") #0.9766
cnn = pd.read_csv("cnn.csv") #0.9732

blend = glove.copy()
col = blend.columns

col = col.tolist()
col.remove('id')

blend[col] = (minmax_scale(glove[col].values)*25 +
              minmax_scale(rdnn[col].values)*22 +
              minmax_scale(rcnn[col].values)*17 +
              minmax_scale(nn[col].values)*14 + 
              minmax_scale(nbk[col].values)*6 +
              minmax_scale(lgbk[col].values)*4 +  
              minmax_scale(lgbn[col].values)*3 + 
              minmax_scale(lr[col].values)*3 + 
              minmax_scale(glm2[col].values)*2 + 
              minmax_scale(word2vec[col].values)*2 + 
              minmax_scale(cnn[col].values)*2) / 100

blend.to_csv("../sub/ensemble_minmax.csv", index=False)
