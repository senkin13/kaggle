%%time
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings 
warnings.filterwarnings('ignore')

################historical
#df_hist = pd.read_csv("../input/historical_transactions.csv")
df_new = pd.read_csv("../input/new_merchant_transactions.csv")

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

temp = df_new[['card_id','merchant_id']].groupby(['card_id','merchant_id'],as_index=False)['card_id'].agg({'count'}).reset_index()

lbl1,lbl2 = LabelEncoder(),LabelEncoder()
temp['merchant_id'] = lbl1.fit_transform(temp['merchant_id'].map(str))

temp['card_id'] = lbl2.fit_transform(temp['card_id']) + (temp['merchant_id'].max()+1)

temp.to_csv('../deepwalk/new_merchantid_deepwalk.csv',index=False,header=False,sep=' ')

import os
os.system('deepwalk --input ../deepwalk/new_merchantid_deepwalk.csv --format edgelist \
--output ../deepwalk/new_merchantid_deepwalk.emb --workers 40')

npy = np.loadtxt('../deepwalk/new_merchantid_deepwalk.emb', delimiter=' ')

kfc = pd.DataFrame()
kfc['card_id'] = npy[:, 0]
for i in range(1, 65):
    #print (i)
    kfc['merchantid_deepwalk_' + str(i)] = npy[:, i]
    
kfc = kfc[~kfc['card_id'].isin(temp['merchant_id'].unique())]

kfc['card_id'] = kfc['card_id'] - (temp['merchant_id'].max()+1)

kfc['card_id'] = kfc['card_id'].astype(int)

kfc['card_id'] = lbl2.inverse_transform(kfc['card_id'])

kfc.to_pickle('../deepwalk/kfc_new.pkl')

%%time
import networkx as nx
import node2vec as n2v
from node2vec import Node2Vec

G = nx.DiGraph()
G.add_weighted_edges_from(temp[['card_id','merchant_id','count']].values)

node2vec = Node2Vec(G, dimensions=64,walk_length=30,num_walks=200,workers=8)

model = node2vec.fit(window=5,min_count=1,batch_words=4)

# Save embeddings for later use
model.wv.save_word2vec_format('../deepwalk/node2vec_new.bin')

npy = np.loadtxt('../deepwalk/node2vec_new.bin', delimiter=' ')
n2v = pd.DataFrame()
n2v['card_id'] = npy[:, 0]
for i in range(1, 65):
    #print (i)
    n2v['merchantid_node2vec_' + str(i)] = npy[:, i]
    
n2v = n2v[~n2v['card_id'].isin(temp['merchant_id'].unique())]
n2v['card_id'] = n2v['card_id'] - (temp['merchant_id'].max()+1)
n2v['card_id'] = n2v['card_id'].astype(int)
n2v['card_id'] = lbl2.inverse_transform(n2v['card_id'])

n2v.to_pickle('../deepwalk/n2v_new.pkl')
