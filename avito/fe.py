import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
import numpy as np
import pickle
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df_all = pickle.load(open('../input/df_all.pkl','rb'))

%%time

## count text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
import re
import string

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))

        
df_all_text = df_all[['description','title']]
df_all_text['text'] = (df_all_text['description'].fillna('') + ' ' + df_all_text['title'].fillna(''))        
all = df_all_text.copy()

# Meta Text Features
textfeats = ['text']
for cols in textfeats:   

    all[cols + '_num_pun1'] = all[cols].apply(lambda x: count_regexp_occ('.', x))
    all[cols + '_num_pun2'] = all[cols].apply(lambda x: count_regexp_occ(',', x))
    all[cols + '_num_pun3'] = all[cols].apply(lambda x: count_regexp_occ('-', x))
    all[cols + '_num_pun4'] = all[cols].apply(lambda x: count_regexp_occ('!', x))
    all[cols + '_num_pun5'] = all[cols].apply(lambda x: count_regexp_occ('/', x)) 
    all[cols + '_num_pun6'] = all[cols].apply(lambda x: count_regexp_occ('Г', x))    
    all[cols + '_num_pun7'] = all[cols].apply(lambda x: count_regexp_occ('\(', x))
    all[cols + '_num_pun8'] = all[cols].apply(lambda x: count_regexp_occ('"', x)) 
    all[cols + '_num_pun9'] = all[cols].apply(lambda x: count_regexp_occ('\'', x))  
    
textfeats = ['title']
for cols in textfeats:   

    all[cols + '_num_pun1'] = all[cols].apply(lambda x: count_regexp_occ('.', x))
    all[cols + '_num_pun2'] = all[cols].apply(lambda x: count_regexp_occ(',', x))
    all[cols + '_num_pun3'] = all[cols].apply(lambda x: count_regexp_occ('-', x))
    all[cols + '_num_pun4'] = all[cols].apply(lambda x: count_regexp_occ('!', x))
    all[cols + '_num_pun5'] = all[cols].apply(lambda x: count_regexp_occ('/', x)) 
    all[cols + '_num_pun6'] = all[cols].apply(lambda x: count_regexp_occ('Г', x))    
    all[cols + '_num_pun7'] = all[cols].apply(lambda x: count_regexp_occ('\(', x))
    all[cols + '_num_pun8'] = all[cols].apply(lambda x: count_regexp_occ('"', x)) 
    all[cols + '_num_pun9'] = all[cols].apply(lambda x: count_regexp_occ('\'', x)) 
    
df_all_tmp = all.drop(['description','title','text'],axis=1)
tmp_columns = df_all_tmp.columns.values

for i in tmp_columns:
    print (i)
    df_all_tmp[i].to_pickle('../features/text_agg/' + str(i))
    
