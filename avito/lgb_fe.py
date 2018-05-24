%%time

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
import glob

df_all = pickle.load(open('../input/df_all_agg.pkl','rb'))
for fn in glob.glob('../features/text_agg/*'):
    tmp = pickle.load(open(fn,'rb'))
    df_all[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    print (os.path.basename(fn))
for fn in glob.glob('../features/number_agg/*'):
    tmp = pickle.load(open(fn,'rb'))
    df_all[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    print (os.path.basename(fn))    
    
    
    
