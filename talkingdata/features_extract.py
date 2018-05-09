import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import pickle
import gensim

%%time

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        }
X = pd.read_csv('../input/full.csv',parse_dates=['click_time'],dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time','is_attributed'])
X['day'] = X['click_time'].dt.day
X['hour'] = X['click_time'].dt.hour


