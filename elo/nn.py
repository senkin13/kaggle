%%time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import optimizers
from keras.optimizers import Adam
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.layers.advanced_activations import PReLU
from keras import layers
from keras.layers import LeakyReLU
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers import Dropout, Embedding, Permute, Concatenate, Flatten, Reshape
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D, PReLU
from keras.layers.advanced_activations import LeakyReLU, PReLU,ELU
from keras.models import Model
from keras.layers import GRU, LSTM, CuDNNGRU, CuDNNLSTM, Bidirectional, Lambda
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,LearningRateScheduler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from keras.regularizers import l1, l2, l1_l2
from keras import optimizers
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import random
SEED = 2019
np.random.seed(SEED)
random.seed(SEED)

# save load_iris() sklearn dataset to iris
# if you'd like to check dataset type use: type(load_iris())
# if you'd like to view list of attributes use: dir(load_iris())
iris = load_iris()

# np.c_ is the numpy concatenate function
# which is used to concat iris['data'] and iris['target'] arrays 
# for pandas column argument: concat iris['feature_names'] list
# and string list (in this case one string); you can make this anything you'd like..  
# the original dataset would probably call this ['Species']
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= ['f1','f2','f3','f4','target'])
                     

def nn(num_cols, cat_cols):
    # numerical
    numerical = Input(shape=(len(num_cols),), dtype='float32')
    numerical_inputs = [numerical]
    n = numerical
    n = BatchNormalization()(n)
    n = Dense(64)(n)
    
    # categorical
    f4_embedding_size = 8
    f4_input = Input(shape=(1,), )
    categorical_inputs = [f4_input]
    f4_embedded = Embedding(4, f4_embedding_size, 
                                        input_length=1, name='f4_embedding')(f4_input)
    c = Flatten()(f4_embedded)
    
    # timeseries
    timeseries = Input(shape=(1,len(num_cols)))
    timeseries_inputs = [timeseries]
    t = Bidirectional(LSTM(32,return_sequences=True))(timeseries)
    t = LSTM(32)(t)

    # merge
    x = concatenate([n, c, t])
    x = Dense(64)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(numerical_inputs+categorical_inputs+timeseries_inputs, output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

    
cat_cols = ['f4']
num_cols = ['f1','f2']

df_numerical = df[num_cols].values
df_embedding = df[cat_cols].values.astype(np.int32)
df_ts = df_numerical.reshape((df_numerical.shape[0], 1, df_numerical.shape[1]))
y = df['target'].values
    
# max_values = {}
# for col in cat_cols:
#     print(col)
#     lbl = LabelEncoder()
#     df[col] = lbl.fit_transform(df[col].astype('str'))
#     max_values[col] = max(df[col].max())  + 2
    
# df,y = get_keras_data(df,num_cols, cat_cols)
model = nn(num_cols, cat_cols)
history = model.fit([df_numerical]+[df_embedding]+[df_ts], df['target'], batch_size=256, epochs=20, verbose=1,validation_split=.05,)   

