import pickle
import pandas as pd
import numpy as np

print ('load pickle....')
train_df = pickle.load(open('../input/train_df.pkl','rb'))
test_df = pickle.load(open('../input/test_df.pkl','rb'))
y_train = pickle.load(open('../input/y_train.pkl','rb'))

print ('neural network....')
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
max_dev = np.max([train_df['device'].max(), test_df['device'].max()])+1
max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
max_h = np.max([train_df['hour'].max(), test_df['hour'].max()])+1
max_d = np.max([train_df['day'].max(), test_df['day'].max()])+1
max_wd = np.max([train_df['wday'].max(), test_df['wday'].max()])+1
max_qty = np.max([train_df['qty'].max(), test_df['qty'].max()])+1
max_c1 = np.max([train_df['ip_app_count'].max(), test_df['ip_app_count'].max()])+1
max_c2 = np.max([train_df['ip_app_os_count'].max(), test_df['ip_app_os_count'].max()])+1
def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'd': np.array(dataset.day),
        'wd': np.array(dataset.wday),
        'qty': np.array(dataset.qty),
        'c1': np.array(dataset.ip_app_count),
        'c2': np.array(dataset.ip_app_os_count)
    }
    return X
train_df = get_keras_data(train_df)

emb_n = 50
dense_n = 1000
in_app = Input(shape=[1], name = 'app')
emb_app = Embedding(max_app, emb_n)(in_app)
in_ch = Input(shape=[1], name = 'ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_dev = Input(shape=[1], name = 'dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev)
in_os = Input(shape=[1], name = 'os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_h = Input(shape=[1], name = 'h')
emb_h = Embedding(max_h, emb_n)(in_h) 
in_d = Input(shape=[1], name = 'd')
emb_d = Embedding(max_d, emb_n)(in_d) 
in_wd = Input(shape=[1], name = 'wd')
emb_wd = Embedding(max_wd, emb_n)(in_wd) 
in_qty = Input(shape=[1], name = 'qty')
emb_qty = Embedding(max_qty, emb_n)(in_qty) 
in_c1 = Input(shape=[1], name = 'c1')
emb_c1 = Embedding(max_c1, emb_n)(in_c1) 
in_c2 = Input(shape=[1], name = 'c2')
emb_c2 = Embedding(max_c2, emb_n)(in_c2) 
fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h), 
                 (emb_d), (emb_wd), (emb_qty), (emb_c1), (emb_c2)])
s_dout = SpatialDropout1D(0.2)(fe)
x = Flatten()(s_dout)
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_d,in_wd,in_qty,in_c1,in_c2], outputs=outp)

batch_size = 20000
epochs = 2
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train_df) / batch_size) * epochs
lr_init, lr_fin = 0.001, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.001, decay=lr_decay)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

model.summary()

model.fit(train_df, y_train, batch_size=batch_size, epochs=2, verbose=1)
del train_df, y_train; gc.collect()
model.save_weights('../hdf5/dnn.h5')

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
test_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)
test_df = get_keras_data(test_df)

print("predicting....")
sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=1)
del test_df; gc.collect()
print("writing....")
sub.to_csv('../models/dnn.csv',index=False)
