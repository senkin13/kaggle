import numpy as np
np.random.seed(99)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model,load_model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Dropout
from keras.layers import CuDNNGRU, GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler
from keras.optimizers import Adam, Nadam
import pickle
import gc

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
test_id = submission.drop(['id'],axis=1)

#X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
#X_test = test["comment_text"].fillna("fillna").values

max_features = 100000
maxlen = 500
embed_size = 300

x_train = pickle.load(open('../input/x_train_100000_500_clean.pkl','rb'))
x_test = pickle.load(open('../input/x_test_100000_500_clean.pkl','rb'))
embeddings_index = pickle.load(open('../input/embeddings_index_glove.pkl','rb'))
word_index = pickle.load(open('../input/word_index_100000_500_clean.pkl','rb'))


all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

#tokenizer = text.Tokenizer(num_words=max_features)
#tokenizer.fit_on_texts(list(X_train) + list(X_test))
#word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

class RocAucEvaluation(Callback):
    def __init__(self, filepath, validation_data=(), interval=10, max_epoch = 20):
        super(Callback, self).__init__()

        self.interval = interval
        self.filepath = filepath
        self.stopped_epoch = max_epoch
        self.best = 0
        self.X_val, self.y_val = validation_data
        self.y_pred = np.zeros(self.y_val.shape)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            """Important lines"""
            current = roc_auc_score(self.y_val, y_pred)
            logs['roc_auc_val'] = current

            if current > self.best: #save model
                self.best = current
                self.y_pred = y_pred
                self.stopped_epoch = epoch+1
                self.model.save(self.filepath, overwrite = True)
            print("--- AUC - epoch: {:d} - score: {:.5f}\n".format(epoch+1, current))

    def cv_score(self):
        self.cv_score = self.best
        return self.cv_score

print('FE Done')            

def get_model():
    inp = Input(shape = (max_len,))
    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.1)(x)

    x = Bidirectional(CuDNNGRU(256, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    outp = Dense(6, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    lr=0.001
    decay = lr / 10
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


from sklearn.model_selection import KFold
from datetime import datetime
NFOLDS = 10
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=228)
y_pred = test_id
batch_size = 32
epochs = 5

for i,(train_index,val_index) in enumerate(kfold.split(x_train)):
    print("Running fold {} / {}".format(i + 1, NFOLDS))
    print("Train Index:",train_index,",Val Index:",val_index)
    X_tra,X_val = x_train[train_index],x_train[val_index]
    y_tra,y_val = y_train[train_index],y_train[val_index]
    print (X_tra.shape)
    print (X_val.shape)
    print (y_tra.shape)
    print (y_val.shape)

    model = get_model()
    file_path = "rcnn_fold " + str(i+1) + " best_model.hdf5"
    ra_val = RocAucEvaluation(file_path, validation_data=(X_val, y_val), interval=1)
    es = EarlyStopping(monitor='roc_auc_val', patience=2, mode='max', verbose=1)

    if i >= 0:
        hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[ra_val, es], verbose=1)

        model = load_model(file_path)
        del X_tra,y_tra,X_val,y_val
        gc.collect()

        y_pred = model.predict(x_test, batch_size=256)
        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
        submission.to_csv("../models/rcnn_glove_kfold_" + str(i+1) + ".csv" , index=False)
