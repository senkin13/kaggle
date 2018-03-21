import numpy as np
np.random.seed(100)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model,load_model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Dropout, Add, Flatten
from keras.layers import CuDNNGRU, GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler
from keras.optimizers import Adam, Nadam
import pickle
import gc
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm

import gensim

import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
test_id = submission.drop(['id'],axis=1)

# replace urls
re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
                    .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                    re.MULTILINE|re.UNICODE)
# replace ips
re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

# setup tokenizer
tokenizer = WordPunctTokenizer()

vocab = Counter()

def text_to_wordlist(text, lower=False):
    # replace URLs
    text = re_url.sub("URL", text)
    
    # replace IPs
    text = re_ip.sub("IPADDRESS", text)
    
    # Tokenize
    text = tokenizer.tokenize(text)
    
    # optional: lower case
    if lower:
        text = [t.lower() for t in text]
    
    # Return a list of words
    vocab.update(text)
    return text

def process_comments(list_sentences, lower=False):
    comments = []
    for text in tqdm(list_sentences):
        txt = text_to_wordlist(text, lower=lower)
        comments.append(txt)
    return comments


list_sentences_train = list(train_df["comment_text"].fillna("NAN_WORD").values)
list_sentences_test = list(test_df["comment_text"].fillna("NAN_WORD").values)

comments = process_comments(list_sentences_train + list_sentences_test, lower=True)

from gensim.models import Word2Vec

model = Word2Vec(comments, size=100, window=5, min_count=5, workers=16, sg=0, negative=5)

word_vectors = model.wv

print("Number of word vectors: {}".format(len(word_vectors.vocab)))

model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])

MAX_NB_WORDS = len(word_vectors.vocab)
MAX_SEQUENCE_LENGTH = 200

from keras.preprocessing.sequence import pad_sequences

word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(MAX_NB_WORDS))}
sequences = [[word_index.get(t, 0) for t in comment]
             for comment in comments[:len(list_sentences_train)]]
test_sequences = [[word_index.get(t, 0)  for t in comment] 
                  for comment in comments[len(list_sentences_train):]]

# pad
train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, 
                     padding="pre", truncating="post")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_df[list_classes].values
print('Shape of data tensor:', train_data.shape)
print('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre",
                          truncating="post")
print('Shape of test_data tensor:', test_data.shape)

WV_DIM = 100
nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
# we initialize the matrix with random numbers
wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass    
    
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)
    
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

print('FE Done')            

def get_model():
    inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    x = wv_layer(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(512, return_sequences=True))(x)
    x = Conv1D(128, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(x)
    #x = Dropout(0.5)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(0.5)(x)
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
NFOLDS = 10
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=1998)
y_pred = test_id
batch_size = 32
epochs = 10

for i,(train_index,val_index) in enumerate(kfold.split(train_data)):
    print("Running fold {} / {}".format(i + 1, NFOLDS))
    print("Train Index:",train_index,",Val Index:",val_index)
    X_tra,X_val = train_data[train_index],train_data[val_index]
    y_tra,y_val = y[train_index],y[val_index]
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

        y_pred = model.predict(test_data, batch_size=2048)
        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
        submission.to_csv("../models/rcnn_w2v_kfold_" + str(i+1) + ".csv" , index=False)
