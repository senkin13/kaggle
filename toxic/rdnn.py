import os
import codecs
INPUT = '../'

from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, SpatialDropout1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNLSTM, GRU, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler, Callback
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Nadam
from keras import backend as K
import gc
import random
max_features = 200000
maxlen = 500
SEED = 17
np.random.seed(SEED)
random.seed(SEED)
train = pd.read_csv(INPUT+"input/train.csv",encoding='utf-8')
test = pd.read_csv(INPUT+"input/test.csv",encoding='utf-8')
submission = pd.read_csv('../input/sample_submission.csv')
test_id = submission.drop(['id'],axis=1)

#preprocessing
import regex as re
import unicodedata
fword_list = ['****','****','****'] #some toxic words
def preprocess(text):
    try:
        text = re.sub(r"\p{P}+|\p{Z}+|\p{S}+|\p{N}+", ' ', text)
        text = unicodedata.normalize('NFKD',text)#.encode('ascii','ignore')
        text = re.sub(r"\p{M}+", '', text)
        text = re.sub(r"\p{P}+|\p{S}+|\p{N}+|\p{Cs}+|\p{Cf}+|\p{Co}+", '', text)
        
        text = re.sub("([A-Za-z]+)", lambda m:m.group(1).lower(),text)
        text = re.sub(r'([^\x00-\x7f])', lambda m:' '+m.group(1)+' ', text)
        
        text = re.sub(r"(\w)\1{2,}",lambda m:m.group(1), text)
        text = re.sub("(\s+)", ' ',text)
        
        for fword in fword_list:
            fre_ = ''
            for i in xrange(len(fword)):
                w = fword[i]
                fre_ += w + "+\s*" #if i < (len(fword)-1) else w + "+" 
            text = re.sub(fre_, ' '+fword+' ',text)
        text = re.sub("(\s+)", ' ',text)
        return text
    except: 
        return text

train['comment_text'] = train['comment_text'].apply(lambda x: preprocess(x))
test['comment_text'] = test['comment_text'].apply(lambda x: preprocess(x))
train["comment_text"].fillna("[na]")
test["comment_text"].fillna("[na]")


from collections import Counter
cc = Counter()
def get_split(sentence):
    try:
        s = sentence.strip().split(' ')
        d = Counter(s)
        cc.update(d)
        return s
    except: 
        return sentence

train['comment_split'] = train['comment_text'].apply(lambda x: get_split(x))
test['comment_split'] = test['comment_text'].apply(lambda x: get_split(x))

def remove_one(text):
    try:       
        text = u" ".join(
        [x for x in [y for y in text.strip().split(u" ")] \
         if cc[x] > 1])
        return text
    except: 
        return text

train['comment_text'] = train['comment_text'].apply(lambda x: remove_one(x))
test['comment_text'] = test['comment_text'].apply(lambda x: remove_one(x))

list_sentences_train = train["comment_text"].fillna("[na]").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_classes_pred = ["toxic_pred", "severe_toxic_pred", "obscene_pred", "threat_pred", "insult_pred", "identity_hate_pred"]
y_train = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("[na]").values
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

x_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
x_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

EMBEDDING_FILE='../input/crawl-300d-2M.vec'
EMBEDDING_MY = '../input/vectors.txt'
embed_size = 300 # how big is each word vector
cn = 0
def get_coefs(word,*arr): 
    global cn
    cn += 1
    dict_v = np.asarray(arr, dtype='float32')
    if len(dict_v)!=embed_size:
        dict_v = np.zeros((embed_size))
    return word, dict_v
f_emb = codecs.open(EMBEDDING_FILE)
emb_list = f_emb.readlines()
embeddings_index = dict(get_coefs(*o.strip().split()) for o in emb_list)
print (cn)
f_emb.close()

f_emb = codecs.open(EMBEDDING_MY,'r','utf-8')
emb_list = f_emb.readlines()
cn = 0
embeddings_index_my = dict(get_coefs(*o.strip().split()) for o in emb_list)
print (cn)
f_emb.close()

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print (emb_mean,emb_std)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
no_embedding = 0
for word, i in word_index.items():
    if i >= nb_words: continue
    embedding_vector = embeddings_index.get(word)
    embedding_vector_my = embeddings_index_my.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    elif embedding_vector_my is not None: 
        embedding_matrix[i] = embedding_vector_my
    else: 
        print (word)
        no_embedding += 1
print (no_embedding,nb_words-no_embedding,len(word_index))

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

def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(64, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    opt = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

from sklearn.model_selection import KFold
from datetime import datetime
NFOLDS = 10
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
y_pred = test_id
batch_size = 32
epochs = 10

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

    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(x_train)/batch_size) * epochs
    lr_init, lr_fin = 0.001, 0.0005
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)

    num = 0
    if not os.path.isdir(INPUT+"models/"):
        os.mkdir(INPUT+"models/")
    if not os.path.isdir(INPUT+"models/"+str(num)):
        os.mkdir(INPUT+"models/"+str(num))
    file_path_best=INPUT+"models/"+str(num)+"/"+"weights_best"+str(num)+".hdf5"
    ra_val = RocAucEvaluation(file_path_best, validation_data=(X_val, y_val), interval=1)
    es = EarlyStopping(monitor='roc_auc_val', patience=2, mode='max', verbose=1)

    callbacks_list = [ra_val, es]
    if i >= 0:
        model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val,y_val), callbacks=callbacks_list, verbose=1)

        if os.path.isfile(file_path_best):
            print ('load ',file_path_best)
            model.load_weights(file_path_best)

        y_pred = model.predict([x_test], batch_size=256, verbose=1)
        del X_tra,y_tra,X_val,y_val
        gc.collect()
        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
        submission.to_csv(INPUT+"rdnn_kfold"+str(i+1)+".csv", index=False)
