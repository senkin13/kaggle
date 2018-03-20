# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:35:00 2018

@author: zhanjin
"""

import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence
import pickle
from text_processing import text_processing
import os

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

max_features = 150000
maxlen = 600
embed_size = 300

train["comment_text"].fillna("fillna")
test["comment_text"].fillna("fillna")

text_processer = text_processing()
train["clean_comment"] = text_processer.remove_stopwords(train["comment_text"]) 
test["clean_comment"] = text_processer.remove_stopwords(test["comment_text"]) 
train["clean_comment"] = train["clean_comment"].apply(
    lambda x: text_processer.clean_text(x))
test["clean_comment"] = test["clean_comment"].apply(
    lambda x: text_processer.clean_text(x))
train["clean_comment"] = train["clean_comment"].apply(
    lambda x: text_processer.glove_preprocess(x))
test["clean_comment"] = test["clean_comment"].apply(
    lambda x: text_processer.glove_preprocess(x))

list_sentences_train = train["clean_comment"]
list_sentences_test = test["clean_comment"]

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
x_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
x_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

word_index = tokenizer.word_index

with open('../input/word_index_150000_600_clean.pkl', 'wb') as handle:
    pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

x_train.dump('../input/x_train_150000_600_clean.pkl')
x_test.dump('../input/x_test_150000_600_clean.pkl')

# fastext
EMBEDDING_FILE = '../input/crawl-300d-2M.vec'

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().split(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))

with open('../input/embeddings_index_fasttext.pkl', 'wb') as handle:
    pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

# glove
EMBEDDING_FILE = '../input/glove.840B.300d.txt'

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().split(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))

with open('../input/embeddings_index_glove.pkl', 'wb') as handle:
    pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# twitter
EMBEDDING_FILE = '../input/glove.twitter.27B.200d.txt'

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().split(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))

with open('../input/embeddings_index_twitter.pkl', 'wb') as handle:
    pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
