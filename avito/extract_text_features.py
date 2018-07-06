import itertools
import gc
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
from functools import partial
from gensim.models import FastText, word2vec
from gensim.models.word2vec import PathLineSentences
from gensim import corpora, models
from tqdm import tqdm
from nltk.corpus import stopwords 
from pymystem3 import Mystem

seed = 777
np.random.seed(seed)

target = 'deal_probability'
text_features = ['title', 'description']
# text_features = ['title', 'category_name', 'parent_category_name', 'description', 'param_1', 'param_2', 'param_3']
numerical_features = ['price']
remove_columns = ['index', 'image', 'user_id', 'item_id', 'text']

# preprocess parameters
maxlen = 50
num_words = 100000
max_features = 18000 # tfidf
padding = 'pre'
# padding = 'post'
lower = True
lemmatize = False
remove_stopwords = False
remove_non_ascii = False

ru_stopwords = stopwords.words('russian')
mystem = Mystem() 

def preprocess_text(text, lower, lemmatize, remove_stopwords):
    if lower:
        text = text.lower()
    if lemmatize:
        text = mystem.lemmatize(text)
    if remove_stopwords:
        text = " ".join([w for w in text.split() if not w in ru_stopwords])
    if remove_non_ascii:
        pattern = re.compile('[^(?u)\w\s]+')
        text = re.sub(pattern, "", text)
    if len(text) == 0:
        text = ' '
        
    return text

def merge_text(dataframe, text_features):
    dataframe['text'] = dataframe['title'].str.cat(dataframe['description'], sep=' ', na_rep='')
#     dataframe['text'] = dataframe['text'].apply(lambda x: preprocess_text(x, lower, lemmatize, remove_stopwords))
    
    return dataframe
    
# save text files for gensim PathLineSentences
chuncksize = 1000000
files = ['../input/train.csv', '../input/test.csv', '../input/train_active.csv', '../input/test_active.csv']

count = 0
for file in files:
    print('processing {}...'.format(file))
    for dataframe in pd.read_csv(file, chunksize=chuncksize, usecols=text_features):
        dataframe = merge_text(dataframe, text_features)
        np.savetxt('../input/text/text_{}.txt'.format(count), dataframe.text.values, fmt='%s')
        del dataframe; gc.collect()
        count += 1
        
model = FastText(PathLineSentences('../input/text/'), size=300, window=5, min_count=5, word_ngrams=1, seed=seed, workers=6)
# save model
model.save('./../data/text_model/avito.ru.f.vec')
model.wv.save_word2vec_format('./../data/avito.ru.ft.vec')

model = word2vec.Word2Vec(PathLineSentences('../input/text/'), size=300, window=5, min_count=5, seed=seed, workers=6)
# save model
model.save('./../data/text_model/avito.ru.w2v.vec')
model.wv.save_word2vec_format('./../data/avito.ru.w2v.vec')

# LDA
# train = pd.read_feather("../input/train.ftr")
# test = pd.read_feather("../input/test.ftr")
# print(train.shape)
# print(test.shape)
# len_train = len(train)
# train = train.append(test, sort=False).reset_index(drop=True)
# train['is_train'] = np.hstack((np.ones(len_train), np.zeros(len(test))))
# del test; gc.collect()
# train.head()

# # preprocess parameters
# lower = True
# lemmatize = False
# remove_stopwords = False
# remove_non_ascii = False

# # tokenizer parameters
# maxlen = 50
# num_words = 100000
# max_features = 18000 # tfidf
# embedding_dims = 300
# padding = 'pre'
# # padding = 'post'

# ru_stopwords = stopwords.words('russian')
# mystem = Mystem() 

# def preprocess_text(text, lower, lemmatize, remove_stopwords):
#     if lower:
#         text = text.lower()
#     if lemmatize:
#         text = mystem.lemmatize(text)
#     if remove_stopwords:
#         text = " ".join([w for w in text.split() if not w in ru_stopwords])
#     if remove_non_ascii:
#         pattern = re.compile('[^(?u)\w\s]+')
#         text = re.sub(pattern, "", text)
#     if len(text) == 0:
#         text = ' '
#     # remove multiple spaces
# #     text = re.sub(' +', ' ', text)
    
#     return text

# _preprocess_text = partial(preprocess_text, lower=lower, lemmatize=lemmatize, remove_stopwords=remove_stopwords)

# def merge_text(dataframe, text_features, new_feature_name, drop=False):
#     dataframe[new_feature_name] = ""
#     for text_col in text_features:
#         dataframe[new_feature_name] += " " + dataframe[text_col].astype(str).fillna("")
# #     dataframe['text'] = dataframe['title'].str.cat(dataframe['description'], sep=' ', na_rep='')
#     dataframe[new_feature_name] = dataframe[new_feature_name].apply(_preprocess_text)
#     if drop:
#         dataframe = dataframe.drop(text_features, axis=1)
#     return dataframe

# train = merge_text(train, text_features, 'text')
# train = train['text']
# docs = train['text'].map(lambda x: [t for t in x if t not in ru_stopwords]) 
# gc.collect()

# dictionary = corpora.Dictionary(docs)
# corpus = [dictionary.doc2bow(doc) for doc in docs]
# ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=20, passes=5, minimum_probability=0)

# ldamodel .show_topics(num_topics=10, num_words=20)

# text_lda = np.array([[y for (x, y) in ldamodel[corpus[i]]] for i in range(len(corpus))])
# np.save('./../input/text_lda.npy', text_lda.astype('float32'))
