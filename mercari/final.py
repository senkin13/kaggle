# Modified from Bojan LGBM Script
# Ensamble with Ridge and NN can reach my current LB position

import pyximport
pyximport.install()
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_extraction import stop_words
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, normalize
import lightgbm as lgb
import time
import gc
import math
import re
import string
from nltk.stem import WordNetLemmatizer
lemma  = WordNetLemmatizer()

class LemmaVectorizer(CountVectorizer):
    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        l_adder = self.lemma_adder()
        return lambda doc: self._word_ngrams(l_adder(tokenize(preprocess(self.decode(doc)))), stop_words)

    def lemma_adder(self):
        def lemmatizer(tokens):
            return list(set([lemma.lemmatize(w) for w in tokens] + tokens))
            
        return lemmatizer

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        def tokenizer(doc):
            token_pattern = re.compile(self.token_pattern)
            return token_pattern.findall(doc)
            
        return tokenizer 

def rmsle(y, h): 
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

def rmse(y, h): 
    return np.sqrt(np.square(h-y).mean())


def rmsle_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    
    y_pred = np.expm1(preds)
    y_tar  = np.expm1(y)
    
    score = rmsle(y_tar, y_pred) 
    return 'rmsle', score, False


def cleanName(text):
    try:
        textProc = text.lower()
        textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        regex = re.compile(u'[^A-Za-z0-9]+')
        textProc = regex.sub(" ", textProc)
        textProc = " ".join(textProc.split())
        
        return textProc
    except: 
        return "name error"

def getLastTwo(text):
    try:
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text)
        text = text.split()
        
        if len(text)==1:
            return text[0]
        
        text = text[-2]+" "+text[-1]

        return text
    except: 

        return " "

def split_cat(text):
    try: return text.split("/")
    except: return ("None", "None", "None")

NUM_BRANDS         = 5000
DESC_MAX_FEAT      = 500000
LGBM_NAME_MIN_DF   = 20
NAME_MIN_DF        = 2
DESC_MIN_DF        = 2

print("Reading in Data")
df     = pd.read_csv('../input/train.tsv', sep='\t')
dfTest = pd.read_csv('../input/test.tsv', sep='\t')
n_trains = df.shape[0]
y = np.log1p(df["price"].values)
submission: pd.DataFrame = dfTest[['test_id']]

print(df.shape)

print("Data Cleaning Training")

# Clean all the mess
df["brand_name"]        = df["brand_name"].fillna("unknown")
df["item_description"]  = df["item_description"].fillna("None")
df["name"]              = df["name"].fillna("None")
df["item_condition_id"] = df["item_condition_id"].fillna(0)
df["shipping"]          = df["shipping"].fillna(0)
df['category_name']     = df['category_name'].fillna("None/None/None")    
    
pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

df['general_cat'], df['subcat_1'], df['subcat_2'] = \
zip(*df['category_name'].apply(lambda x: split_cat(x)))

df['general_cat'].fillna(value='None', inplace=True)
df['subcat_1'].fillna(value='None', inplace=True)
df['subcat_2'].fillna(value='None', inplace=True)

df['general_cat']        = df['general_cat'].astype('category')
df['subcat_1']           = df['subcat_1'].astype('category')
df['subcat_2']           = df['subcat_2'].astype('category')     
df["item_condition_id"]  = df["item_condition_id"].astype("category")

df['name']               = df['name'].apply(lambda x: cleanName(x))
df["item_description"]   = df["item_description"].apply(lambda x: cleanName(x))
df['category_name']      = df['category_name'].apply(lambda x: cleanName(x))
df['object']             = df['name'].apply(lambda x: getLastTwo(x))

print("Data Cleaning Testing")
dfTest["brand_name"]        = dfTest["brand_name"].fillna("unknown")
dfTest["item_description"]  = dfTest["item_description"].fillna("None")
dfTest["name"]              = dfTest["name"].fillna("None")
dfTest["item_condition_id"] = dfTest["item_condition_id"].fillna(0)
dfTest["shipping"]          = dfTest["shipping"].fillna(0)
dfTest['category_name']     = dfTest['category_name'].fillna("None/None/None")    
    
dfTest.loc[~dfTest["brand_name"].isin(pop_brands), "brand_name"] = "Other"

dfTest['general_cat'], dfTest['subcat_1'], dfTest['subcat_2'] = \
zip(*dfTest['category_name'].apply(lambda x: split_cat(x)))

dfTest['general_cat'].fillna(value='None', inplace=True)
dfTest['subcat_1'].fillna(value='None', inplace=True)
dfTest['subcat_2'].fillna(value='None', inplace=True)

dfTest['general_cat']        = dfTest['general_cat'].astype('category')
dfTest['subcat_1']           = dfTest['subcat_1'].astype('category')
dfTest['subcat_2']           = dfTest['subcat_2'].astype('category')     
dfTest["item_condition_id"]  = dfTest["item_condition_id"].astype("category")

dfTest['name']               = dfTest['name'].apply(lambda x: cleanName(x))
dfTest["item_description"]   = dfTest["item_description"].apply(lambda x: cleanName(x))
dfTest['category_name']      = dfTest['category_name'].apply(lambda x: cleanName(x))
dfTest['object']             = dfTest['name'].apply(lambda x: getLastTwo(x))

print("Name Features 1")
count = LemmaVectorizer(min_df=LGBM_NAME_MIN_DF,
                        decode_error = 'replace',
                        ngram_range = (1,1),
                        token_pattern = r"(?u)\b\w+\b",
                        strip_accents = 'unicode')
X_name_1 = count.fit_transform(df["name"])
X_name_1_Test = count.transform(dfTest["name"])
del count    

print("category Features")
count = CountVectorizer(ngram_range = (1,1),
                        decode_error = 'replace',
                        token_pattern = r"(?u)\b\w+\b",
                        strip_accents = 'unicode')
X_category = count.fit_transform(df["category_name"])
X_category_Test = count.transform(dfTest["category_name"])
del count    

print('Object Features')
df['object'] = df['name'].apply(lambda x: getLastTwo(x))
count = CountVectorizer(min_df=LGBM_NAME_MIN_DF,decode_error = 'replace',)
X_object = count.fit_transform(df["object"])
X_object_Test = count.transform(dfTest["object"])
del count 

print("Brand Features")
count = LemmaVectorizer(ngram_range = (1,1),
                        decode_error = 'replace',
                        token_pattern = r"(?u)\b\w+\b",
                        strip_accents = 'unicode')
X_brand = count.fit_transform(df["brand_name"])
X_brand_Test = count.transform(dfTest["brand_name"])
del count
gc.collect()

print("Item Cond & Shipping Features")
ohe = OneHotEncoder(dtype=np.float32, handle_unknown='ignore')
X_dummies = ohe.fit_transform(np.array([df["item_condition_id"].tolist(),df["shipping"].tolist()]).T)
X_dummies_Test = ohe.transform(np.array([dfTest["item_condition_id"].tolist(),dfTest["shipping"].tolist()]).T)
gc.collect()

print("Description Features")
count_descp = CountVectorizer(max_features = DESC_MAX_FEAT,
                              decode_error = 'replace',
                              min_df=LGBM_NAME_MIN_DF,
                              ngram_range = (1,1),
                              token_pattern = r"(?u)\b\w+\b",
                              strip_accents = 'unicode')
X_descp = count_descp.fit_transform(df["item_description"])
X_descp_add = count_descp.transform(df["name"])
X_descp = X_descp + X_descp_add

X_descp_Test = count_descp.transform(dfTest["item_description"])
X_descp_add_Test = count_descp.transform(dfTest["name"])
X_descp_Test = X_descp_Test + X_descp_add_Test

del count_descp
del X_descp_add
del X_descp_add_Test
gc.collect()

del df
gc.collect()

print("X_name_1 {}".format(X_name_1.shape)) 
print("X_category {}".format(X_category.shape))    
print("X_object {}".format(X_object.shape))    
print("X_brand {}".format(X_brand.shape))    
print("X_dummies {}".format(X_dummies.shape))    
print("X_descp {}".format(X_descp.shape))    


print("Concatenate X_1")
X_1 = hstack((X_dummies,            
              X_descp,
              X_brand,              
              X_category,
              X_name_1,              
              X_object,              
              )).tocsr()
X_1_Test = hstack((X_dummies_Test,            
              X_descp_Test,
              X_brand_Test,              
              X_category_Test,
              X_name_1_Test,              
              X_object_Test,              
              )).tocsr()

del X_dummies            
del X_descp
del X_brand              
del X_category
del X_object
del X_name_1
del X_dummies_Test            
del X_descp_Test
del X_brand_Test              
del X_category_Test
del X_object_Test
del X_name_1_Test

gc.collect()

X_1.data = X_1.data.astype(np.float32)
X_1_Test.data = X_1_Test.data.astype(np.float32)

print("X_1 {}".format(X_1.shape)) 

np.random.seed(0)

filterTrain     = np.where((np.expm1(y) > 1) )
y     = y[filterTrain[0]]
X_1   = X_1[filterTrain[0]]  

print("Training LGBM")
d_train = lgb.Dataset(X_1, label=y)
params = {
    'max_bin':255,
    'min_data_in_leaf':1,
    'learning_rate': 0.3,
    'application': 'regression',
    'max_depth': 20,
    'num_leaves': 90,
    'verbosity': -1,
    'metric': 'RMSE',
    'data_random_seed': 1,
    'bagging_freq' : 0, 
    'bagging_fraction' : 0.5,
    'feature_fraction' : 1,
    'lambda_l1' : 2, 
    'lambda_l2' : 0,        
    'nthread': 8,
    'bin_construct_sample_cnt': 50000
}

model      = lgb.train(params, train_set=d_train, num_boost_round=1500,verbose_eval=100) 
del X_1
del y
del d_train
gc.collect()

predsL   = model.predict(X_1_Test)    
del X_1_Test
gc.collect()

## Ridge

import multiprocessing as mp
import pandas as pd
from time import time
from scipy.sparse import csr_matrix
import os
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
import re
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

INPUT_PATH = r'../input'


def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, start_time=time()):
        self.field = field
        self.start_time = start_time

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        print(f'[{time()-self.start_time}] select {self.field}')
        dt = dataframe[self.field].dtype
        if is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]


class DropColumnsByDf(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self

    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]


def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


def split_cat(text):
    try:
        cats = text.split("/")
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return 'other', 'other', 'other', 'other/other'


def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    print(f"Before empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    print(f"After empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()


def preprocess_regex(dataset, start_time=time()):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)
    print(f'[{time() - start_time}] Karats normalized.')

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)
    print(f'[{time() - start_time}] Units glued.')


def preprocess_pandas(train, test, start_time=time()):
    train = train[train.price > 0.0].reset_index(drop=True)
    print('Train shape without zero price: ', train.shape)

    nrow_train = train.shape[0]
    y_train = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])

    del train
    del test
    gc.collect()

    merge['has_category'] = (merge['category_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_category filled.')

    merge['category_name'] = merge['category_name'] \
        .fillna('other/other/other') \
        .str.lower() \
        .astype(str)
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'], merge['gen_subcat1'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    print(f'[{time() - start_time}] Split categories completed.')

    merge['has_brand'] = (merge['brand_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_brand filled.')

    merge['gencat_cond'] = merge['general_cat'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_1_cond'] = merge['subcat_1'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_2_cond'] = merge['subcat_2'].map(str) + '_' + merge['item_condition_id'].astype(str)
    print(f'[{time() - start_time}] Categories and item_condition_id concancenated.')

    merge['name'] = merge['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['brand_name'] = merge['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['item_description'] = merge['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace(to_replace='No description yet', value='')
    print(f'[{time() - start_time}] Missing filled.')

    preprocess_regex(merge, start_time)

    brands_filling(merge)
    print(f'[{time() - start_time}] Brand name filled.')

    merge['name'] = merge['name'] + ' ' + merge['brand_name']
    print(f'[{time() - start_time}] Name concancenated.')

    merge['item_description'] = merge['item_description'] \
                                + ' ' + merge['name'] \
                                + ' ' + merge['subcat_1'] \
                                + ' ' + merge['subcat_2'] \
                                + ' ' + merge['general_cat'] \
                                + ' ' + merge['brand_name']
    print(f'[{time() - start_time}] Item description concatenated.')

    merge.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)

    return merge, y_train, nrow_train


def intersect_drop_columns(train: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    v = valid.tocsc()
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_valid
    res = t[:, nnz_cols], v[:, nnz_cols]
    return res

mp.set_start_method('forkserver', True)

start_time = time()

train = pd.read_table(os.path.join(INPUT_PATH, 'train.tsv'),
                      engine='c',
                      dtype={'item_condition_id': 'category',
                             'shipping': 'category'}
                      )
test = pd.read_table(os.path.join(INPUT_PATH, 'test.tsv'),
                     engine='c',
                     dtype={'item_condition_id': 'category',
                            'shipping': 'category'}
                         )
print(f'[{time() - start_time}] Finished to load data')
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

submission: pd.DataFrame = test[['test_id']]

merge, y_train, nrow_train = preprocess_pandas(train, test, start_time)

del train
del test
gc.collect()

meta_params = {'name_ngram': (1, 2),
               'name_max_f': 75000,
               'name_min_df': 10,

               'category_ngram': (2, 3),
               'category_token': '.+',
               'category_min_df': 10,

               'brand_min_df': 10,

               'desc_ngram': (1, 3),
               'desc_max_f': 150000,
               'desc_max_df': 0.5,
               'desc_min_df': 10}

stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])

vectorizer = FeatureUnion([
        ('name', Pipeline([
            ('select', ItemSelector('name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 2),
                n_features=2 ** 27,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('category_name', Pipeline([
            ('select', ItemSelector('category_name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 1),
                token_pattern='.+',
                tokenizer=split_cat,
                n_features=2 ** 27,
                norm='l2',
                lowercase=False
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('brand_name', Pipeline([
            ('select', ItemSelector('brand_name', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('gencat_cond', Pipeline([
            ('select', ItemSelector('gencat_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_1_cond', Pipeline([
            ('select', ItemSelector('subcat_1_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_2_cond', Pipeline([
            ('select', ItemSelector('subcat_2_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('has_brand', Pipeline([
            ('select', ItemSelector('has_brand', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('shipping', Pipeline([
            ('select', ItemSelector('shipping', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('item_condition_id', Pipeline([
            ('select', ItemSelector('item_condition_id', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('item_description', Pipeline([
            ('select', ItemSelector('item_description', start_time=start_time)),
            ('hash', HashingVectorizer(
                ngram_range=(1, 3),
                n_features=2 ** 27,
                dtype=np.float32,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2)),
        ]))
], n_jobs=1)

sparse_merge = vectorizer.fit_transform(merge)
print(f'[{time() - start_time}] Merge vectorized')
print(sparse_merge.shape)

tfidf_transformer = TfidfTransformer()

X = tfidf_transformer.fit_transform(sparse_merge)
print(f'[{time() - start_time}] TF/IDF completed')

X_train = X[:nrow_train]
print(X_train.shape)

X_test = X[nrow_train:]
del merge
del sparse_merge
del vectorizer
del tfidf_transformer
gc.collect()

X_train, X_test = intersect_drop_columns(X_train, X_test, min_df=1)
print(f'[{time() - start_time}] Drop only in train or test cols: {X_train.shape[1]}')
gc.collect()

ridge = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)
ridge.fit(X_train, y_train)
print(f'[{time() - start_time}] Train Ridge completed. Iterations: {ridge.n_iter_}')

del nrow_train
del X
del X_train
del y_train
gc.collect()

predsR = ridge.predict(X_test)
print(f'[{time() - start_time}] Predict Ridge completed.')
del X_test
gc.collect()

## ensemble
preds = (predsL * 0.5 + predsR * 0.5)

submission.loc[:, 'price'] = np.expm1(preds)
submission.loc[submission['price'] < 0.0, 'price'] = 0.0
