%%time
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings 
warnings.filterwarnings('ignore')

################historical
df_hist = pd.read_csv("../input/historical_transactions.csv")
df_hist = df_hist.sort_values(by=['card_id', 'purchase_date'], ascending=True)
df_hist['authorized_flag'] = df_hist['authorized_flag'].map({'Y':1, 'N':0})
df_hist['category_1'] = df_hist['category_1'].map({'Y':1, 'N':0})
df_hist['category_3'] = df_hist['category_3'].map({'A':0, 'B':1, 'C':2})
# combi
df_hist['category_123'] = df_hist['category_1'].map(str) + df_hist['category_2'].map(str) + df_hist['category_3'].map(str)
# ohe
df_hist = pd.get_dummies(df_hist, columns=['category_2', 'category_3'])
# trim
df_hist['purchase_amount'] = df_hist['purchase_amount'].apply(lambda x: min(x, 135766.05564212))
# date
df_hist['purchase_date'] = pd.to_datetime(df_hist['purchase_date'])
df_hist['year'] = df_hist['purchase_date'].dt.year
df_hist['month'] = df_hist['purchase_date'].dt.month
df_hist['woy'] = df_hist['purchase_date'].dt.weekofyear
df_hist['doy'] = df_hist['purchase_date'].dt.dayofyear
df_hist['wday'] = df_hist['purchase_date'].dt.dayofweek
df_hist['weekend'] = (df_hist.purchase_date.dt.weekday >=5).astype(int)
df_hist['day'] = df_hist['purchase_date'].dt.day
df_hist['hour'] = df_hist['purchase_date'].dt.hour
df_hist['month_diff'] = ((datetime.date(2018, 5, 1)  - df_hist['purchase_date'].dt.date).dt.days)//30
df_hist['month_diff'] += df_hist['month_lag']
df_hist['day_diff'] = ((datetime.date(2018, 5, 1)  - df_hist['purchase_date'].dt.date).dt.days)
df_hist['pre_purchase_diff'] = df_hist[['card_id','purchase_date']].groupby(['card_id'])['purchase_date'].transform(lambda x: x.diff().shift(0).dt.days)
df_hist['next_purchase_diff'] = df_hist[['card_id','purchase_date']].groupby(['card_id'])['purchase_date'].transform(lambda x: x.diff().shift(-1).dt.days)
df_hist['refer_date'] = df_hist['year'].map(lambda x:0 if x==2017 else 12) + df_hist['month'] - df_hist['month_lag']
df_hist['pre_purchase_amount_diff'] = df_hist[['card_id','purchase_amount']].groupby(['card_id'])['purchase_amount'].transform(lambda x: x.diff().shift(0))
df_hist['next_purchase_amount_diff'] = df_hist[['card_id','purchase_amount']].groupby(['card_id'])['purchase_amount'].transform(lambda x: x.diff().shift(-1))
df_hist['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - df_hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_hist['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - df_hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_hist['Children_day_2017'] = (pd.to_datetime('2017-10-12') - df_hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_hist['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - df_hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_hist['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - df_hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_hist['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - df_hist['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_hist['installments'].replace(-1, np.nan,inplace=True)
df_hist['installments'].replace(999, np.nan,inplace=True)

df_hist['month_pay_installments'] = df_hist['purchase_amount'] / df_hist['installments']
df_hist['positive_weighted_purchase_amount'] = df_hist['purchase_amount'] / df_hist['refer_date']
df_hist['positive_weighted_installments'] = df_hist['month_pay_installments'] / df_hist['refer_date']
df_hist['negative_weighted_purchase_amount'] = df_hist['purchase_amount'] * df_hist['refer_date']
df_hist['negative_weighted_installments'] = df_hist['month_pay_installments'] * df_hist['refer_date']

df_hist['purchase_amount_bin'] = df_hist['purchase_amount'].map(lambda x:0 if x<0 else 1) 
df_hist['purchase_amount_positive'] = df_hist['purchase_amount'].map(lambda x:x if x>0 else 0) 
df_hist['purchase_amount_negative'] = df_hist['purchase_amount'].map(lambda x:x if x<0 else 0)
df_hist['date'] = df_hist['purchase_date'].dt.strftime('%Y%m%d')
df_hist['purchase_year_month'] = df_hist['year'].map(lambda x:0 if x==2011 
                                            else 12 if x==2012
                                            else 24 if x==2013
                                            else 36 if x==2014
                                            else 48 if x==2015
                                            else 60 if x==2016
                                            else 72 if x==2017
                                            else 84 if x==2018
                                            else x
                                            ) + df_hist['month'] 
df_hist['refer_year_month'] = df_hist['purchase_year_month'] - df_hist['month_lag']

################new
df_new = pd.read_csv("../input/new_merchant_transactions.csv")
df_new = df_new.sort_values(by=['card_id', 'purchase_date'], ascending=True)
df_new['authorized_flag'] = df_new['authorized_flag'].map({'Y':1, 'N':0})
df_new['category_1'] = df_new['category_1'].map({'Y':1, 'N':0})
df_new['category_3'] = df_new['category_3'].map({'A':0, 'B':1, 'C':2})
# combi
df_new['category_123'] = df_new['category_1'].map(str) + df_new['category_2'].map(str) + df_new['category_3'].map(str)
# ohe
df_new = pd.get_dummies(df_new, columns=['category_2', 'category_3'])
# trim
df_new['purchase_amount'] = df_new['purchase_amount'].apply(lambda x: min(x, 135766.05564212))
# date
df_new['purchase_date'] = pd.to_datetime(df_new['purchase_date'])
df_new['year'] = df_new['purchase_date'].dt.year
df_new['month'] = df_new['purchase_date'].dt.month
df_new['woy'] = df_new['purchase_date'].dt.weekofyear
df_new['doy'] = df_new['purchase_date'].dt.dayofyear
df_new['wday'] = df_new['purchase_date'].dt.dayofweek
df_new['weekend'] = (df_new.purchase_date.dt.weekday >=5).astype(int)
df_new['day'] = df_new['purchase_date'].dt.day
df_new['hour'] = df_new['purchase_date'].dt.hour
df_new['month_diff'] = ((datetime.date(2018, 5, 1)  - df_new['purchase_date'].dt.date).dt.days)//30
df_new['month_diff'] += df_new['month_lag']
df_new['day_diff'] = ((datetime.date(2018, 5, 1)  - df_new['purchase_date'].dt.date).dt.days)
df_new['pre_purchase_diff'] = df_new[['card_id','purchase_date']].groupby(['card_id'])['purchase_date'].transform(lambda x: x.diff().shift(0).dt.days)
df_new['next_purchase_diff'] = df_new[['card_id','purchase_date']].groupby(['card_id'])['purchase_date'].transform(lambda x: x.diff().shift(-1).dt.days)
df_new['refer_date'] = df_new['year'].map(lambda x:0 if x==2017 else 12) + df_new['month'] - df_new['month_lag']
df_new['pre_purchase_amount_diff'] = df_new[['card_id','purchase_amount']].groupby(['card_id'])['purchase_amount'].transform(lambda x: x.diff().shift(0))
df_new['next_purchase_amount_diff'] = df_new[['card_id','purchase_amount']].groupby(['card_id'])['purchase_amount'].transform(lambda x: x.diff().shift(-1))
df_new['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - df_new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_new['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - df_new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_new['Children_day_2017'] = (pd.to_datetime('2017-10-12') - df_new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_new['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - df_new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_new['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - df_new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_new['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - df_new['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
df_new['installments'].replace(-1, np.nan,inplace=True)
df_new['installments'].replace(999, np.nan,inplace=True)

df_new['month_pay_installments'] = df_new['purchase_amount'] / df_new['installments']
df_new['positive_weighted_purchase_amount'] = df_new['purchase_amount'] / df_new['refer_date']
df_new['positive_weighted_installments'] = df_new['month_pay_installments'] / df_new['refer_date']
df_new['negative_weighted_purchase_amount'] = df_new['purchase_amount'] * df_new['refer_date']
df_new['negative_weighted_installments'] = df_new['month_pay_installments'] * df_new['refer_date']

df_new['purchase_amount_bin'] = df_new['purchase_amount'].map(lambda x:0 if x<0 else 1) 
df_new['purchase_amount_positive'] = df_new['purchase_amount'].map(lambda x:x if x>0 else 0) 
df_new['purchase_amount_negative'] = df_new['purchase_amount'].map(lambda x:x if x<0 else 0)
df_new['date'] = df_new['purchase_date'].dt.strftime('%Y%m%d')

df_new['purchase_year_month'] = df_new['year'].map(lambda x:0 if x==2011 
                                            else 12 if x==2012
                                            else 24 if x==2013
                                            else 36 if x==2014
                                            else 48 if x==2015
                                            else 60 if x==2016
                                            else 72 if x==2017
                                            else 84 if x==2018
                                            else x
                                            ) + df_new['month'] 
df_new['refer_year_month'] = df_new['purchase_year_month'] - df_new['month_lag']

##############train test
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
df = pd.concat([train,test],axis=0).reset_index(drop=True)
del train,test
gc.collect()
# fillna
df['first_active_month'].fillna('2017-09',inplace=True)
# date
df['first_active_month'] = pd.to_datetime(df['first_active_month'])
df['first_year'] = df['first_active_month'].dt.year
df['first_month'] = df['first_active_month'].dt.month
df['elapsed_days'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
df['first_year_month'] = df['first_year'].map(lambda x:0 if x==2011 
                                            else 12 if x==2012
                                            else 24 if x==2013
                                            else 36 if x==2014
                                            else 48 if x==2015
                                            else 60 if x==2016
                                            else 72 if x==2017
                                            else 84 if x==2018
                                            else x
                                            ) + df['first_month'] 

################save to pickle
df.to_pickle('../feature/df.pkl')
df_hist.to_pickle('../feature/df_hist.pkl')
df_new.to_pickle('../feature/df_new.pkl')

%%time
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from scipy import sparse
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import NMF,LatentDirichletAllocation,TruncatedSVD
from gensim.sklearn_api.ldamodel import LdaTransformer
from gensim.models import LdaMulticore
from gensim import corpora
from gensim.models import Word2Vec
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
traintest = pd.concat([train,test],axis=0)[['card_id','target']]

def encode(df, col):
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())
    return df

for col in ['month','day','hour']:
    df_hist = encode(df_hist, col)
    df_new = encode(df_new, col)
    df_hist_new = encode(df_hist_new, col)
    
def mod(arr):
    return mode(arr)[0][0]  

def count_vector_feature(df, traintest, groupby, target):
    count_vec = CountVectorizer()
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    df_bag[target + '_list']=df_bag['list'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',',' '))
    
    df_bag = df_bag.merge(traintest,on=groupby,how='left')
    df_bag_train = df_bag[df_bag['target'].notnull()].reset_index(drop=True)
    df_bag_test = df_bag[df_bag['target'].isnull()].reset_index(drop=True)
    
    count_full_vector = count_vec.fit_transform(df_bag[target + '_list'])
    count_train_vector = count_vec.transform(df_bag_train[target + '_list'])
    count_test_vector = count_vec.transform(df_bag_test[target + '_list'])
    print ('count_full_vector:' + str(count_full_vector.shape))
    print ('count_train_vector:' + str(count_train_vector.shape))
    print ('count_test_vector:' + str(count_test_vector.shape))
    return count_train_vector,count_test_vector
    
def tfidf_vector_feature(df, traintest, groupby, target):
    tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=10000)
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    df_bag[target + '_list']=df_bag['list'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',',' '))
    
    df_bag = df_bag.merge(traintest,on=groupby,how='left')
    df_bag_train = df_bag[df_bag['target'].notnull()].reset_index(drop=True)
    df_bag_test = df_bag[df_bag['target'].isnull()].reset_index(drop=True)
    
    tfidf_full_vector = tfidf_vec.fit_transform(df_bag[target + '_list'])
    tfidf_train_vector = tfidf_vec.transform(df_bag_train[target + '_list'])
    tfidf_test_vector = tfidf_vec.transform(df_bag_test[target + '_list'])
    print ('tfidf_full_vector:' + str(tfidf_full_vector.shape))
    print ('tfidf_train_vector:' + str(tfidf_train_vector.shape))
    print ('tfidf_test_vector:' + str(tfidf_test_vector.shape))
    return tfidf_full_vector,tfidf_train_vector,tfidf_test_vector
  
def lda_feature(prefix,df, groupby, target,n_topic):
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag[target] = df_bag[target].astype(str)
    df_bag[target].fillna('NAN', inplace=True)    
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))})
    df_bag['sentence'] = df_bag['list'].apply(lambda x: list(map(str,x)))
    docs = df_bag['sentence'].tolist() 
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(text) for text in docs]
    lda = LdaMulticore(corpus, id2word=dictionary, num_topics=n_topic)
    docres = [dict(lda[doc_bow]) for doc_bow in corpus]
    df_lda = pd.DataFrame(docres,dtype=np.float16).fillna(0.001)
    df_lda.columns = ['lda_%s_%s_%d'%(prefix,target,x) for x in range(n_topic)]
    df_lda[groupby] = df_bag[groupby]
    print ('df_lda:' + str(df_lda.shape))
    return df_lda
  
def word2vec_feature(prefix, df, groupby, target,size):
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag[target] = df_bag[target].astype(str)
    df_bag[target].fillna('NAN', inplace=True)
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    doc_list = list(df_bag['list'].values)
    w2v = Word2Vec(doc_list, size=size, window=3, min_count=1, workers=32)
    vocab_keys = list(w2v.wv.vocab.keys())
    w2v_array = []
    for v in vocab_keys :
        w2v_array.append(list(w2v.wv[v]))
    df_w2v = pd.DataFrame()
    df_w2v['vocab_keys'] = vocab_keys    
    df_w2v = pd.concat([df_w2v, pd.DataFrame(w2v_array)], axis=1)
    df_w2v.columns = [target] + ['w2v_%s_%s_%d'%(prefix,target,x) for x in range(size)]
    print ('df_w2v:' + str(df_w2v.shape))
    return df_w2v

def svd_feature(prefix, df, traintest, groupby, target,n_comp):
    tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=None)
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    df_bag[target + '_list']=df_bag['list'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',',' '))
    
    df_bag = df_bag.merge(traintest,on=groupby,how='left')
    df_bag_train = df_bag[df_bag['target'].notnull()].reset_index(drop=True)
    df_bag_test = df_bag[df_bag['target'].isnull()].reset_index(drop=True)
    
    tfidf_full_vector = tfidf_vec.fit_transform(df_bag[target + '_list'])
    tfidf_train_vector = tfidf_vec.transform(df_bag_train[target + '_list'])
    tfidf_test_vector = tfidf_vec.transform(df_bag_test[target + '_list'])
    
    svd_vec = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_vec.fit(tfidf_full_vector)
    svd_train = pd.DataFrame(svd_vec.transform(tfidf_train_vector))
    svd_test = pd.DataFrame(svd_vec.transform(tfidf_test_vector))

    svd_train.columns = ['svd_%s_%s_%d'%(prefix,target,x) for x in range(n_comp)]
    svd_train[groupby] = df_bag_train[groupby]
    svd_test.columns = ['svd_%s_%s_%d'%(prefix,target,x) for x in range(n_comp)]
    svd_test[groupby] = df_bag_test[groupby]
    #df_svd = pd.concat([svd_train,svd_test],axis=0)
    print ('svd_train:' + str(svd_train.shape))
    print ('svd_test:' + str(svd_test.shape))
    return svd_train,svd_test



for col in ['date','merchant_category_id','merchant_id','city_id','state_id','subsector_id']:
    print ('===============',col,'===============')
    
    print ('w2v',col)
    new_w2v = word2vec_feature('category_3', df_new, 'card_id', col,5)
    hist_w2v = word2vec_feature('category_3', df_hist, 'card_id', col,5)
    hist_new_w2v = word2vec_feature('category_3', df_hist_new, 'card_id', col,5)  
    df_new[col] = df_new[col].astype(str)
    df_new = df_new.merge(new_w2v,on=col,how='left')
    df_hist[col] = df_hist[col].astype(str)
    df_hist = df_hist.merge(hist_w2v,on=col,how='left')
    df_hist_new[col] = df_hist_new[col].astype(str)
    df_hist_new = df_hist_new.merge(hist_new_w2v,on=col,how='left')
    

    print ('svd',col)
    hist_train_svd,hist_test_svd = svd_feature('hist',df_hist,traintest, 'card_id', col,5)
    new_train_svd,new_test_svd = svd_feature('new',df_new,traintest, 'card_id', col,5)
    hist_new_train_svd,hist_new_test_svd = svd_feature('hist_new',df_hist_new,traintest, 'card_id', col,5)

%%time
############################################################################################################################
target = 'purchase_amount'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

    for col in ['authorized_flag',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_card['hist_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].mean())
            df_hist_card['hist_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].median())
            df_hist_card['hist_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].max())
            df_hist_card['hist_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].min())
            df_hist_card['hist_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].std())
            df_hist_card['hist_card_var_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].var())
            df_hist_card['hist_card_skew_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].skew())

    for col in ['month_lag',]:
        for d in df_new[col].unique():
            print (col,d,target)
            df_new_card['new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].sum())        
            df_new_card['new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].mean())
            df_new_card['new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].median())
            df_new_card['new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].max())
            df_new_card['new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].min())
            df_new_card['new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].std())
            df_new_card['new_card_var_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].var())
            df_new_card['new_card_skew_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].skew())

 
    for col in ['category_1',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_card['hist_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].mean())
            df_hist_card['hist_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].median())
            df_hist_card['hist_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].max())
            df_hist_card['hist_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].min())
            df_hist_card['hist_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].std())
            df_hist_card['hist_card_var_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].var())
            df_hist_card['hist_card_skew_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].skew())


    for col1 in ['category_1']: 
        for d in df_new[col1].unique():
            for col2 in ['month_lag']: 
                for c in df_new[col2].unique():
                    print (col1,d,col2,c,target)
                    df_new_card['new_sum_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].sum())        
                    df_new_card['new_mean_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].mean())        
                    df_new_card['new_median_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].median())        
                    df_new_card['new_max_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].max())        
                    df_new_card['new_min_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].min())        
                    df_new_card['new_std_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].std())        
                    df_new_card['new_var_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].var())        
                    df_new_card['new_skew_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].skew())        


df_hist_card['hist_card_sum_purchase_amount_1months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_2months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_3months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_4months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-11-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_5months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-10-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_6months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-09-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_7months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-08-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_8months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-07-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_9months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-06-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_10months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-05-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_11months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-04-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_12months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-03-01'].groupby(['card_id'])[target].sum())

###########################################################################################################################        
target = 'positive_weighted_purchase_amount'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')


    for col in ['month_lag',]:
        for d in df_new[col].unique():
            print (col,d,target)
            df_new_card['new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].sum())        
            df_new_card['new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].mean())
            df_new_card['new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].median())
            df_new_card['new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].max())
            df_new_card['new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].min())
            df_new_card['new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].std())
            df_new_card['new_card_var_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].var())
            df_new_card['new_card_skew_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].skew())
  
    for col1 in ['category_1']: 
        for d in df_new[col1].unique():
            for col2 in ['month_lag']: 
                for c in df_new[col2].unique():
                    print (col1,d,col2,c,target)
                    df_new_card['new_sum_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].sum())        
                    df_new_card['new_mean_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].mean())        
                    df_new_card['new_median_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].median())        
                    df_new_card['new_max_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].max())        
                    df_new_card['new_min_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].min())        
                    df_new_card['new_std_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].std())        
                    df_new_card['new_var_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].var())        
                    df_new_card['new_skew_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].skew())        
    
############################################################################################################################        
target = 'purchase_amount_bin'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_new_card')
 
    for col in ['authorized_flag',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_card['hist_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].mean())

############################################################################################################################        
target = 'month_pay_installments'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')


############################################################################################################################        
target = 'installments'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card')   

    for d in df_hist[target].unique():
        df_hist_card[str(target) + '_hist_' + str(d)] = df_hist_card['card_id'].map(df_hist[df_hist[target]==d].groupby(['card_id'])[target].count())
        df_hist_card[str(target) + '_hist_' + str(d)] = df_hist_card[str(target) + '_hist_' + str(d)].fillna(0)
        print (str(d)+' df_hist_card')    
############################################################################################################################        

############################################################################################################################        
target = 'refer_date'
for col in [target]:
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_new_card')
############################################################################################################################        
target = 'authorized_flag'
for col in [target]:
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_new_card')

############################################################################################################################        
target = 'category_1'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_new_card')    
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_new_card')
    
for d in df_hist_new[target].unique():
    df_hist_new_card[str(target) + '_hist_new_' + str(d)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[target]==d].groupby(['card_id'])[target].count())
    df_hist_new_card[str(target) + '_hist_new_' + str(d)] = df_hist_new_card[str(target) + '_hist_new_' + str(d)].fillna(0)
    print (str(d)+' df_hist_new_card')    
############################################################################################################################        
target = 'category_3'#'category_3_2.0'#'category_3_1.0'#,'category_3_0.0'#'category_2_5.0'#,'category_2_4.0'#'category_2_3.0'#,'category_2_2.0' #'category_2_1.0'#, , , '  
for col in ['category_3_2.0','category_3_1.0','category_3_0.0']:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_new_card')    
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_new_card')
# ############################################################################################################################        
target = 'purchase_year_month'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

############################################################################################################################        
target = 'month_lag'
for col in [target]:
  
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card')  

for d in df_hist[target].unique():
    df_hist_card[str(target) + '_hist_' + str(d)] = df_hist_card['card_id'].map(df_hist[df_hist[target]==d].groupby(['card_id'])[target].count())
    df_hist_card[str(target) + '_hist_' + str(d)] = df_hist_card[str(target) + '_hist_' + str(d)].fillna(0)
    print (str(d)+' df_hist_card')
 
############################################################################################################################        
target = 'day_diff'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card') 

############################################################################################################################        
target = 'pre_purchase_diff'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')  

############################################################################################################################        
target = 'purchase_date'
for col in [target]:
    df_new_card['new_card_purchase_date_max'] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].max())
    df_new_card['new_card_purchase_date_min'] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].min())
    df_new_card['new_card_minmax_diff_purchase_date'] = (df_new_card['new_card_purchase_date_max'] - df_new_card['new_card_purchase_date_min']).map(lambda x:x.days)
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_purchase_date_max'] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].max())
    df_hist_card['hist_card_purchase_date_min'] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].min())
    df_hist_card['hist_card_minmax_diff_purchase_date'] = (df_hist_card['hist_card_purchase_date_max'] - df_hist_card['hist_card_purchase_date_min']).map(lambda x:x.days)
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_purchase_date_max'] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].max())
    df_hist_new_card['hist_new_card_purchase_date_min'] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].min())
    df_hist_new_card['hist_new_card_minmax_diff_purchase_date'] = (df_hist_new_card['hist_new_card_purchase_date_max'] - df_hist_new_card['hist_new_card_purchase_date_min']).map(lambda x:x.days)
    print (str(col)+' df_hist_new_card')   
############################################################################################################################        
target = 'card_id'
for col in [target]:
    df_new_card['new_card_count'] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].count())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_count'] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].count())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_count'] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].count())    
    print (str(col)+' df_hist_new_card')   

df_hist_card['hist_card_count_1months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_2months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_3months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_4months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-11-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_5months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-10-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_6months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-09-01'].groupby(['card_id'])[target].count())


############################################################################################################################
target = 'purchase_amount'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

    for col in ['authorized_flag',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_card['hist_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].mean())
            df_hist_card['hist_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].median())
            df_hist_card['hist_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].max())
            df_hist_card['hist_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].min())
            df_hist_card['hist_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].std())
            df_hist_card['hist_card_var_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].var())
            df_hist_card['hist_card_skew_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].skew())

    for col in ['month_lag',]:
        for d in df_new[col].unique():
            print (col,d,target)
            df_new_card['new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].sum())        
            df_new_card['new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].mean())
            df_new_card['new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].median())
            df_new_card['new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].max())
            df_new_card['new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].min())
            df_new_card['new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].std())
            df_new_card['new_card_var_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].var())
            df_new_card['new_card_skew_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].skew())

 
    for col in ['category_1',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_card['hist_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].mean())
            df_hist_card['hist_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].median())
            df_hist_card['hist_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].max())
            df_hist_card['hist_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].min())
            df_hist_card['hist_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].std())
            df_hist_card['hist_card_var_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].var())
            df_hist_card['hist_card_skew_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].skew())


    for col1 in ['category_1']: 
        for d in df_new[col1].unique():
            for col2 in ['month_lag']: 
                for c in df_new[col2].unique():
                    print (col1,d,col2,c,target)
                    df_new_card['new_sum_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].sum())        
                    df_new_card['new_mean_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].mean())        
                    df_new_card['new_median_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].median())        
                    df_new_card['new_max_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].max())        
                    df_new_card['new_min_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].min())        
                    df_new_card['new_std_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].std())        
                    df_new_card['new_var_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].var())        
                    df_new_card['new_skew_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].skew())        


df_hist_card['hist_card_sum_purchase_amount_1months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_2months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_3months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_4months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-11-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_5months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-10-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_6months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-09-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_7months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-08-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_8months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-07-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_9months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-06-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_10months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-05-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_11months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-04-01'].groupby(['card_id'])[target].sum())
df_hist_card['hist_card_sum_purchase_amount_12months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-03-01'].groupby(['card_id'])[target].sum())

###########################################################################################################################        
target = 'positive_weighted_purchase_amount'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')


    for col in ['month_lag',]:
        for d in df_new[col].unique():
            print (col,d,target)
            df_new_card['new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].sum())        
            df_new_card['new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].mean())
            df_new_card['new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].median())
            df_new_card['new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].max())
            df_new_card['new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].min())
            df_new_card['new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].std())
            df_new_card['new_card_var_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].var())
            df_new_card['new_card_skew_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].skew())
  
    for col1 in ['category_1']: 
        for d in df_new[col1].unique():
            for col2 in ['month_lag']: 
                for c in df_new[col2].unique():
                    print (col1,d,col2,c,target)
                    df_new_card['new_sum_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].sum())        
                    df_new_card['new_mean_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].mean())        
                    df_new_card['new_median_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].median())        
                    df_new_card['new_max_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].max())        
                    df_new_card['new_min_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].min())        
                    df_new_card['new_std_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].std())        
                    df_new_card['new_var_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].var())        
                    df_new_card['new_skew_' + str(col1) + '_' + str(d) + '_' + str(col2) + '_' + str(c) + str(target)] = df_new_card['card_id'].map(df_new[(df_new[col1]==d) & (df_new[col2]==c)].groupby(['card_id'])[target].skew())        
    
############################################################################################################################        
target = 'month_pay_installments'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')


############################################################################################################################        
target = 'installments'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card')   

    for d in df_hist[target].unique():
        df_hist_card[str(target) + '_hist_' + str(d)] = df_hist_card['card_id'].map(df_hist[df_hist[target]==d].groupby(['card_id'])[target].count())
        df_hist_card[str(target) + '_hist_' + str(d)] = df_hist_card[str(target) + '_hist_' + str(d)].fillna(0)
        print (str(d)+' df_hist_card')    
############################################################################################################################        

############################################################################################################################        
target = 'refer_date'
for col in [target]:
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_new_card')
############################################################################################################################        
target = 'authorized_flag'
for col in [target]:
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_new_card')

############################################################################################################################        
target = 'category_1'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_new_card')    
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_new_card')
    
for d in df_hist_new[target].unique():
    df_hist_new_card[str(target) + '_hist_new_' + str(d)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[target]==d].groupby(['card_id'])[target].count())
    df_hist_new_card[str(target) + '_hist_new_' + str(d)] = df_hist_new_card[str(target) + '_hist_new_' + str(d)].fillna(0)
    print (str(d)+' df_hist_new_card')    
############################################################################################################################        
target = 'category_3'#'category_3_2.0'#'category_3_1.0'#,'category_3_0.0'#'category_2_5.0'#,'category_2_4.0'#'category_2_3.0'#,'category_2_2.0' #'category_2_1.0'#, , , '  
for col in ['category_3_2.0','category_3_1.0','category_3_0.0']:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_new_card')    
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_hist_new_card')
# ############################################################################################################################        
target = 'purchase_year_month'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

############################################################################################################################        
target = 'month_lag'
for col in [target]:
  
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card')  

for d in df_hist[target].unique():
    df_hist_card[str(target) + '_hist_' + str(d)] = df_hist_card['card_id'].map(df_hist[df_hist[target]==d].groupby(['card_id'])[target].count())
    df_hist_card[str(target) + '_hist_' + str(d)] = df_hist_card[str(target) + '_hist_' + str(d)].fillna(0)
    print (str(d)+' df_hist_card')
 
############################################################################################################################        
target = 'day_diff'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')

    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card') 

############################################################################################################################        
target = 'pre_purchase_diff'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    df_new_card['new_card_var_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].var()) 
    df_new_card['new_card_skew_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].skew())         
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    df_hist_card['hist_card_var_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].var())
    df_hist_card['hist_card_skew_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].skew())            
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    df_hist_new_card['hist_new_card_var_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].var())
    df_hist_new_card['hist_new_card_skew_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].skew())     
    print (str(col)+' df_hist_new_card')  

############################################################################################################################        
target = 'purchase_date'
for col in [target]:
    df_new_card['new_card_purchase_date_max'] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].max())
    df_new_card['new_card_purchase_date_min'] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].min())
    df_new_card['new_card_minmax_diff_purchase_date'] = (df_new_card['new_card_purchase_date_max'] - df_new_card['new_card_purchase_date_min']).map(lambda x:x.days)
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_purchase_date_max'] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].max())
    df_hist_card['hist_card_purchase_date_min'] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].min())
    df_hist_card['hist_card_minmax_diff_purchase_date'] = (df_hist_card['hist_card_purchase_date_max'] - df_hist_card['hist_card_purchase_date_min']).map(lambda x:x.days)
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_purchase_date_max'] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].max())
    df_hist_new_card['hist_new_card_purchase_date_min'] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].min())
    df_hist_new_card['hist_new_card_minmax_diff_purchase_date'] = (df_hist_new_card['hist_new_card_purchase_date_max'] - df_hist_new_card['hist_new_card_purchase_date_min']).map(lambda x:x.days)
    print (str(col)+' df_hist_new_card')   
############################################################################################################################        
target = 'card_id'
for col in [target]:
    df_new_card['new_card_count'] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].count())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_count'] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].count())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_count'] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].count())    
    print (str(col)+' df_hist_new_card')   

df_hist_card['hist_card_count_1months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_2months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_3months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_4months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-11-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_5months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-10-01'].groupby(['card_id'])[target].count())
df_hist_card['hist_card_count_6months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-09-01'].groupby(['card_id'])[target].count())

############################################################################################################################        
# lable encoder 
def lbl_enc(df):
    for c in ['merchant_id']:
        if df[c].dtype == 'object':
            lbl = LabelEncoder()
            df[c] = lbl.fit_transform(df[c].astype(str))
    return df

df_hist = lbl_enc(df_hist)
df_new = lbl_enc(df_new)
df_hist_new = lbl_enc(df_hist_new)

target = 'merchant_id'
for col in [target]:
    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card')     


    print (str(col)+str(' mode'))
    df_hist_card['hist_card_mode_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].agg(mod))
    df_new_card['new_card_mode_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].agg(mod))
    df_hist_new_card['hist_new_card_mode_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].agg(mod))

############################################################################################################################        
target = 'merchant_category_id'
for col in [target]:
    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card') 

    print (str(col)+str(' mode'))
    df_hist_card['hist_card_mode_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].agg(mod))
    df_new_card['new_card_mode_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].agg(mod))
    df_hist_new_card['hist_new_card_mode_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].agg(mod))

############################################################################################################################        
target = 'city_id'
for col in [target]:
    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card')       

    print (str(col)+str(' mode'))
    df_hist_card['hist_card_mode_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].agg(mod))
    df_new_card['new_card_mode_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].agg(mod))
    df_hist_new_card['hist_new_card_mode_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].agg(mod))

############################################################################################################################        
target = 'state_id'
for col in [target]:
    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card')   

    print (str(col)+str(' mode'))
    df_hist_card['hist_card_mode_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].agg(mod))
    df_new_card['new_card_mode_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].agg(mod))
    df_hist_new_card['hist_new_card_mode_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].agg(mod))

############################################################################################################################        
target = 'subsector_id'
for col in [target]:
    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card')   

    for d in df_hist['subsector_id'].unique():
        df_hist_card['hist_subsector_id_' + str(d)] = df_hist_card['card_id'].map(df_hist[df_hist['subsector_id']==d].groupby(['card_id'])['subsector_id'].count())
        df_hist_card['hist_subsector_id_' + str(d)] = df_hist_card['hist_subsector_id_' + str(d)].fillna(0)
    print ('subsector_id Done') 

    print (str(col)+str(' mode'))
    df_hist_card['hist_card_mode_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].agg(mod))
    df_new_card['new_card_mode_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].agg(mod))
    df_hist_new_card['hist_new_card_mode_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].agg(mod))

    for col in ['authorized_flag',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
############################################################################################################################        
target = 'month'
for col in [target]:
    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    print (str(col)+' df_hist_new_card')   

############################################################################################################################        
target = 'year'
for col in [target]:
    df_new_card['new_card_nunique_' + str(target)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[target].nunique())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_nunique_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].nunique())
    df_hist_card['hist_card_mean_' + str(target)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[target].mean())
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_nunique_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].nunique())    
    df_hist_new_card['hist_new_card_mean_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[target].mean())    
    print (str(col)+' df_hist_new_card')   
    
    

############################################################################################################################
target = 'purchase_amount_new'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    print (str(col)+' df_hist_new_card')    

    for col in ['authorized_flag',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_card['hist_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].mean())
            df_hist_card['hist_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].median())
            df_hist_card['hist_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].max())
            df_hist_card['hist_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].min())
            df_hist_card['hist_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].std())

        for d in [1]:
            print (col,d,target)
            df_hist_new_card['hist_new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_new_card['hist_new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].mean())
            df_hist_new_card['hist_new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].median())
            df_hist_new_card['hist_new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].max())
            df_hist_new_card['hist_new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].min())
            df_hist_new_card['hist_new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].std())            

    for col in ['category_1',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_card['hist_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].mean())
            df_hist_card['hist_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].median())
            df_hist_card['hist_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].max())
            df_hist_card['hist_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].min())
            df_hist_card['hist_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].std())
    
    for col in ['category_1',]:
        for d in df_new[col].unique():
            print (col,d,target)
            df_new_card['new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].sum())        
            df_new_card['new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].mean())
            df_new_card['new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].median())
            df_new_card['new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].max())
            df_new_card['new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].min())
            df_new_card['new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].std())
 
    for col in ['category_1',]:
        for d in df_hist_new[col].unique():
            print (col,d,target)
            df_hist_new_card['hist_new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_new_card['hist_new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].mean())
            df_hist_new_card['hist_new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].median())
            df_hist_new_card['hist_new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].max())
            df_hist_new_card['hist_new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].min())
            df_hist_new_card['hist_new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].std())

 
    for col in ['month_lag',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_card['hist_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].mean())
            df_hist_card['hist_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].median())
            df_hist_card['hist_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].max())
            df_hist_card['hist_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].min())
            df_hist_card['hist_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].std())
    
    for col in ['month_lag',]:
        for d in df_new[col].unique():
            print (col,d,target)
            df_new_card['new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].sum())        
            df_new_card['new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].mean())
            df_new_card['new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].median())
            df_new_card['new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].max())
            df_new_card['new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].min())
            df_new_card['new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].std())
 

    for col in ['installments',]:
        for d in df_hist[col].unique():
            print (col,d,target)
            df_hist_card['hist_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_card['hist_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].mean())
            df_hist_card['hist_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].median())
            df_hist_card['hist_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].max())
            df_hist_card['hist_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].min())
            df_hist_card['hist_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_card['card_id'].map(df_hist[df_hist[col]==d].groupby(['card_id'])[target].std())
    
    for col in ['installments',]:
        for d in df_new[col].unique():
            print (col,d,target)
            df_new_card['new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].sum())        
            df_new_card['new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].mean())
            df_new_card['new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].median())
            df_new_card['new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].max())
            df_new_card['new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].min())
            df_new_card['new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_new_card['card_id'].map(df_new[df_new[col]==d].groupby(['card_id'])[target].std())
 
    for col in ['installments',]:
        for d in df_hist_new[col].unique():
            print (col,d,target)
            df_hist_new_card['hist_new_card_sum_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].sum())        
            df_hist_new_card['hist_new_card_mean_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].mean())
            df_hist_new_card['hist_new_card_median_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].median())
            df_hist_new_card['hist_new_card_max_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].max())
            df_hist_new_card['hist_new_card_min_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].min())
            df_hist_new_card['hist_new_card_std_' + str(col) + '_' + str(d) + '_' + str(target)] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new[col]==d].groupby(['card_id'])[target].std())
 

    df_new_card['new_card_sum_' + str(col) + '_1months'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-04-01'].groupby(['card_id'])[col].sum())
    df_new_card['new_card_sum_' + str(col) + '_halfmonths'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-04-16'].groupby(['card_id'])[col].sum())
    df_new_card['new_card_sum_' + str(col) + '_onehalfmonths'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-03-16'].groupby(['card_id'])[col].sum())
    df_new_card['new_card_sum_' + str(col) + '_onehalfquartermonths'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-03-08'].groupby(['card_id'])[col].sum())

    df_new_card['new_card_mean_' + str(col) + '_1months'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-04-01'].groupby(['card_id'])[col].mean())
    df_new_card['new_card_mean_' + str(col) + '_halfmonths'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-04-16'].groupby(['card_id'])[col].mean())
    df_new_card['new_card_mean_' + str(col) + '_onehalfmonths'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-03-16'].groupby(['card_id'])[col].mean())
    df_new_card['new_card_mean_' + str(col) + '_onehalfquartermonths'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-03-08'].groupby(['card_id'])[col].mean())

    df_new_card['new_card_median_' + str(col) + '_1months'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-04-01'].groupby(['card_id'])[col].median())
    df_new_card['new_card_median_' + str(col) + '_halfmonths'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-04-16'].groupby(['card_id'])[col].median())
    df_new_card['new_card_median_' + str(col) + '_onehalfmonths'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-03-16'].groupby(['card_id'])[col].median())
    df_new_card['new_card_median_' + str(col) + '_onehalfquartermonths'] = df_new_card['card_id'].map(df_new[df_new['purchase_date'] >= '2018-03-08'].groupby(['card_id'])[col].median())
    print ('df_new_card_'+str(col))
    df_hist_card['hist_card_sum_' + str(col) + '_3months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[col].sum())    
    df_hist_card['hist_card_sum_' + str(col) + '_2months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[col].sum())    
    df_hist_card['hist_card_sum_' + str(col) + '_1months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[col].sum())
      
    df_hist_card['hist_card_mean_' + str(col) + '_3months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[col].mean())    
    df_hist_card['hist_card_mean_' + str(col) + '_2months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[col].mean())    
    df_hist_card['hist_card_mean_' + str(col) + '_1months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[col].mean())

    df_hist_card['hist_card_median_' + str(col) + '_3months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[col].median())    
    df_hist_card['hist_card_median_' + str(col) + '_2months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[col].median())    
    df_hist_card['hist_card_median_' + str(col) + '_1months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[col].median())
    
    df_hist_card['hist_card_max_' + str(col) + '_3months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[col].max())    
    df_hist_card['hist_card_max_' + str(col) + '_2months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[col].max())    
    df_hist_card['hist_card_max_' + str(col) + '_1months'] = df_hist_card['card_id'].map(df_hist[df_hist['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[col].max())
    print ('df_hist_card_'+str(col))
    df_hist_new_card['hist_new_card_sum_' + str(col) + '_3months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_sum_' + str(col) + '_4months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_sum_' + str(col) + '_5months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_sum_' + str(col) + '_6months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-11-01'].groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_sum_' + str(col) + '_7months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-10-01'].groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_sum_' + str(col) + '_8months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-09-01'].groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_sum_' + str(col) + '_9months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-08-01'].groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_sum_' + str(col) + '_10months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-07-01'].groupby(['card_id'])[col].sum())

    df_hist_new_card['hist_new_card_max_' + str(col) + '_3months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_max_' + str(col) + '_4months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_max_' + str(col) + '_5months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_max_' + str(col) + '_6months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-11-01'].groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_max_' + str(col) + '_7months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-10-01'].groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_max_' + str(col) + '_8months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-09-01'].groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_max_' + str(col) + '_9months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-08-01'].groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_max_' + str(col) + '_10months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-07-01'].groupby(['card_id'])[col].max())

    df_hist_new_card['hist_new_card_min_' + str(col) + '_3months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2018-02-01'].groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_min_' + str(col) + '_4months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2018-01-01'].groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_min_' + str(col) + '_5months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-12-01'].groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_min_' + str(col) + '_6months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-11-01'].groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_min_' + str(col) + '_7months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-10-01'].groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_min_' + str(col) + '_8months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-09-01'].groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_min_' + str(col) + '_9months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-08-01'].groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_min_' + str(col) + '_10months'] = df_hist_new_card['card_id'].map(df_hist_new[df_hist_new['purchase_date'] >= '2017-07-01'].groupby(['card_id'])[col].min())
    print ('df_hist_new_card_'+str(col))
   
 

df = df.merge(df_new_card,on='card_id',how='left')  
df = df.merge(df_hist_card,on='card_id',how='left') 
df = df.merge(df_hist_new_card,on='card_id',how='left') 
############################################################################################################################
target = 'next_purchase_amount_new_diff'
for col in [target]:
    df_new_card['new_card_sum_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].sum())
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_median_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].median())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_sum_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].sum())
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_median_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].median())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_sum_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].sum())
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_median_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].median())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    print (str(col)+' df_hist_new_card') 

df = df.merge(df_new_card,on='card_id',how='left')  
df = df.merge(df_hist_card,on='card_id',how='left') 
df = df.merge(df_hist_new_card,on='card_id',how='left') 
############################################################################################################################

target = 'w2v'
for col in ['w2v_category_3_category_123_0',
'w2v_category_3_category_123_1',
'w2v_category_3_category_123_2',
'w2v_category_3_category_123_3',
'w2v_category_3_category_123_4',]:
    df_new_card['new_card_mean_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].mean())
    df_new_card['new_card_max_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].max())
    df_new_card['new_card_min_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].min())
    df_new_card['new_card_std_' + str(col)] = df_new_card['card_id'].map(df_new.groupby(['card_id'])[col].std()) 
    print (str(col)+' df_new_card')
    df_hist_card['hist_card_mean_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].mean())
    df_hist_card['hist_card_max_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].max())
    df_hist_card['hist_card_min_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].min())
    df_hist_card['hist_card_std_' + str(col)] = df_hist_card['card_id'].map(df_hist.groupby(['card_id'])[col].std())  
    print (str(col)+' df_hist_card')
    df_hist_new_card['hist_new_card_mean_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].mean())
    df_hist_new_card['hist_new_card_max_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].max())
    df_hist_new_card['hist_new_card_min_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].min())
    df_hist_new_card['hist_new_card_std_' + str(col)] = df_hist_new_card['card_id'].map(df_hist_new.groupby(['card_id'])[col].std())     
    print (str(col)+' df_hist_new_card')
df = df.merge(df_new_card,on='card_id',how='left')  
df = df.merge(df_hist_card,on='card_id',how='left') 
df = df.merge(df_hist_new_card,on='card_id',how='left') 

############################################################################################################################        
target = 'svd'
hist = pd.concat([hist_train_svd,hist_test_svd],axis=0)
new = pd.concat([new_train_svd,new_test_svd],axis=0)
hist_new = pd.concat([hist_new_train_svd,hist_new_test_svd],axis=0)
df = df.merge(hist,on='card_id',how='left')  
df = df.merge(new,on='card_id',how='left') 
df = df.merge(hist_new,on='card_id',how='left') 
############################################################################################################################        


for i in df.columns.values:
    if i not in raw_columns:
        print (i)
        df[i].to_pickle('../feature/' + str(target) + '/' + str(i))       
############################################################################################################################        

%%time
import os
import glob
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings 
warnings.filterwarnings('ignore')

df = pd.read_pickle('../feature/df.pkl')

for target in ['w2v','w2v_cardid','svd','purchase_amount_new','purchase_amount','positive_weighted_purchase_amount','month_pay_installments',
               'installments','refer_date','authorized_flag','category_1','purchase_year_month','month_lag','day_diff',
               'pre_purchase_diff','purchase_date','card_id','merchant_id','merchant_category_id','subsector_id','city_id',
               'state_id','month','refer_year_month',]:
    for fn in glob.glob('../feature/' + str(target) + '/*_*'):
        tmp = pd.read_pickle(fn)
        df[os.path.basename(fn)] = tmp
        del tmp
        gc.collect()
        print (os.path.basename(fn))  
    
######## interaction ##########
df['hist_card_minmax_purchase_year_month'] = df['hist_card_max_purchase_year_month'] / df['hist_card_min_purchase_year_month']
df['new_card_minmax_purchase_year_month'] = df['new_card_max_purchase_year_month'] / df['new_card_min_purchase_year_month']
df['hist_new_card_minmax_purchase_year_month'] = df['hist_new_card_max_purchase_year_month'] / df['hist_new_card_min_purchase_year_month']

df['new_card_purchase_date_average']  = df['new_card_minmax_diff_purchase_date'] / (df['new_card_count'] + 1)
df['hist_card_purchase_date_average']  = df['hist_card_minmax_diff_purchase_date'] / (df['hist_card_count'] + 1)

df['hist_card_first_year_month_diff'] = df['hist_card_min_purchase_year_month'] - df['first_year_month']

df['hist_card_last_woy'] = df['hist_card_purchase_date_max'].dt.weekofyear
df['hist_card_last_doy'] = df['hist_card_purchase_date_max'].dt.dayofyear
df['hist_card_last_day'] = df['hist_card_purchase_date_max'].dt.day


df['new_card_last_woy'] = df['new_card_purchase_date_max'].dt.weekofyear
df['new_card_last_doy'] = df['new_card_purchase_date_max'].dt.dayofyear
df['new_card_last_day'] = df['new_card_purchase_date_max'].dt.day


for f in [
          'new_card_purchase_date_max',#'new_card_purchase_date_min','hist_card_purchase_date_min',
          'hist_card_purchase_date_max',]:
    df[f +'_int'] = df[f].astype(np.int64) * 1e-9   
    
df['hist_new_purchase_date_max_ratio'] = df['new_card_purchase_date_max_int'] / df['hist_card_purchase_date_max_int']

df['hist_elapsed_time_max'] = (datetime.date(2018, 5, 1) - df['hist_card_purchase_date_max'].dt.date).dt.days 
df['new_elapsed_time_max'] = (datetime.date(2018, 5, 1) - df['new_card_purchase_date_max'].dt.date).dt.days  
df['hist_new_elapsed_time_max'] = (datetime.date(2018, 5, 1) - df['hist_new_card_purchase_date_max'].dt.date).dt.days  


print ('df shape:' + str(df.shape))

%%time
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings 
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
from scipy import sparse
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.feature_selection import SelectFromModel, VarianceThreshold,RFE

def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

drop_features=['card_id', 'target', 'first_active_month', 'first_year_month',
'new_card_purchase_date_max', 'new_card_purchase_date_min','hist_card_purchase_date_max', 'hist_card_purchase_date_min',
'hist_new_card_purchase_date_max', 'hist_new_card_purchase_date_min','hist_card_minmax_diff_purchase_date','new_card_minmax_diff_purchase_date',            
              ]



train_df = df[df['target'].notnull()]
test_df = df[df['target'].isnull()]

feats = [f for f in train_df.columns if f not in drop_features]
# outlier tag
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1


cat_features = [c for c in feats if 'feature_' in c]
n_splits= 5

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4590)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
cv_list = []
print ('feats:' + str(len(feats)))
print (train_df[feats].shape)
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
  
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx] 
    
    print("Train Index:",train_idx,",Val Index:",valid_idx)
    
    params = {
               "objective" : "regression", 
               "boosting" : "gbdt", #
               "metric" : "rmse",  
               "max_depth": 9, #9
               "min_data_in_leaf": 70, #70
               "min_gain_to_split": 0.05,#0.05 
               "reg_alpha": 0.1, #0.1,
               "reg_lambda": 20, #20
               "num_leaves" : 120, #120
               "max_bin" : 300, #300
               "learning_rate" : 0.005, #0.005
               "bagging_fraction" : 1,
               "bagging_freq" : 1,
               "bagging_seed" : 4590,
               "feature_fraction" : 0.2, #0.2
               "verbosity": -1,
               "random_state": 4590,
    }

    if n_fold >= 0:
        print("Fold:" + str(n_fold))
        dtrain = lgb.Dataset(
            train_x, label=train_y,categorical_feature=cat_features,)#categorical_feature=cat_features
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain,categorical_feature=cat_features,) #weight=train_df.iloc[valid_idx]['outliers'] *  (-0.1) + 1
        bst = lgb.train(
            params, dtrain, num_boost_round=10000,
            valid_sets=[dval],  early_stopping_rounds=200,verbose_eval=100,)#
        
        new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
        for item in new_list:
            print (item) 

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)#bst.best_iteration
        oof_cv = rmse(valid_y,  oof_preds[valid_idx])
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += bst.predict(test_df[feats], num_iteration=bst.best_iteration) / folds.n_splits # test_df_new

cv = rmse(train_df['target'],  oof_preds)
print('Full OOF RMSE %.6f' % cv)  


oof_df = pd.DataFrame()
oof_df['card_id'] = train_df['card_id']
oof_df['target'] = oof_preds
oof_df[['card_id','target']].to_csv('../ensemble/lgb_v10_oof_' + str(cv) + '.csv',index=False)

test_df['target'] = sub_preds
test_df[['card_id','target']].to_csv('../ensemble/lgb_v10_pred_' + str(cv) + '.csv',index=False)
