%%time

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

train = pd.read_csv('../input/train.csv', parse_dates = ['activation_date'])
test = pd.read_csv('../input/test.csv', parse_dates = ['activation_date'])

train_active = pd.read_csv('../input/train_active.csv', parse_dates = ['activation_date'])
test_active = pd.read_csv('../input/test_active.csv', parse_dates = ['activation_date'])
train_periods = pd.read_csv('../input/periods_train.csv', parse_dates=['activation_date','date_from', 'date_to'])
test_periods = pd.read_csv('../input/periods_test.csv', parse_dates=['activation_date','date_from', 'date_to'])

df_all = pd.concat([
    train,
    train_active,
    test,
    test_active
]).reset_index(drop=True)
df_all.drop_duplicates(['item_id'], inplace=True)

df_all['wday'] = df_all['activation_date'].dt.weekday

df_all['price'].fillna(0, inplace=True)
df_all['price'] = np.log1p(df_all['price'])

df_all['city'] = df_all['city'] + "_" + df_all['region']

df_all['param_123'] = (df_all['param_1'].fillna('') + ' ' + df_all['param_2'].fillna('') + ' ' + df_all['param_3'].fillna('')).astype(str)


text_vars = ['user_id','region', 'city', 'parent_category_name', 'category_name', 'user_type','param_1','param_2','param_3','param_123']
for col in tqdm(text_vars):
    lbl = LabelEncoder()
    lbl.fit(df_all[col].values.astype('str'))
    df_all[col] = lbl.transform(df_all[col].values.astype('str'))
    
all_periods = pd.concat([
    train_periods,
    test_periods
])
all_periods['days_up'] = all_periods['date_to'].dt.dayofyear - all_periods['date_from'].dt.dayofyear

def agg(df,agg_cols):
    for c in tqdm(agg_cols):
        new_feature = '{}_{}_{}'.format('_'.join(c['groupby']), c['agg'], c['target'])
        gp = df.groupby(c['groupby'])[c['target']].agg(c['agg']).reset_index().rename(index=str, columns={c['target']:new_feature})
        df = df.merge(gp,on=c['groupby'],how='left')
    return df

agg_cols = [
    {'groupby': ['item_id'], 'target':'days_up', 'agg':'count'},
    {'groupby': ['item_id'], 'target':'days_up', 'agg':'sum'},   
]

all_periods = agg(all_periods,agg_cols)

all_periods.drop_duplicates(['item_id'], inplace=True)
#all_periods.drop(['activation_date','date_from','date_to','days_up','days_total'],axis=1, inplace=True)
all_periods.drop(['activation_date','date_from','date_to'],axis=1, inplace=True)
all_periods.reset_index(drop=True,inplace=True)

df_all = df_all.merge(all_periods, on='item_id', how='left')

#Impute Days up
df_all['item_id_count_days_up_impute'] = df_all['item_id_count_days_up']
df_all['item_id_sum_days_up_impute'] = df_all['item_id_sum_days_up']

enc = df_all.groupby('category_name')['item_id_count_days_up'].agg('median').astype(np.float32).reset_index()
enc.columns = ['category_name' ,'count_days_up_impute']
df_all = pd.merge(df_all, enc, how='left', on='category_name')
df_all['item_id_count_days_up_impute'].fillna(df_all['count_days_up_impute'], inplace=True)

enc = df_all.groupby('category_name')['item_id_sum_days_up'].agg('median').astype(np.float32).reset_index()
enc.columns = ['category_name' ,'sum_days_up_impute']
df_all = pd.merge(df_all, enc, how='left', on='category_name')
df_all['item_id_sum_days_up_impute'].fillna(df_all['sum_days_up_impute'], inplace=True)


df_numerical_active = df_all[['category_name','city','deal_probability',
              'item_id','item_seq_number','param_1','param_2','param_3','param_123','parent_category_name','price',
              'region','user_id','user_type','wday','item_id_count_days_up','item_id_sum_days_up',
              'item_id_count_days_up_impute','item_id_sum_days_up_impute']]     

# create numerical features with active
#df_numerical_active.to_pickle('/tmp/basic_numerical_active.pkl')

%%time

## with active

df_all_tmp = df_numerical_active.copy()
raw_columns = df_all_tmp.columns.values
print ('1')
## aggregate features
def agg(df,agg_cols):
    for c in tqdm(agg_cols):
        new_feature = '{}_{}_{}'.format('_'.join(c['groupby']), c['agg'], c['target'])
        gp = df.groupby(c['groupby'])[c['target']].agg(c['agg']).reset_index().rename(index=str, columns={c['target']:new_feature})
        df = df.merge(gp,on=c['groupby'],how='left')
    return df

agg_cols = [
############################unique aggregation##################################
    {'groupby': ['user_id'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['parent_category_name'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['category_name'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['region'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['city'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['wday'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['param_1'], 'target':'price', 'agg':'nunique'},

    {'groupby': ['category_name'], 'target':'image_top_1', 'agg':'nunique'},
    {'groupby': ['parent_category_name'], 'target':'image_top_1', 'agg':'nunique'},    
    {'groupby': ['user_id'], 'target':'image_top_1', 'agg':'nunique'},
    
    
    {'groupby': ['user_id'], 'target':'parent_category_name', 'agg':'nunique'},
    {'groupby': ['user_id'], 'target':'category_name', 'agg':'nunique'},
    {'groupby': ['user_id'], 'target':'wday', 'agg':'nunique'},
    {'groupby': ['user_id'], 'target':'param_1', 'agg':'nunique'},
    
############################count aggregation##################################  
    {'groupby': ['user_id'], 'target':'item_id', 'agg':'count'},

    {'groupby': ['user_id','param_1'], 'target':'item_id', 'agg':'count'},

    {'groupby': ['user_id','region'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','city'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','parent_category_name'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','category_name'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','image_top_1'], 'target':'item_id', 'agg':'count'},
 
    {'groupby': ['user_id','wday','category_name'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','image_top_1'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','parent_category_name'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','city'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','region'], 'target':'item_id', 'agg':'count'},
    
    {'groupby': ['user_id','category_name','city'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','category_name','city'], 'target':'item_id', 'agg':'count'},
    
    {'groupby': ['price'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['price','user_id'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['price','category_name'], 'target':'item_id', 'agg':'count'},
    
############################mean/median/sum/min/max aggregation##################################    
    
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'mean'},
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'median'},
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'sum'},
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'max'},

    {'groupby': ['param_2'], 'target':'price', 'agg':'mean'},
    {'groupby': ['param_2'], 'target':'price', 'agg':'max'},
    {'groupby': ['param_3'], 'target':'price', 'agg':'mean'},
    {'groupby': ['param_3'], 'target':'price', 'agg':'max'},
    
    {'groupby': ['user_id'], 'target':'price', 'agg':'mean'},
    {'groupby': ['user_id'], 'target':'price', 'agg':'median'},
    {'groupby': ['user_id'], 'target':'price', 'agg':'sum'},
    {'groupby': ['user_id'], 'target':'price', 'agg':'min'},
    {'groupby': ['user_id'], 'target':'price', 'agg':'max'},

    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'mean'},
    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'median'},
    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'sum'},
    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'min'},
    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'max'},


    {'groupby': ['image_top_1'], 'target':'price', 'agg':'mean'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'median'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'sum'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'max'},

    {'groupby': ['param_1'], 'target':'price', 'agg':'mean'},
    {'groupby': ['param_1'], 'target':'price', 'agg':'max'},


    {'groupby': ['region'], 'target':'price', 'agg':'mean'},
    {'groupby': ['region'], 'target':'price', 'agg':'max'},
    
    {'groupby': ['city'], 'target':'price', 'agg':'mean'},
    {'groupby': ['city'], 'target':'price', 'agg':'max'},
    
    {'groupby': ['parent_category_name'], 'target':'price', 'agg':'mean'},
    {'groupby': ['parent_category_name'], 'target':'price', 'agg':'sum'},
    {'groupby': ['parent_category_name'], 'target':'price', 'agg':'max'},

    {'groupby': ['category_name'], 'target':'price', 'agg':'mean'},
    {'groupby': ['category_name'], 'target':'price', 'agg':'sum'},
    {'groupby': ['category_name'], 'target':'price', 'agg':'max'},   
    
    {'groupby': ['wday','category_name','city'], 'target':'price', 'agg':'mean'},
    {'groupby': ['wday','category_name','city'], 'target':'price', 'agg':'median'},
    {'groupby': ['wday','category_name','city'], 'target':'price', 'agg':'sum'},
    {'groupby': ['wday','category_name','city'], 'target':'price', 'agg':'max'},

    {'groupby': ['wday','region'], 'target':'price', 'agg':'mean'},
    {'groupby': ['wday','region'], 'target':'price', 'agg':'median'},
    {'groupby': ['wday','region'], 'target':'price', 'agg':'sum'},
    {'groupby': ['wday','region'], 'target':'price', 'agg':'max'},    
    
    {'groupby': ['wday','city'], 'target':'price', 'agg':'mean'},
    {'groupby': ['wday','city'], 'target':'price', 'agg':'median'},
    {'groupby': ['wday','city'], 'target':'price', 'agg':'sum'},
    {'groupby': ['wday','city'], 'target':'price', 'agg':'max'},

    {'groupby': ['wday','user_id'], 'target':'price', 'agg':'mean'},
    {'groupby': ['wday','user_id'], 'target':'price', 'agg':'median'},
    {'groupby': ['wday','user_id'], 'target':'price', 'agg':'sum'},
    {'groupby': ['wday','user_id'], 'target':'price', 'agg':'min'},
    {'groupby': ['wday','user_id'], 'target':'price', 'agg':'max'},
    
    
    {'groupby': ['user_id'], 'target':'item_id_sum_days_up', 'agg':'mean'},
    {'groupby': ['user_id'], 'target':'item_id_count_days_up', 'agg':'mean'},     
]
print ('2')
df_all_tmp = agg(df_all_tmp,agg_cols)
tmp_columns = df_all_tmp.columns.values

df_train = df_all_tmp[df_all_tmp['deal_probability'].notnull()]
test_id = pd.read_csv('../input/test.csv',usecols=['item_id'])        
test_id = test_id.merge(df_all_tmp,on='item_id',how='left')
del df_all_tmp

for i in tmp_columns:
    if i not in raw_columns:
        print (i)
        df_train[i].to_pickle('/tmp/features/number_agg/clean_train_active/' + str(i))
        test_id[i].to_pickle('/tmp/features/number_agg/clean_test_active/' + str(i))
     

%%time

train = pd.read_csv('../input/train.csv', parse_dates = ['activation_date'])
test = pd.read_csv('../input/test.csv', parse_dates = ['activation_date'])

df_all = pd.concat([train,test],axis=0).reset_index(drop=True)
df_all['wday'] = df_all['activation_date'].dt.weekday

df_all['price'].fillna(0, inplace=True)
df_all['price'] = np.log1p(df_all['price'])

df_all['city'] = df_all['city'] + "_" + df_all['region']

df_all['param_123'] = (df_all['param_1'].fillna('') + ' ' + df_all['param_2'].fillna('') + ' ' + df_all['param_3'].fillna('')).astype(str)

df_all['title'] = df_all['title'].fillna('').astype(str)
df_all['text'] = df_all['description'].fillna('').astype(str) + ' ' + df_all['title'].fillna('').astype(str) + ' ' + df_all['param_123'].fillna('').astype(str)

# from https://www.kaggle.com/christofhenkel/text2image-top-1
# tr.csv is train_image_top_1_features.csv + test_image_top_1_features.csv
image_top_2 = pd.read_csv('/tmp/features/category/tr.csv')
df_all['image_top_2'] = image_top_2['image_top_1']


text_vars = ['user_id','region', 'city', 'parent_category_name', 'category_name', 'user_type','param_1','param_2','param_3','param_123']
for col in tqdm(text_vars):
    lbl = LabelEncoder()
    lbl.fit(df_all[col].values.astype('str'))
    df_all[col] = lbl.transform(df_all[col].values.astype('str'))

# create image_top_1,2 category feature
df_train = df_all[df_all['deal_probability'].notnull()]
df_test = df_all[df_all['deal_probability'].isnull()]

df_train['image_top_1'].to_pickle('/tmp/features/number_agg/clean_train_image_top_1/image_top_1')
df_test['image_top_1'].to_pickle('/tmp/features/number_agg/clean_test_image_top_1/image_top_1')  
df_train['image_top_2'].to_pickle('/tmp/features/number_agg/clean_train_image_top_1/image_top_2')
df_test['image_top_2'].to_pickle('/tmp/features/number_agg/clean_test_image_top_1/image_top_2')  

# create image_top_1,2 aggregation feature
df_all_tmp = df_all.copy()
raw_columns = df_all_tmp.columns.values

agg_cols = [

############################unique aggregation##################################
    
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['image_top_2'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['user_id'], 'target':'image_top_1', 'agg':'nunique'},
    {'groupby': ['user_id'], 'target':'image_top_2', 'agg':'nunique'},
    
############################count aggregation##################################  

    {'groupby': ['user_id','image_top_1'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','image_top_2'], 'target':'item_id', 'agg':'count'},

    {'groupby': ['user_id','wday','image_top_1'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','image_top_2'], 'target':'item_id', 'agg':'count'},

############################mean/median/sum/min/max aggregation##################################    
    
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'mean'},
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'median'},
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'sum'},
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'max'},
    {'groupby': ['image_top_2','user_id'], 'target':'price', 'agg':'mean'},
    {'groupby': ['image_top_2','user_id'], 'target':'price', 'agg':'median'},
    {'groupby': ['image_top_2','user_id'], 'target':'price', 'agg':'sum'},
    {'groupby': ['image_top_2','user_id'], 'target':'price', 'agg':'max'},

    {'groupby': ['image_top_1'], 'target':'price', 'agg':'mean'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'median'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'sum'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'max'},
    {'groupby': ['image_top_2'], 'target':'price', 'agg':'mean'},
    {'groupby': ['image_top_2'], 'target':'price', 'agg':'median'},
    {'groupby': ['image_top_2'], 'target':'price', 'agg':'sum'},
    {'groupby': ['image_top_2'], 'target':'price', 'agg':'max'},
]

df_all_tmp = agg(df_all_tmp,agg_cols)
tmp_columns = df_all_tmp.columns.values

df_train = df_all_tmp[df_all_tmp['deal_probability'].notnull()]
df_test = df_all_tmp[df_all_tmp['deal_probability'].isnull()]
for i in tmp_columns:
    if i not in raw_columns:
        print (i)
        df_train[i].to_pickle('/tmp/features/number_agg/clean_train_image_top_1/' + str(i))
        df_test[i].to_pickle('/tmp/features/number_agg/clean_test_image_top_1/' + str(i))  
        
%%time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, model_selection, metrics
from sklearn.decomposition import TruncatedSVD

train = pd.read_csv('../input/train.csv', parse_dates = ['activation_date'])
test = pd.read_csv('../input/test.csv', parse_dates = ['activation_date'])
df_all = pd.concat([train,test],axis=0).reset_index(drop=True)
df_all['param_123'] = (df_all['param_1'].fillna('') + ' ' + df_all['param_2'].fillna('') + ' ' + df_all['param_3'].fillna('')).astype(str)
df_all['title'] = df_all['title'].fillna('').astype(str)
df_all['text'] = df_all['description'].fillna('').astype(str) + ' ' + df_all['title'].fillna('').astype(str) + ' ' + df_all['param_123'].fillna('').astype(str)
df_text = df_all[['deal_probability','title','param_123','text']]
df_train_text = df_text[df_text['deal_probability'].notnull()]
df_test_text = df_text[df_text['deal_probability'].isnull()]

### TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1))

full_title_tfidf = tfidf_vec.fit_transform(df_text['title'].values.tolist() )
train_title_tfidf = tfidf_vec.transform(df_train_text['title'].values.tolist())
test_title_tfidf = tfidf_vec.transform(df_test_text['title'].values.tolist())

### SVD Components ###
n_comp = 40

svd_title_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_title_obj.fit(full_title_tfidf)
train_title_svd = pd.DataFrame(svd_title_obj.transform(train_title_tfidf))
test_title_svd = pd.DataFrame(svd_title_obj.transform(test_title_tfidf))
train_title_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_title_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
for i in train_title_svd.columns:
    print (i)
    test_title_svd[i].to_pickle('/tmp/features/tsvd/train/' + str(i))
    test_title_svd[i].to_pickle('/tmp/features/tsvd/test/' + str(i))  
    
%%time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
import re
import string

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))

stopwords = {x: 1 for x in stopwords.words('russian')}
punct = set(string.punctuation)
emoji = set()
for s in df_all['text'].fillna('').astype(str):
    for c in s:
        if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
            continue
        emoji.add(c)

all = df_text.copy()

# Meta Text Features
textfeats = ['param_123']
for cols in textfeats:   
    all[cols] = all[cols].astype(str) 

    all[cols + '_num_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-ЯA-Z]', x))
    all[cols + '_num_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-яa-z]', x))
    all[cols + '_num_rus_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-Я]', x))
    all[cols + '_num_eng_cap'] = all[cols].apply(lambda x: count_regexp_occ('[A-Z]', x))    
    all[cols + '_num_rus_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-я]', x))
    all[cols + '_num_eng_low'] = all[cols].apply(lambda x: count_regexp_occ('[a-z]', x))
    all[cols + '_num_dig'] = all[cols].apply(lambda x: count_regexp_occ('[0-9]', x))
    
    all[cols + '_num_pun'] = all[cols].apply(lambda x: sum(c in punct for c in x))
    all[cols + '_num_space'] = all[cols].apply(lambda x: sum(c.isspace() for c in x))

    
    all[cols + '_num_chars'] = all[cols].apply(len) # Count number of Characters
    all[cols + '_num_words'] = all[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    all[cols + '_num_unique_words'] = all[cols].apply(lambda comment: len(set(w for w in comment.split())))
    
    all[cols + '_ratio_unique_words'] = all[cols+'_num_unique_words'] / (all[cols+'_num_words']+1)
    
textfeats = ['text']
for cols in textfeats:   
    all[cols] = all[cols].astype(str)
    all[cols + '_num_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-ЯA-Z]', x))
    all[cols + '_num_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-яa-z]', x))
    all[cols + '_num_rus_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-Я]', x))
    all[cols + '_num_eng_cap'] = all[cols].apply(lambda x: count_regexp_occ('[A-Z]', x))    
    all[cols + '_num_rus_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-я]', x))
    all[cols + '_num_eng_low'] = all[cols].apply(lambda x: count_regexp_occ('[a-z]', x))
    all[cols + '_num_dig'] = all[cols].apply(lambda x: count_regexp_occ('[0-9]', x))
    
    all[cols + '_num_pun'] = all[cols].apply(lambda x: sum(c in punct for c in x))
    all[cols + '_num_space'] = all[cols].apply(lambda x: sum(c.isspace() for c in x))
    all[cols + '_num_emo'] = all[cols].apply(lambda x: sum(c in emoji for c in x))
    all[cols + '_num_row'] = all[cols].apply(lambda x: x.count('/\n'))
   
    all[cols + '_num_chars'] = all[cols].apply(len) # Count number of Characters
    all[cols + '_num_words'] = all[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    all[cols + '_num_unique_words'] = all[cols].apply(lambda comment: len(set(w for w in comment.split())))
    
    all[cols + '_ratio_unique_words'] = all[cols+'_num_unique_words'] / (all[cols+'_num_words']+1) # Count Unique Words    

    all[cols +'_num_stopwords'] = all[cols].apply(lambda x: len([w for w in x.split() if w in stopwords]))
    all[cols +'_num_words_upper'] = all[cols].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    all[cols +'_num_words_lower'] = all[cols].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    all[cols +'_num_words_title'] = all[cols].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
  
    
textfeats = ['title']
for cols in textfeats:   
    all[cols] = all[cols].astype(str)
    all[cols + '_num_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-ЯA-Z]', x))
    all[cols + '_num_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-яa-z]', x))
    all[cols + '_num_rus_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-Я]', x))
    all[cols + '_num_eng_cap'] = all[cols].apply(lambda x: count_regexp_occ('[A-Z]', x))    
    all[cols + '_num_rus_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-я]', x))
    all[cols + '_num_eng_low'] = all[cols].apply(lambda x: count_regexp_occ('[a-z]', x))
    all[cols + '_num_dig'] = all[cols].apply(lambda x: count_regexp_occ('[0-9]', x))
    
    all[cols + '_num_pun'] = all[cols].apply(lambda x: sum(c in punct for c in x))
    all[cols + '_num_space'] = all[cols].apply(lambda x: sum(c.isspace() for c in x))
    
    all[cols + '_num_chars'] = all[cols].apply(len) # Count number of Characters
    all[cols + '_num_words'] = all[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    all[cols + '_num_unique_words'] = all[cols].apply(lambda comment: len(set(w for w in comment.split())))
    
    all[cols + '_ratio_unique_words'] = all[cols+'_num_unique_words'] / (all[cols+'_num_words']+1)
    
df_train = all[all['deal_probability'].notnull()]
df_test = all[all['deal_probability'].isnull()]
df_all_tmp = all.drop(['deal_probability','param_123','title','text'],axis=1)
tmp_columns = df_all_tmp.columns.values
for i in tmp_columns:
    print (i)
    df_train[i].to_pickle('/tmp/features/text_agg/train/' + str(i))
    df_test[i].to_pickle('/tmp/features/text_agg/test/' + str(i))  
    
%%time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
import pickle
from scipy import sparse
from nltk.tokenize.toktok import ToktokTokenizer # tokenizer tested on russian
from nltk.stem.snowball import RussianStemmer
from nltk import sent_tokenize # should be multilingual
from string import punctuation
from nltk import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import FastText
import re
from string import punctuation

punct = set(punctuation)

# Tf-Idf
def clean_text(s):
    s = re.sub('м²|\d+\\/\d|\d+-к|\d+к', ' ', s.lower())
    s = re.sub('\\s+', ' ', s)
    s = s.strip()
    return s
    
print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

df_text['param_123'] = df_text['param_123'].apply(lambda x: clean_text(x))
df_text['title'] = df_text['title'].apply(lambda x: clean_text(x))
df_text["text"] = df_text["text"].apply(lambda x: clean_text(x))

df_train_text = df_text[df_text['deal_probability'].notnull()]
df_test_text = df_text[df_text['deal_probability'].isnull()]

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "lowercase": True,
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}

def get_col(col_name): return lambda x: x[col_name]
vectorizer = FeatureUnion([
        ('text',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=200000,
            **tfidf_para,
            preprocessor=get_col('text'))),
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            #lowercase=True,
            #max_features=7000,
            preprocessor=get_col('title'))),
        ('param_123',TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            #lowercase=True,
            #max_features=7000,
            preprocessor=get_col('param_123')))    
    ])
 

vectorizer.fit(df_text.to_dict('records'))

ready_df_train = vectorizer.transform(df_train_text.to_dict('records'))
ready_df_test = vectorizer.transform(df_test_text.to_dict('records'))

tfvocab = vectorizer.get_feature_names()

sparse.save_npz('/tmp/features/nlp/ready_df_train_200000_new.npz', ready_df_train)
sparse.save_npz('/tmp/features/nlp/ready_df_test_200000_new.npz', ready_df_test)

with open('/tmp/features/nlp/tfvocab_200000_new.pkl', 'wb') as tfvocabfile:  
    pickle.dump(tfvocab, tfvocabfile)
    
##LGBM Model

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import sparse
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
import os
import glob

df_all = pickle.load(open('/tmp/basic_numerical_active.pkl','rb'))
df_train = df_all[df_all['deal_probability'].notnull()]
df_test = df_all[df_all['deal_probability'].isnull()].reset_index(drop=True)
y = df_all[df_all['deal_probability'].notnull()].deal_probability

# tfidf
ready_df_train = sparse.load_npz('/tmp/features/nlp/ready_df_train_200000_new.npz')
ready_df_test = sparse.load_npz('/tmp/features/nlp/ready_df_test_200000_new.npz')
tfvocab = pickle.load(open('/tmp/features/nlp/tfvocab_200000_new.pkl', 'rb'))

# image - put features to /tmp/features/image/train/ /data/features/image/test/
for fn in glob.glob('/tmp/features/image/train/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/image/test/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    
df_train['dullnessminuswhiteness'] = df_train['dullness'] - df_train['whiteness']
df_test['dullnessminuswhiteness'] = df_test['dullness'] - df_test['whiteness']

# tsvd
for fn in glob.glob('/tmp/features/tsvd/tmp/train/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/tsvd/tmp/test/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    
# text agg
for fn in glob.glob('/tmp/features/text_agg/train/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/text_agg/test/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))       
    
#number agg    
for fn in glob.glob('/tmp/features/number_agg/clean_train_active/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/number_agg/clean_test_active/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))     
    
# image_top_1 image_top_2
for fn in glob.glob('/tmp/features/number_agg/clean_train_image_top_1/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('/tmp/features/number_agg/clean_test_image_top_1/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))     
    
# diff features
df_train['image_top_1_diff_price'] = df_train['price'] - df_train['image_top_1_median_price']
df_train['parent_category_name_diff_price'] = df_train['price'] - df_train['parent_category_name_mean_price']
df_train['category_name_diff_price'] = df_train['price'] - df_train['category_name_mean_price']
df_train['param_1_diff_price'] = df_train['price'] - df_train['param_1_mean_price']
df_train['param_2_diff_price'] = df_train['price'] - df_train['param_2_mean_price']
df_train['item_seq_number_diff_price'] = df_train['price'] - df_train['item_seq_number_mean_price']
df_train['user_id_diff_price'] = df_train['price'] - df_train['user_id_mean_price']
df_train['region_diff_price'] = df_train['price'] - df_train['region_mean_price']
df_train['city_diff_price'] = df_train['price'] - df_train['city_mean_price']

df_test['image_top_1_diff_price'] = df_test['price'] - df_test['image_top_1_median_price']
df_test['parent_category_name_diff_price'] = df_test['price'] - df_test['parent_category_name_mean_price']
df_test['category_name_diff_price'] = df_test['price'] - df_test['category_name_mean_price']
df_test['param_1_diff_price'] = df_test['price'] - df_test['param_1_mean_price']
df_test['param_2_diff_price'] = df_test['price'] - df_test['param_2_mean_price']
df_test['item_seq_number_diff_price'] = df_test['price'] - df_test['item_seq_number_mean_price']
df_test['user_id_diff_price'] = df_test['price'] - df_test['user_id_mean_price']
df_test['region_diff_price'] = df_test['price'] - df_test['region_mean_price']
df_test['city_diff_price'] = df_test['price'] - df_test['city_mean_price']

# drop_list
drop_list = [
    'param_123',
    'wday_region_mean_price',
    'wday_region_median_price',
    'wday_region_sum_price',
    'wday_region_max_price',   
    'wday_city_mean_price',
    'wday_city_median_price',
    'wday_city_sum_price',
    'wday_city_max_price', 
    'param_123_num_space',
    'param_123_num_pun',
    'title_num_pun',
    'title_num_space',

 ]

for d in drop_list:
    df_train.drop([d],axis=1,inplace=True)
    df_test.drop([d],axis=1,inplace=True)
    
# final feature    
from scipy.sparse import hstack, csr_matrix

df_train = df_train.drop([
                'deal_probability'],axis=1)   

df_test = df_test.drop([
                'deal_probability'],axis=1) 

X_tr = hstack([csr_matrix(df_train),ready_df_train]) # Sparse Matrix
X_test = hstack([csr_matrix(df_test),ready_df_test])

tfvocab = df_train.columns.tolist() + tfvocab

for shape in [X_tr,X_test]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))    

# train model and predict

from sklearn.model_selection import KFold

X = X_tr.tocsr()
#del X_tra
gc.collect()

test_pred = np.zeros(X_test.shape[0])
cat_features=['region','city','parent_category_name',
              'category_name',
              'user_type','image_top_1','param_1','param_2','param_3','wday']

params = {
    'boosting_type': 'gbdt',
    'objective': 'xentropy',
    'metric': 'rmse',
    'learning_rate': 0.015,
    'num_leaves': 600,  
    #'max_depth': 15,  
    'max_bin': 256,  
    'subsample': 1,  
    'colsample_bytree': 0.1,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
    'verbose': 1
    }

MAX_ROUNDS = 15000
NFOLDS = 5
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=228)
xval_err = 0

for i,(train_index,val_index) in enumerate(kfold.split(X,y)):
    print("Running fold {} / {}".format(i + 1, NFOLDS))
    print("Train Index:",train_index,",Val Index:",val_index)
    X_tra,X_val,y_tra,y_val = X[train_index, :], X[val_index, :], y[train_index], y[val_index]
    if i >=0:

        dtrain = lgb.Dataset(
            X_tra, label=y_tra, feature_name=tfvocab, categorical_feature=cat_features)
        dval = lgb.Dataset(
            X_val, label=y_val, reference=dtrain, feature_name=tfvocab, categorical_feature=cat_features)    
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dval], early_stopping_rounds=200, verbose_eval=200)
        val_pred = bst.predict(
            X_val, num_iteration=bst.best_iteration or MAX_ROUNDS)
        e = val_pred-y_val
        xval_err += np.dot(e,e)
        del dtrain,dval
        del X_tra,y_tra,y_val,X_val
        gc.collect()
        
        new_list = sorted(
            zip(tfvocab, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True)[:200]
        for item in new_list:
            print (item)  
            
        test_pred_current = bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
        test_pred += bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
        test_pred_current.dump('../models/kfold5_' + str(i) + '.pkl')
        del test_pred_current
        gc.collect()

print("Full Validation RMSE:", np.sqrt(xval_err/X.shape[0]))

test_pred /= NFOLDS

test = pd.read_csv('../input/test.csv', index_col = 'item_id', parse_dates = ['activation_date'])
testdex = test.index
sub = pd.DataFrame(test_pred,columns=["deal_probability"],index=testdex)
sub['deal_probability'] = sub['deal_probability'].clip(0.0, 1.0) # Between 0 and 1
sub.to_csv("../models/sub.csv",index=True,header=True)    
