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
    

