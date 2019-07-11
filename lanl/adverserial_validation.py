%%time
import os
import glob
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings('ignore')

train = pd.read_pickle('../input/datamatrixrosa_30000.pkl').reset_index(drop=True)
test = pd.read_pickle('../input/datamatrixrosatest_30000.pkl').reset_index(drop=True)
test['quake_id'] = 1000
train_index = pd.read_pickle('../input/allindex-300000-to-150000.pkl').reset_index(drop=True)
train['bag0'] = train_index['bag0']
train['bag1'] = train_index['bag1']
train['bag2'] = train_index['bag2']
train['bag3'] = train_index['bag3']
train['bag4'] = train_index['bag4']
train['quake_id'] = train_index['quake_id']
submission = pd.read_csv('../input/sample_submission.csv')
oof_df = pd.DataFrame()
oof_df['index'] = train['index']
oof_df['label'] = train['label']

drop_features=['index','label',
       'quake_id', 'class', 'bag0', 'bag1', 'bag2', 'bag3', 'bag4'
              ]
train_df = train.copy()
test_df = test.copy()

feats = [f for f in train_df.columns if f not in drop_features]

train_ad = train[(train['label']>0)][feats]
train_ad['target'] = 1
test_ad = test[feats]
test_ad['target'] = 0
train_test = pd.concat([train_ad, test_ad], axis =0)
target = train_test['target'].values

import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score, roc_curve



feats = [
"logtmean_0_5",
"logtnum_peaks_1_0_2",
"cid_ce_z_norm_mean_win_128",
"std_16_5-p",
"rollingVar10_quantile_4",
"cid_ce_z_norm_skew_win_128",
"rollingVar300_quantile_2",
"melspectrogram_mean_6",
"zero_crossing_rate_mean",
"logtmad_0_5",
"autocorre_mean_mean_win_128",
"num_peaks_10",
"logtnum_peaks_5_0_3",
"tnum_peaks_10_0_5",
"tstd_0_10",
"std_4096_5-p",
"melspectrogram_std_6",
"melspectrogram_mean_10",
"logtmean_0_3",
"tkurt_0_15",
"tskew_0_20",
"tkurt_0_20",
"logtnum_peaks_1_0_3",
"chroma_stft_mean_11",
"mfcc_mean_5",
"tskew_0_30",
"mean_mean_4096",
"abs_max",
"mfcc_std_3",
"tskew_0_10",
"mfcc_std_2",
"mean_mean_16",
"num_peaks_1",
"sta_lta_40.0_max",
"melspectrogram_std_10",
"mfcc_mean_3",
"mean",
"tmad_0_10",
"tskew_0_15",
"melspectrogram_mean_7",
"mfcc_std_0",
"spectral_bandwidth_mean",
"sta_lta_40.0_mean",
"num_peaks_5",
"skew_max_16",
"melspectrogram_std_9",
"tkurt_0_40",
"chroma_stft_mean_0",
"tskew_0_25",
"tkurt_0_10",
"rollingVar10_quantile_1",
"tskew_0_40",
"mean_max_4096",
"n_peaks_count_0_10000",
"logtnum_peaks_10_0_3",
"tmedian_0_10",
"range_m1000_0",
"sta_lta_400.0_median",
"melspectrogram_std_3",
"tstd_0_5",
"tauto10_0_30",
"mean_max_16",
"melspectrogram_std_7",
"skew_std_16",
"n_peaks_count_0_50",
"tmad_0_15",
"tkurt_0_25",
"logtskew_0_5",
"p_peaks_count_0_10000",
"tmean_0_10",
"tkurt_0_30",
"rollingMean10_quantile_4",
"mfcc_std_5",
"spectral_flatness_std",
"tmean_0_20",
"melspectrogram_mean_5",
"logtskew_0_3",
"tstd_0_15",
"sta_lta_400.0_std",
"mfcc_std_1",
"mfcc_mean_4",
"spectral_flatness_mean",
"sta_lta_4000.0_max",
"tmean_0_40",
"tmad_0_5",
"kurt_mean_16",
"tkurt_0_5",
"tmean_0_30",
"logtkurt_0_5",
"c_5",
"tmedian_0_40",
"abs_95-p",
"melspectrogram_std_1",
"logtskew_0_2",
"logtnum_peaks_10_0_5",
"logtnum_peaks_10_0_2",
"melspectrogram_std_4",
"abs_mean_4096",
"logtauto5_0_5",
"melspectrogram_mean_11",
"sta_lta_40.0_std",
"abs_std_16",
"tnum_peaks_1_0_20",
"sta_lta_400.0_max",
"skew_mean_16",
"chroma_stft_mean_10",
"logtstd_0_5",
"logtnum_peaks_5_0_2",
"melspectrogram_std_11",
"chroma_stft_std_8",
"tauto10_0_5",
"skew",
"skew_max_4096",
]    


feats = [
"logtmean_0_5",
"logtnum_peaks_1_0_2",
"cid_ce_z_norm_mean_win_128",
"std_16_5-p",
"rollingVar10_quantile_4",
"cid_ce_z_norm_skew_win_128",
"rollingVar300_quantile_2",
"melspectrogram_mean_6",
"zero_crossing_rate_mean",
"logtmad_0_5",
"autocorre_mean_mean_win_128",
"num_peaks_10",
"logtnum_peaks_5_0_3",
"tnum_peaks_10_0_5",
"tstd_0_10",
"std_4096_5-p",
"melspectrogram_std_6",
"melspectrogram_mean_10",
"logtmean_0_3",
"tkurt_0_15",    
]
n_splits= 5

folds = KFold(n_splits=n_splits, shuffle=True, random_state=223)

oof_preds = np.zeros(train_test.shape[0])

cv_list = []
print ('feats:' + str(len(feats)))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_test[feats], train_test['target'])):
    train_x, train_y = train_test[feats].iloc[train_idx], train_test['target'].iloc[train_idx]
    
    valid_x, valid_y = train_test[feats].iloc[valid_idx], train_test['target'].iloc[valid_idx] 
    
    print("Train Shape:",train_x.shape,",Val Shape:",valid_x.shape)
    
    
    params = {
               "objective" : "binary", 
               "boosting" : "gbdt", #
               "metric" : "auc",  
               "max_depth": 10, #10
               "reg_alpha": 0.1, #0.1,
               "reg_lambda": 10, #10
               "num_leaves" : 31, 
               "max_bin" : 256, 
               "learning_rate" : 0.01, 
               "bagging_fraction" : 0.6,#0.6
               "bagging_freq" : 3,
               "bagging_seed" : 0,
               "feature_fraction" : 0.8, 
               "random_state": 0,
    }

    if n_fold >= 0:
        print("Fold:" + str(n_fold))
        dtrain = lgb.Dataset(
            train_x, label=train_y,)#categorical_feature=cat_features
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain,) #weight=train_df.iloc[valid_idx]['outliers'] *  (-0.1) + 1
        bst = lgb.train(
            params, dtrain, num_boost_round=10000,
            valid_sets=[dval],  early_stopping_rounds=200,verbose_eval=100,)#
        
        new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:30]
        for item in new_list:
            print (item) 

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)#bst.best_iteration
        oof_cv = roc_auc_score(valid_y,  oof_preds[valid_idx])
        cv_list.append(oof_cv)
        print (cv_list)
 
cv = roc_auc_score(train_test['target'],  oof_preds)
print('Full OOF AUC %.6f' % cv)  


%%time
from sklearn.model_selection import KFold
import time
from lightgbm import LGBMClassifier
import lightgbm as lgb

def get_feature_importances(data, shuffle, seed=None):
    # Gather real features
    train_features = [f for f in data if f not in ['target','index']]
    # Go over fold and keep track of CV score (train and valid) and feature importances
    
    # Shuffle target if required
    y = data['target'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['target'].copy().sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=False)

    lgb_params = {
               "objective" : "binary", 
               "boosting" : "gbdt", #
               "metric" : "auc",  
               "max_depth": 10, #10
               "reg_alpha": 0.1, #0.1,
               "reg_lambda": 10, #10
               "num_leaves" : 31, 
               "max_bin" : 256, 
               "learning_rate" : 0.01, 
               "bagging_fraction" : 0.6,#0.6
               "bagging_freq" : 3,
               "bagging_seed" : 0,
               "feature_fraction" : 0.8, 
               "random_state": 0,
    }
  
    
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=900)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))
    
    return imp_df

np.random.seed(817)
# Get the actual importance, i.e. without shuffling
#train_test = train_test.copy()
feats = [f for f in train_test.columns if f not in ['index',
       'quake_id', 'class', 'bag','label', 'bag0', 'bag1', 'bag2', 'bag3', 'bag4']]

feats = ['target',
"logtmean_0_5",
"logtnum_peaks_1_0_2",
"cid_ce_z_norm_mean_win_128",
"std_16_5-p",
"rollingVar10_quantile_4",
"cid_ce_z_norm_skew_win_128",
"rollingVar300_quantile_2",
"melspectrogram_mean_6",
"zero_crossing_rate_mean",
"logtmad_0_5",
"autocorre_mean_mean_win_128",
"num_peaks_10",
"logtnum_peaks_5_0_3",
"tnum_peaks_10_0_5",
"tstd_0_10",
"std_4096_5-p",
"melspectrogram_std_6",
"melspectrogram_mean_10",
"logtmean_0_3",
"tkurt_0_15",
"tskew_0_20",
"tkurt_0_20",
"logtnum_peaks_1_0_3",
"chroma_stft_mean_11",
"mfcc_mean_5",
"tskew_0_30",
"mean_mean_4096",
"abs_max",
"mfcc_std_3",
"tskew_0_10",
"mfcc_std_2",
"mean_mean_16",
"num_peaks_1",
"sta_lta_40.0_max",
"melspectrogram_std_10",
"mfcc_mean_3",
"mean",
"tmad_0_10",
"tskew_0_15",
"melspectrogram_mean_7",
"mfcc_std_0",
"spectral_bandwidth_mean",
"sta_lta_40.0_mean",
"num_peaks_5",
"skew_max_16",
"melspectrogram_std_9",
"tkurt_0_40",
"chroma_stft_mean_0",
"tskew_0_25",
"tkurt_0_10",
"rollingVar10_quantile_1",
"tskew_0_40",
"mean_max_4096",
"n_peaks_count_0_10000",
"logtnum_peaks_10_0_3",
"tmedian_0_10",
"range_m1000_0",
"sta_lta_400.0_median",
"melspectrogram_std_3",
"tstd_0_5",
"tauto10_0_30",
"mean_max_16",
"melspectrogram_std_7",
"skew_std_16",
"n_peaks_count_0_50",
"tmad_0_15",
"tkurt_0_25",
"logtskew_0_5",
"p_peaks_count_0_10000",
"tmean_0_10",
"tkurt_0_30",
"rollingMean10_quantile_4",
"mfcc_std_5",
"spectral_flatness_std",
"tmean_0_20",
"melspectrogram_mean_5",
"logtskew_0_3",
"tstd_0_15",
"sta_lta_400.0_std",
"mfcc_std_1",
"mfcc_mean_4",
"spectral_flatness_mean",
"sta_lta_4000.0_max",
"tmean_0_40",
"tmad_0_5",
"kurt_mean_16",
"tkurt_0_5",
"tmean_0_30",
"logtkurt_0_5",
"c_5",
"tmedian_0_40",
"abs_95-p",
"melspectrogram_std_1",
"logtskew_0_2",
"logtnum_peaks_10_0_5",
"logtnum_peaks_10_0_2",
"melspectrogram_std_4",
"abs_mean_4096",
"logtauto5_0_5",
"melspectrogram_mean_11",
"sta_lta_40.0_std",
"abs_std_16",
"tnum_peaks_1_0_20",
"sta_lta_400.0_max",
"skew_mean_16",
"chroma_stft_mean_10",
"logtstd_0_5",
"logtnum_peaks_5_0_2",
"melspectrogram_std_11",
"chroma_stft_std_8",
"tauto10_0_5",
"skew",
"skew_max_4096",
]    
actual_imp_df = get_feature_importances(data=train_test[feats], shuffle=False)

%%time
null_imp_df = pd.DataFrame()
nb_runs = 50
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=train_test[feats], shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)
    
null_imp_df.to_csv('null_importances_distribution_rf.csv')
actual_imp_df.to_csv('actual_importances_ditribution_rf.csv')    

%%time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
%matplotlib inline

feature_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(1, 2)
# Plot Split importances
# ax = plt.subplot(gs[0, 0])
# sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:100], ax=ax)
# ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
# Plot Gain importances
ax = plt.subplot(gs[0, 1])
sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:100], ax=ax)
ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
plt.tight_layout()

pd.set_option('max_rows',400)
new_list = scores_df.sort_values(by=['gain_score'],ascending=True).reset_index(drop=True)
print (new_list.tail(400))

for item in new_list['feature']:
    #print (item) 
    print ('"' + str(item) +  '",') 
