class FFMFormat:
    def __init__(self, vector_feat, one_hot_feat, continous_feat):
        self.field_index_ = None
        self.feature_index_ = None
        self.vector_feat = vector_feat
        self.one_hot_feat = one_hot_feat
        self.continous_feat = continous_feat

    def get_params(self):
        pass

    def set_params(self, **parameters):
        pass

    def fit(self, df, y=None):
        self.field_index_ = {col: i for i, col in enumerate(df.columns)}
        self.feature_index_ = dict()
        last_idx = 0
        for col in df.columns:
            if col in self.one_hot_feat:
                print('Fitting Category Column: ' + col)
                df[col] = df[col].astype('int')
                vals = np.unique(df[col])
                for val in vals:
                    #if val == -1: continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            elif col in self.vector_feat:
                print('Fitting Vector Column: ' + col)
                vals = []
                for data in df[col].apply(str):
                    #if data != '-1':
                    for word in data.strip().split(' '):
                        vals.append(word)
                vals = np.unique(vals)
                for val in vals:
                    #if val == '-1': continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            else:
                print('Fitting Continous Column: ' + col)
                self.feature_index_[col] = last_idx
                last_idx += 1
#             if col in aid_inter_oh_feature:
#                 print('Fitting column (inter aid): ' + col)
#                 self.field_index_['aid_{}'.format(col)] = len(self.field_index_)
#                 df['aid'] = df['aid'].astype('int')
#                 df[col] = df[col].astype('int')
#                 vals = df[['aid', col]].drop_duplicates()
#                 for idx, val in vals.iterrows():
#                     if val[col] == -1: continue
#                     name = 'aid_{}_{}_{}'.format(val['aid'], col, val[col])
#                     if name not in self.feature_index_:
#                         self.feature_index_[name] = last_idx
#                         last_idx += 1
#             elif col in aid_inter_vec_feature:
#                 print('Fitting column (inter aid): ' + col)
#                 self.field_index_['aid_{}'.format(col)] = len(self.field_index_)
#                 df['aid'] = df['aid'].astype('int')
#                 df[col] = df[col].astype(str)
#                 vals = df[['aid', col]].drop_duplicates()
#                 for idx, val in vals.iterrows():
#                     if val[col] == '-1': continue
#                     for word in val[col].strip().split(' '):
#                         name = 'aid_{}_{}_{}'.format(val['aid'], col, word)
#                         if name not in self.feature_index_:
#                             self.feature_index_[name] = last_idx
#                             last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row):
        ffm = []
        for col, val in row.loc[row != 0].to_dict().items():
            if col in self.one_hot_feat:
                name = '{}_{}'.format(col, val)
                if name in self.feature_index_:
                    ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col in self.vector_feat:
                for word in str(val).split(' '):
                    name = '{}_{}'.format(col, word)
                    if name in self.feature_index_:
                        ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col in self.continous_feat:
                #if val != -1:
                ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
#             if col in aid_inter_oh_feature:
#                 name = 'aid_{}_{}_{}'.format(row['aid'], col, val)
#                 if name in self.feature_index_:
#                     ffm.append('{}:{}:1'.format(self.field_index_['aid_{}'.format(col)], self.feature_index_[name]))
#             elif col in aid_inter_vec_feature:
#                 for word in str(val).split(' '):
#                     name = 'aid_{}_{}_{}'.format(row['aid'], col, word)
#                     if name in self.feature_index_:
#                         ffm.append('{}:{}:1'.format(self.field_index_['aid_{}'.format(col)], self.feature_index_[name]))
        return ' '.join(ffm)

    def transform(self, df):
        return pd.Series({idx: self.transform_row_(row) for idx, row in df.iterrows()})
        
%%time
one_hot_feature = ['feature_1', 'feature_2', 'feature_3', 'hist_new_card_sum_category_1',
                   'hist_new_category_1_1','hist_card_last_doy','hist_card1_unique_month_lag','hist_new_card1_sum_category_1',
                   'hist_card1_unique_month','hist_card_sum_category_1','hist_card_last_woy','hist_new_card_unique_month_lag',
                   'hist_new_card_last_day','hist_new_card_last_doy','new_card_mean_month_lag']
vector_feature = []
continous_feature = ['hist_card0_mean_month_diff','hist_new_card1_mean_month_diff','hist_card1_mean_month_diff','new_card_max_purchase_amount',
                    'hist_card_mean_month_diff','sum_category_1_1_purchase_amount','new_card_std_purchase_amount','new_max_month_lag_2_purchase_amount',
                    'new_card_mean_purchase_amount','new_max_month_lag_1_purchase_amount','new_mean_month_lag_2_purchase_amount',
                    'new_min_month_lag_2_purchase_amount']  
df_ffm = ffm[one_hot_feature + vector_feature+ continous_feature]

ffm_fmt = FFMFormat(vector_feature, one_hot_feature, continous_feature)
user_ffm = ffm_fmt.fit_transform(df_ffm)
user_ffm.to_csv('../ffm/ffm.csv', index=False)

%%time
train = ffm[ffm['target'].notnull()]
test = ffm[ffm['target'].isnull()]
train_y = train['target'].values
with open('../ffm/ffm.csv') as f_in:
    f_out_train = open('../ffm/train_ffm.csv', 'w')
    f_out_test = open('../ffm/test_ffm.csv', 'w')
    for (i, line) in enumerate(f_in):
        if i < train.shape[0]:
            #print (i,line)
            f_out_train.write(str(train_y[i]) + ' ' + line)
        else:
            #print (i,line)
            f_out_test.write(line)
            
%%time
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

train_df = ffm[ffm['target'].notnull()]
test_df = ffm[ffm['target'].isnull()]

one_hot_feature = ['feature_1', 'feature_2', 'feature_3', 'hist_new_card_sum_category_1',
                   'hist_new_category_1_1','hist_card_last_doy','hist_card1_unique_month_lag','hist_new_card1_sum_category_1',
                   'hist_card1_unique_month','hist_card_sum_category_1','hist_card_last_woy','hist_new_card_unique_month_lag',
                   'hist_new_card_last_day','hist_new_card_last_doy','new_card_mean_month_lag']
vector_feature = []
continous_feature = ['hist_card0_mean_month_diff','hist_new_card1_mean_month_diff','hist_card1_mean_month_diff','new_card_max_purchase_amount',
                    'hist_card_mean_month_diff','sum_category_1_1_purchase_amount','new_card_std_purchase_amount','new_max_month_lag_2_purchase_amount',
                    'new_card_mean_purchase_amount','new_max_month_lag_1_purchase_amount','new_mean_month_lag_2_purchase_amount',
                    'new_min_month_lag_2_purchase_amount']  
# df_ffm = df[one_hot_feature + vector_feature + continous_feature]

# ffm_fmt = FFMFormat(vector_feature, one_hot_feature, continous_feature)
# user_ffm = ffm_fmt.fit_transform(df_ffm)
# user_ffm.to_csv('../ffm/ffm.csv', index=False)


# outlier
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1

n_splits= 5

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4590)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['outliers'])):
  
    train_x, train_y = train_df[one_hot_feature + vector_feature + continous_feature].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[one_hot_feature + vector_feature + continous_feature].iloc[valid_idx], train_df['target'].iloc[valid_idx]  
    print ('train_x shape:' + str(train_x.shape))
    print ('valid_x shape:' + str(valid_x.shape))
    
    print ('transform to train ffm')
    ffm_fmt = FFMFormat(vector_feature, one_hot_feature, continous_feature)
    train_x_ffm = ffm_fmt.fit_transform(train_x)
    train_x_ffm.to_csv('../ffm/train_x_' + str(n_fold) + '_ffm.csv', index=False)
    train_y.to_csv('../ffm/train_y_' + str(n_fold) + '_ffm.csv', index=False)

    print ('merge train ffm label')
    train_y = np.loadtxt('../ffm/train_y_' + str(n_fold) + '_ffm.csv')
    with open('../ffm/train_x_' + str(n_fold) + '_ffm.csv') as f_in:
        f_train_out = open('../ffm/train_' + str(n_fold) + '_ffm.csv', 'w')
        for (i, line) in enumerate(f_in):
            f_train_out.write(str(train_y[i]) + ' ' + line)
            
    print ('transform to valid ffm')
    valid_x_ffm = ffm_fmt.fit_transform(valid_x)
    valid_x_ffm.to_csv('../ffm/valid_x_' + str(n_fold) + '_ffm.csv', index=False)
    valid_y.to_csv('../ffm/valid_y_' + str(n_fold) + '_ffm.csv', index=False)
        
    print ('merge valid ffm label')
    valid_y = np.loadtxt('../ffm/valid_y_' + str(n_fold) + '_ffm.csv')
    with open('../ffm/valid_x_' + str(n_fold) + '_ffm.csv') as f_in:
        f_valid_out = open('../ffm/valid_' + str(n_fold) + '_ffm.csv', 'w')
        for (i, line) in enumerate(f_in):
            f_valid_out.write(str(valid_y[i]) + ' ' + line) 
            
    print ('transform to test ffm')
    test_ffm = ffm_fmt.fit_transform(test_df[one_hot_feature + vector_feature + continous_feature])
    test_ffm.to_csv('../ffm/test_' + str(n_fold) + '_ffm.csv', index=False)
 
 
import xlearn as xl

ffm_model = xl.create_ffm()
ffm_model.setTrain('../input/train_ffm.csv')
ffm_model.setTest('../input/test_ffm.csv')

#ffm_model.disableEarlyStop()


# In[ ]:


param = {
    'task': 'reg',
    'lr': 0.01,
    'lambda': 0.001,
    'metric': 'rmse',
    #'opt': 'ftrl',
    'epoch': 7,  # 5
    'k': 4,
    'alpha': 1.5,
    'beta': 0.01,
    'lambda_1': 0.0,
    'lambda_2': 0.0
}




ffm_model.cv(param)



ffm_model.fit(param, '../input/ffm.out')

ffm_model.predict('../input/ffm.out', '../input/ffm.txt')




sub = test[['card_id']]
sub['pred'] = np.loadtxt('../input/ffm.txt') 
