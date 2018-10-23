def target_mean(df_train,train_df,test_df,cols):
    min_samples_leaf=100
    smoothing=10
    noise_level=0.01 #0.01
    for c in tqdm(cols):
        new_feature = '{}_{}'.format('_'.join(c['groupby']), c['func'])
        averages = df_train.groupby(c['groupby'])[['bin']].agg(['mean','count']).bin.reset_index()
        smoothing = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing))
        averages[new_feature] = df_train['bin'].mean() * (1 - smoothing) + averages['mean'] * smoothing
        averages.drop(['mean', 'count'], axis=1, inplace=True)

        np.random.seed(42)
        noise = np.random.randn(len(averages[new_feature])) * noise_level
        averages[new_feature] = averages[new_feature] + noise

        train_df = train_df.merge(averages,on=c['groupby'],how='left')
        test_df = test_df.merge(averages,on=c['groupby'],how='left')
        
    return train_df,test_df    

def woe(df_train,train_df,test_df,cols):
    s = 0.1**8
    for c in tqdm(cols):
        new_feature = '{}_{}'.format('_'.join(c['groupby']), c['func'])
        gp = df_train.groupby(c['groupby'])[['bin']].agg(['count','sum']).bin.reset_index()
        print (gp.columns)
        pos = df_train['bin'].sum()
        neg = len(df_train) - pos
        gp[new_feature] = np.log((gp['sum']/pos)/((gp['count']-gp['sum']+s)/neg)+1)
        gp.drop(['count','sum'],axis=1,inplace=True)
        
        train_df = train_df.merge(gp,on=c['groupby'],how='left')
        test_df = test_df.merge(gp,on=c['groupby'],how='left')
        del gp
        gc.collect()
    return train_df,test_df    

def frequency_encoding(df_train,train_df,test_df,col):
    freq_encoding = df_train.groupby([col]).size()/df_train.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    train_df = train_df.merge(freq_encoding,on=col,how='left')
    test_df = test_df.merge(freq_encoding,on=col,how='left')    
    del freq_encoding
    gc.collect()
    return train_df,test_df  
    
    
    
cat_features=[]

# target encoding
    df_train = train_df.iloc[train_idx]
    train_df_new,test_df_new = train_df,test_df
    for col in tqdm(cat_features):
        cols = [{'groupby': [col], 'func':'targetmean'}]
        train_df_new,test_df_new = target_mean(df_train,train_df_new,test_df_new,cols)


#frequency
    for col in tqdm(cat_features):
        train_df_new,test_df_new = frequency_encoding(df_train,train_df_new,test_df_new,col)

    
    feats = [f for f in train_df_new.columns if f not in drop_features]
    print ('ALL feats:' + str(len(feats)))
    train_x, train_y = train_df_new[feats].iloc[train_idx], train_df_new['totals_transactionRevenue'].iloc[train_idx]
    valid_x, valid_y = train_df_new[feats].iloc[valid_idx], train_df_new['totals_transactionRevenue'].iloc[valid_idx]    
