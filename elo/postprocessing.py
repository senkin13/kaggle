import pandas as pd
best_submission =  pd.read_csv('../ensemble/lgb_v2_pl_pred_3.629183834985668.csv')
#../ensemble/lgb_v5_pred_3.6360141831208637.csv' post_new_3.674.csv
                              #../ensemble/lgb_pred_3.6335657657465226.csv'
# ../model/post_40000outliers.csv
                              #baseline_3.6370789515282826.csv') #lgb_pred_3.635678930826137.csv #lgb_pred_3.305631695951106.csv
model_without_outliers = pd.read_csv('../ensemble/pred_1.5499681539479533.csv') #
#base2 =  pd.read_csv('../ensemble/lgb_pred_3.6360118167957447.csv')
#df_outlier_prob = pd.read_csv('../ensemble/binary_pred_0.9052140026974451.csv')

pd.set_option('max_rows',200)
best_submission[['card_id','target']].sort_values(by=['target'],ascending=True).head(200).reset_index(drop=True)

best_submission['target'] = best_submission['target'].map(lambda x:10 if x>3 else x)
best_submission[['card_id','target']].to_csv('../model/post_40000outliers_27positive.csv',index=False)

pd.set_option('max_rows',100)
model_without_outliers[['card_id','target']].sort_values(by=['target'],ascending=True).head(100).reset_index()

%%time

outlier_id = pd.DataFrame(best_submission.sort_values(by='target',ascending = True).head(40000)[['card_id','target']])

#outlier_id_60000 = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000)['card_id'])
#most_likely_liers = best_submission.merge(outlier_id,how='right')
#most_likely_liers_180 = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(180)['card_id'])
#most_likely_liers_180['target'] = -33.21928095

# for card_id in nooutlier_id['card_id']:
#     best_submission.loc[best_submission["card_id"].isin(nooutlier_id["card_id"].values), "target"] = model_without_outliers[model_without_outliers["card_id"].isin(nooutlier_id["card_id"].values)]["target"]

for card_id in outlier_id['card_id']:
    model_without_outliers.loc[model_without_outliers['card_id']==card_id,'target']\
    = outlier_id.loc[outlier_id['card_id']==card_id,'target'].values    
    
%%time
for id in [
'C_ID_922f9c5ea6', #public post 
'C_ID_5ee52cf9f6', #public post 
'C_ID_877b879853', #public post 
'C_ID_a74b12dcf8',
'C_ID_aae50409e7', #public post 
'C_ID_86ddafb51c', 
'C_ID_ac114ef831', #public post 
'C_ID_bf939aa4e9',
'C_ID_be92f84f5c',
'C_ID_70c457436a',
'C_ID_4299911620',
'C_ID_ed2e21ada2',
'C_ID_a8e502820c',
'C_ID_bced41d837', #public post 
'C_ID_b237ce01cb',
'C_ID_1afd4a6a1f', #public post 
'C_ID_12b8c312ef',
'C_ID_9a9bc5b779',
'C_ID_75519a6d74',
'C_ID_c153fbd69e',
'C_ID_d8f884e305',
'C_ID_77cafbf835', 
'C_ID_3d40705001', #public post 
'C_ID_cde027fde7',
'C_ID_a45d2e9338',
'C_ID_1edbae8171', #public post 
'C_ID_493de71141',
'C_ID_e54aeb08f7', #public post 
#'C_ID_55f33eeda6',#public nopost 
'C_ID_e7f772dfc0', 
'C_ID_a4dfae60b0', 
'C_ID_0c22eb865e', 
'C_ID_761c27a0f2', #public post 
'C_ID_dc60219f6c',
'C_ID_8a93017bd2',
'C_ID_4cdcd7bbb1',
'C_ID_9771b57a38',
'C_ID_b0c41ca140',
'C_ID_151cb0ccbd',
'C_ID_93c6231450',
'C_ID_b789a1772c', #public post 
'C_ID_4cf141070f',
'C_ID_63d38cafcb', #public post 
'C_ID_e7c702e96f',
'C_ID_f06cab397a',
'C_ID_a4ca7a3f5d',
'C_ID_32734f4f7c',
#'C_ID_d2fb662751',   #public nopost 
'C_ID_24fbe1a353',
'C_ID_df36580698',
'C_ID_938dec7a1a', #public post 
#'C_ID_647901f731',    #public nopost 
'C_ID_44c140917d',
'C_ID_944c62886f', #public post 
'C_ID_3a02ca2f57', #public post 
'C_ID_c32d4bfa02',
'C_ID_7d3b40444e',
#'C_ID_92df9b9d44',    #public nopost 
'C_ID_02871a2207',
'C_ID_464246c0a5', #public post 
'C_ID_e94aaeede0',
'C_ID_7f5f950342',
'C_ID_a49dce6d51',
'C_ID_68c05d1e23',
'C_ID_00bf566320', #public post 
'C_ID_c27856d211',
'C_ID_3b5972c942',
'C_ID_a3875ce807', #public post 
'C_ID_84e90acaf9',
'C_ID_3804897561',
'C_ID_26f775a95b',
'C_ID_1c86db6e57', #public post 
'C_ID_3e3bb028d1',
'C_ID_aae5d02fad',
'C_ID_8ed05c0045',
'C_ID_eff57c75a3',
'C_ID_099d9774e2',
'C_ID_126a403be9',
'C_ID_739df4d2ac',
    
                  # 3.656
#binary_old     
'C_ID_671674f319',# private       
'C_ID_767923bdb9',# private    
'C_ID_a7d1cd6b7c',# private
'C_ID_ba513d72b8',#public post
'C_ID_9af169bf4a',# private
'C_ID_aeedecc9ae',# private
                   # 3.651
'C_ID_e8865af4e4',# private
'C_ID_81640e36a1',# private
'C_ID_91cc0c06ca',# private
                   # 3.650
'C_ID_da37839463',  # private
#'C_ID_efbf650295', #public nopost 3.650
'C_ID_c179e7bb20',# private
'C_ID_c38764a32f',# public post 3.653
'C_ID_5bc03fa634',# public post 3.651
                   # 3.649    
    
#binary            # 3.656
'C_ID_99db441486', # private    
'C_ID_74dd007aec', # private
'C_ID_6a5ec7646a', #public post 3.656
#3.654

# new 3.68
# 'C_ID_767923bdb9',# private    
# 'C_ID_3e3bb028d1',# private    
# 'C_ID_8ed05c0045',# private    
# 'C_ID_da88f008d8',# private    
# 'C_ID_a475834914',# private    
# 'C_ID_0be3fb41d2',    
# 'C_ID_d9bf7f6d01',    # one is public nopost
    
# # new 3.674
# 'C_ID_833aa2f7af',
# 'C_ID_73ddacd425',
# 'C_ID_aeedecc9ae',
# 'C_ID_04214ceb9f',
# 'C_ID_360a57f2ac',
# 'C_ID_739df4d2ac',
# 'C_ID_57356326f4',
# 'C_ID_0103766ffb',
# 'C_ID_9620897ad7',
    # 0.007 worse
]:
    best_submission.loc[best_submission['card_id']==id,'target'] = -33.21928095
    
best_submission[['card_id','target']].to_csv('../model/post_pl_fold_v2.csv',index=False)    
