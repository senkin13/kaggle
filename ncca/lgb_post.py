%%time
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings 
warnings.filterwarnings('ignore')

df_seed = pd.read_csv('../input/womens/WNCAATourneySeeds.csv')
df_tour_detailed = pd.read_csv('../input/womens/WNCAATourneyDetailedResults.csv')
df_regular_detailed = pd.read_csv('../input/womens/WRegularSeasonDetailedResults.csv')
sub = pd.read_csv('../input/womens/WSampleSubmissionStage2.csv')
sub['Season'] = sub['ID'].apply(lambda x: x.split('_')[0]).astype(int)
sub['T1_TeamID'] = sub['ID'].apply(lambda x: x.split('_')[1]).astype(int)
sub['T2_TeamID'] = sub['ID'].apply(lambda x: x.split('_')[2]).astype(int)
sub = sub.drop(['Pred'],axis=1)

print ('df_seed',df_seed.shape)
print ('df_tour_detailed',df_tour_detailed.shape)
print ('df_regular_detailed',df_regular_detailed.shape)
print ('sub',sub.shape)

def prepare_data(df):
    df['target'] = 1
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 
    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
    'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]
    dfswap['target'] = 0
    
    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'    
      
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output.loc[output.location=='N','location'] = '0'
    output.loc[output.location=='H','location'] = '1'
    output.loc[output.location=='A','location'] = '-1'
    output.location = output.location.astype(int)
    
    output['Point_Diff'] = output['T1_Score'] - output['T2_Score']
    output['ID'] = output.apply(lambda r: '_'.join([r['Season'].astype(str),r['T1_TeamID'].astype(str),r['T2_TeamID'].astype(str)]), axis=1)
    
    return output

df_regular_detailed = prepare_data(df_regular_detailed)
df_tour_detailed = prepare_data(df_tour_detailed)

print ('Win and Lost df_regular_detailed', df_regular_detailed.shape)
print ('Win and Lost df_tour_detailed', df_tour_detailed.shape)

# sort 
df_tour_detailed = df_tour_detailed.sort_values(by=['Season', 'DayNum'], ascending=True)
df_tour_detailed = pd.concat([df_tour_detailed, sub],axis=0)

# seed
df_seed['seed'] = df_seed['Seed'].apply(lambda x: int(x[1:3]))
seeds_T1 = df_seed[['Season','TeamID','seed']].copy()
seeds_T2 = df_seed[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed']
df_tour_detailed = pd.merge(df_tour_detailed, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
df_tour_detailed = pd.merge(df_tour_detailed, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')
df_tour_detailed["Seed_diff"] = df_tour_detailed["T1_seed"] - df_tour_detailed["T2_seed"]

%%time
df_regular_detailed['T1_Target'] = df_regular_detailed['target']
df_regular_detailed['T2_Target'] = df_regular_detailed['target']

df_regular_detailed['T1_Point_Diff'] = df_regular_detailed['Point_Diff']
df_regular_detailed['T2_Point_Diff'] = df_regular_detailed['Point_Diff']

df_regular_detailed['T1_Score_Allowed'] = df_regular_detailed['T2_Score']
df_regular_detailed['T2_Score_Allowed'] = df_regular_detailed['T1_Score']

###################Regular Same Season #####################
# regular mean
for target in ['T1_Score','T1_Score_Allowed','T1_FGM','T1_FGA','T1_FGM3','T1_FGA3','T1_FTM','T1_FTA','T1_OR','T1_DR','T1_Ast','T1_TO','T1_Stl','T1_Blk','T1_PF',]:
    df_regular_id1_mean = df_regular_detailed.groupby(['Season','T1_TeamID'])[target].mean().reset_index().rename(columns={target:'Regular_Mean_' + str(target)})
    df_tour_detailed = df_tour_detailed.merge(df_regular_id1_mean,on=['Season','T1_TeamID'],how='left')

for target in ['T2_Score','T2_Score_Allowed','T2_FGM','T2_FGA','T2_FGM3','T2_FGA3','T2_FTM','T2_FTA','T2_OR','T2_DR','T2_Ast','T2_TO','T2_Stl','T2_Blk','T2_PF',]:        
    df_regular_id2_mean = df_regular_detailed.groupby(['Season','T2_TeamID'])[target].mean().reset_index().rename(columns={target:'Regular_Mean_' + str(target)})
    df_tour_detailed = df_tour_detailed.merge(df_regular_id2_mean,on=['Season','T2_TeamID'],how='left')

# regular sum
for target in ['T1_Target','T1_Point_Diff']:
    df_regular_id1_sum = df_regular_detailed.groupby(['Season','T1_TeamID'])[target].sum().reset_index().rename(columns={target:'Regular_Sum_' + str(target)})
    df_tour_detailed = df_tour_detailed.merge(df_regular_id1_sum,on=['Season','T1_TeamID'],how='left')

for target in ['T2_Target','T2_Point_Diff']:        
    df_regular_id2_sum = df_regular_detailed.groupby(['Season','T2_TeamID'])[target].sum().reset_index().rename(columns={target:'Regular_Sum_' + str(target)})
    df_tour_detailed = df_tour_detailed.merge(df_regular_id2_sum,on=['Season','T2_TeamID'],how='left')

# diff    
df_tour_detailed['Regular_Mean_Score_Allowed_Diff'] = df_tour_detailed['Regular_Mean_T1_Score_Allowed'] - df_tour_detailed['Regular_Mean_T2_Score_Allowed']
df_tour_detailed['Regular_Mean_Score_Diff'] = df_tour_detailed['Regular_Mean_T1_Score'] - df_tour_detailed['Regular_Mean_T2_Score']
df_tour_detailed['Regular_Mean_FGM_Diff'] = df_tour_detailed['Regular_Mean_T1_FGM'] - df_tour_detailed['Regular_Mean_T2_FGM']
df_tour_detailed['Regular_Mean_FGA_Diff'] = df_tour_detailed['Regular_Mean_T1_FGA'] - df_tour_detailed['Regular_Mean_T2_FGA']
df_tour_detailed['Regular_Mean_FGM3_Diff'] = df_tour_detailed['Regular_Mean_T1_FGM3'] - df_tour_detailed['Regular_Mean_T2_FGM3']
df_tour_detailed['Regular_Mean_FGA3_Diff'] = df_tour_detailed['Regular_Mean_T1_FGA3'] - df_tour_detailed['Regular_Mean_T2_FGA3']
df_tour_detailed['Regular_Mean_FTM_Diff'] = df_tour_detailed['Regular_Mean_T1_FTM'] - df_tour_detailed['Regular_Mean_T2_FTM']
df_tour_detailed['Regular_Mean_FTA_Diff'] = df_tour_detailed['Regular_Mean_T1_FTA'] - df_tour_detailed['Regular_Mean_T2_FTA']
df_tour_detailed['Regular_Mean_OR_Diff'] = df_tour_detailed['Regular_Mean_T1_OR'] - df_tour_detailed['Regular_Mean_T2_OR']
df_tour_detailed['Regular_Mean_DR_Diff'] = df_tour_detailed['Regular_Mean_T1_DR'] - df_tour_detailed['Regular_Mean_T2_DR']
df_tour_detailed['Regular_Mean_Ast_Diff'] = df_tour_detailed['Regular_Mean_T1_Ast'] - df_tour_detailed['Regular_Mean_T2_Ast']
df_tour_detailed['Regular_Mean_TO_Diff'] = df_tour_detailed['Regular_Mean_T1_TO'] - df_tour_detailed['Regular_Mean_T2_TO']
df_tour_detailed['Regular_Mean_Stl_Diff'] = df_tour_detailed['Regular_Mean_T1_Stl'] - df_tour_detailed['Regular_Mean_T2_Stl']
df_tour_detailed['Regular_Mean_Blk_Diff'] = df_tour_detailed['Regular_Mean_T1_Blk'] - df_tour_detailed['Regular_Mean_T2_Blk']
df_tour_detailed['Regular_Mean_PF_Diff'] = df_tour_detailed['Regular_Mean_T1_PF'] - df_tour_detailed['Regular_Mean_T2_PF']


df_tour_detailed['Regular_Sum_Target_Diff'] = df_tour_detailed['Regular_Sum_T1_Target'] - df_tour_detailed['Regular_Sum_T2_Target']
df_tour_detailed['Regular_Sum_Point_Diff_Diff'] = df_tour_detailed['Regular_Sum_T1_Point_Diff'] - df_tour_detailed['Regular_Sum_T2_Point_Diff']    

# ratio
df_tour_detailed['Regular_Mean_Score_Ratio'] = df_tour_detailed['Regular_Mean_T1_Score'] / (df_tour_detailed['Regular_Mean_T2_Score'] + 1)
df_tour_detailed['Regular_Mean_FGM_Ratio'] = df_tour_detailed['Regular_Mean_T1_FGM'] / (df_tour_detailed['Regular_Mean_T2_FGM'] + 1)
df_tour_detailed['Regular_Mean_FGA_Ratio'] = df_tour_detailed['Regular_Mean_T1_FGA'] / (df_tour_detailed['Regular_Mean_T2_FGA'] + 1)
df_tour_detailed['Regular_Mean_FGM3_Ratio'] = df_tour_detailed['Regular_Mean_T1_FGM3'] / (df_tour_detailed['Regular_Mean_T2_FGM3'] + 1)
df_tour_detailed['Regular_Mean_FGA3_Ratio'] = df_tour_detailed['Regular_Mean_T1_FGA3'] / (df_tour_detailed['Regular_Mean_T2_FGA3'] + 1)
df_tour_detailed['Regular_Mean_FTM_Ratio'] = df_tour_detailed['Regular_Mean_T1_FTM'] / (df_tour_detailed['Regular_Mean_T2_FTM'] + 1)
df_tour_detailed['Regular_Mean_FTA_Ratio'] = df_tour_detailed['Regular_Mean_T1_FTA'] / (df_tour_detailed['Regular_Mean_T2_FTA'] + 1)
df_tour_detailed['Regular_Mean_OR_Ratio'] = df_tour_detailed['Regular_Mean_T1_OR'] / (df_tour_detailed['Regular_Mean_T2_OR'] + 1)
df_tour_detailed['Regular_Mean_DR_Ratio'] = df_tour_detailed['Regular_Mean_T1_DR'] / (df_tour_detailed['Regular_Mean_T2_DR'] + 1)
df_tour_detailed['Regular_Mean_Ast_Ratio'] = df_tour_detailed['Regular_Mean_T1_Ast'] / (df_tour_detailed['Regular_Mean_T2_Ast'] + 1)
df_tour_detailed['Regular_Mean_TO_Ratio'] = df_tour_detailed['Regular_Mean_T1_TO'] / (df_tour_detailed['Regular_Mean_T2_TO'] + 1)
df_tour_detailed['Regular_Mean_Stl_Ratio'] = df_tour_detailed['Regular_Mean_T1_Stl'] / (df_tour_detailed['Regular_Mean_T2_Stl'] + 1)
df_tour_detailed['Regular_Mean_Blk_Ratio'] = df_tour_detailed['Regular_Mean_T1_Blk'] / (df_tour_detailed['Regular_Mean_T2_Blk'] + 1)
df_tour_detailed['Regular_Mean_PF_Ratio'] = df_tour_detailed['Regular_Mean_T1_PF'] / (df_tour_detailed['Regular_Mean_T2_PF'] + 1)

df_tour_detailed['Regular_Sum_Target_Ratio'] = df_tour_detailed['Regular_Sum_T1_Target'] / (df_tour_detailed['Regular_Sum_T2_Target'] + 1)

###################Regular Same Season  Last One Match#####################
df_regular_last1 = df_regular_detailed.groupby(['Season','T1_TeamID'])['DayNum'].nlargest(1).reset_index()[['Season','T1_TeamID','DayNum']]
df_regular_last1 = df_regular_last1.merge(df_regular_detailed,on=['Season','T1_TeamID','DayNum'],how='left')

# regular sum
for target in ['T1_Target',]:
    df_regular_id1_sum = df_regular_last1.groupby(['Season','T1_TeamID'])[target].sum().reset_index().rename(columns={target:'Regular_Sum_last1_' + str(target)})
    df_tour_detailed = df_tour_detailed.merge(df_regular_id1_sum,on=['Season','T1_TeamID'],how='left')

for target in ['T2_Target',]:        
    df_regular_id2_sum = df_regular_last1.groupby(['Season','T2_TeamID'])[target].mean().reset_index().rename(columns={target:'Regular_Sum_last1_' + str(target)})
    df_tour_detailed = df_tour_detailed.merge(df_regular_id2_sum,on=['Season','T2_TeamID'],how='left')
    
df_tour_detailed['Regular_Sum_Target_last1_Diff'] = df_tour_detailed['Regular_Sum_last1_T1_Target'] - df_tour_detailed['Regular_Sum_last1_T2_Target']

df_tour_detailed['Regular_Sum_Target_last1_Ratio'] = df_tour_detailed['Regular_Sum_last1_T1_Target'] / (df_tour_detailed['Regular_Sum_last1_T2_Target'] + 1)

# ###################Regular Same Season  Last Three Match#####################
# df_regular_last3 = df_regular_detailed.groupby(['Season','T1_TeamID'])['DayNum'].nlargest(10).reset_index()[['Season','T1_TeamID','DayNum']]
# df_regular_last3 = df_regular_last3.merge(df_regular_detailed,on=['Season','T1_TeamID','DayNum'],how='left')

# # regular mean
# for target in ['T1_Score','T1_FGM','T1_FGA','T1_FGM3','T1_FGA3','T1_FTM','T1_FTA','T1_OR','T1_DR','T1_Ast','T1_TO','T1_Stl','T1_Blk','T1_PF',]:
#     df_regular_id1_mean = df_regular_last3.groupby(['Season','T1_TeamID'])[target].mean().reset_index().rename(columns={target:'Regular_Mean_last3_' + str(target)})
#     df_tour_detailed = df_tour_detailed.merge(df_regular_id1_mean,on=['Season','T1_TeamID'],how='left')

# for target in ['T2_Score','T2_FGM','T2_FGA','T2_FGM3','T2_FGA3','T2_FTM','T2_FTA','T2_OR','T2_DR','T2_Ast','T2_TO','T2_Stl','T2_Blk','T2_PF',]:        
#     df_regular_id2_mean = df_regular_last3.groupby(['Season','T2_TeamID'])[target].mean().reset_index().rename(columns={target:'Regular_Mean_last3_' + str(target)})
#     df_tour_detailed = df_tour_detailed.merge(df_regular_id2_mean,on=['Season','T2_TeamID'],how='left')

# # regular sum
# for target in ['T1_Target']:
#     df_regular_id1_sum = df_regular_last3.groupby(['Season','T1_TeamID'])[target].sum().reset_index().rename(columns={target:'Regular_Sum_last3_' + str(target)})
#     df_tour_detailed = df_tour_detailed.merge(df_regular_id1_sum,on=['Season','T1_TeamID'],how='left')

# for target in ['T2_Target']:        
#     df_regular_id2_sum = df_regular_last3.groupby(['Season','T2_TeamID'])[target].mean().reset_index().rename(columns={target:'Regular_Sum_last3_' + str(target)})
#     df_tour_detailed = df_tour_detailed.merge(df_regular_id2_sum,on=['Season','T2_TeamID'],how='left')
    
# df_tour_detailed['Regular_Mean_Score_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_Score'] - df_tour_detailed['Regular_Mean_last3_T2_Score']
# df_tour_detailed['Regular_Mean_FGM_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_FGM'] - df_tour_detailed['Regular_Mean_last3_T2_FGM']
# df_tour_detailed['Regular_Mean_FGA_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_FGA'] - df_tour_detailed['Regular_Mean_last3_T2_FGA']
# df_tour_detailed['Regular_Mean_FGM3_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_FGM3'] - df_tour_detailed['Regular_Mean_last3_T2_FGM3']
# df_tour_detailed['Regular_Mean_FGA3_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_FGA3'] - df_tour_detailed['Regular_Mean_last3_T2_FGA3']
# df_tour_detailed['Regular_Mean_FTM_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_FTM'] - df_tour_detailed['Regular_Mean_last3_T2_FTM']
# df_tour_detailed['Regular_Mean_FTA_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_FTA'] - df_tour_detailed['Regular_Mean_last3_T2_FTA']
# df_tour_detailed['Regular_Mean_OR_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_OR'] - df_tour_detailed['Regular_Mean_last3_T2_OR']
# df_tour_detailed['Regular_Mean_DR_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_DR'] - df_tour_detailed['Regular_Mean_last3_T2_DR']
# df_tour_detailed['Regular_Mean_Ast_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_Ast'] - df_tour_detailed['Regular_Mean_last3_T2_Ast']
# df_tour_detailed['Regular_Mean_TO_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_TO'] - df_tour_detailed['Regular_Mean_last3_T2_TO']
# df_tour_detailed['Regular_Mean_Stl_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_Stl'] - df_tour_detailed['Regular_Mean_last3_T2_Stl']
# df_tour_detailed['Regular_Mean_Blk_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_Blk'] - df_tour_detailed['Regular_Mean_last3_T2_Blk']
# df_tour_detailed['Regular_Mean_PF_last3_Diff'] = df_tour_detailed['Regular_Mean_last3_T1_PF'] - df_tour_detailed['Regular_Mean_last3_T2_PF']

# df_tour_detailed['Regular_Sum_Target_last3_Diff'] = df_tour_detailed['Regular_Sum_last3_T1_Target'] - df_tour_detailed['Regular_Sum_last3_T2_Target']


# ###################All Season#####################
# # regular mean
# for target in ['T1_Score','T1_FGM','T1_FGA','T1_FGM3','T1_FGA3','T1_FTM','T1_FTA','T1_OR','T1_DR','T1_Ast','T1_TO','T1_Stl','T1_Blk','T1_PF',]:
#     df_regular_id1_mean = df_regular_detailed.groupby(['T1_TeamID'])[target].mean().reset_index().rename(columns={target:'Regular_All_Mean_' + str(target)})
#     df_tour_detailed = df_tour_detailed.merge(df_regular_id1_mean,on=['T1_TeamID'],how='left')

# for target in ['T2_Score','T2_FGM','T2_FGA','T2_FGM3','T2_FGA3','T2_FTM','T2_FTA','T2_OR','T2_DR','T2_Ast','T2_TO','T2_Stl','T2_Blk','T2_PF',]:        
#     df_regular_id2_mean = df_regular_detailed.groupby(['T2_TeamID'])[target].mean().reset_index().rename(columns={target:'Regular_All_Mean_' + str(target)})
#     df_tour_detailed = df_tour_detailed.merge(df_regular_id2_mean,on=['T2_TeamID'],how='left')

# # regular sum
# for target in ['T1_Target',]:
#     df_regular_id1_sum = df_regular_detailed.groupby(['T1_TeamID'])[target].sum().reset_index().rename(columns={target:'Regular_All_Sum_' + str(target)})
#     df_tour_detailed = df_tour_detailed.merge(df_regular_id1_sum,on=['T1_TeamID'],how='left')

# for target in ['T2_Target',]:        
#     df_regular_id2_sum = df_regular_detailed.groupby(['T2_TeamID'])[target].mean().reset_index().rename(columns={target:'Regular_All_Sum_' + str(target)})
#     df_tour_detailed = df_tour_detailed.merge(df_regular_id2_sum,on=['T2_TeamID'],how='left')
    
# df_tour_detailed['Regular_All_Mean_Score_Diff'] = df_tour_detailed['Regular_All_Mean_T1_Score'] - df_tour_detailed['Regular_All_Mean_T2_Score']
# df_tour_detailed['Regular_All_Mean_FGM_Diff'] = df_tour_detailed['Regular_All_Mean_T1_FGM'] - df_tour_detailed['Regular_All_Mean_T2_FGM']
# df_tour_detailed['Regular_All_Mean_FGA_Diff'] = df_tour_detailed['Regular_All_Mean_T1_FGA'] - df_tour_detailed['Regular_All_Mean_T2_FGA']
# df_tour_detailed['Regular_All_Mean_FGM3_Diff'] = df_tour_detailed['Regular_All_Mean_T1_FGM3'] - df_tour_detailed['Regular_All_Mean_T2_FGM3']
# df_tour_detailed['Regular_All_Mean_FGA3_Diff'] = df_tour_detailed['Regular_All_Mean_T1_FGA3'] - df_tour_detailed['Regular_All_Mean_T2_FGA3']
# df_tour_detailed['Regular_All_Mean_FTM_Diff'] = df_tour_detailed['Regular_All_Mean_T1_FTM'] - df_tour_detailed['Regular_All_Mean_T2_FTM']
# df_tour_detailed['Regular_All_Mean_FTA_Diff'] = df_tour_detailed['Regular_All_Mean_T1_FTA'] - df_tour_detailed['Regular_All_Mean_T2_FTA']
# df_tour_detailed['Regular_All_Mean_OR_Diff'] = df_tour_detailed['Regular_All_Mean_T1_OR'] - df_tour_detailed['Regular_All_Mean_T2_OR']
# df_tour_detailed['Regular_All_Mean_DR_Diff'] = df_tour_detailed['Regular_All_Mean_T1_DR'] - df_tour_detailed['Regular_All_Mean_T2_DR']
# df_tour_detailed['Regular_All_Mean_Ast_Diff'] = df_tour_detailed['Regular_All_Mean_T1_Ast'] - df_tour_detailed['Regular_All_Mean_T2_Ast']
# df_tour_detailed['Regular_All_Mean_TO_Diff'] = df_tour_detailed['Regular_All_Mean_T1_TO'] - df_tour_detailed['Regular_All_Mean_T2_TO']
# df_tour_detailed['Regular_All_Mean_Stl_Diff'] = df_tour_detailed['Regular_All_Mean_T1_Stl'] - df_tour_detailed['Regular_All_Mean_T2_Stl']
# df_tour_detailed['Regular_All_Mean_Blk_Diff'] = df_tour_detailed['Regular_All_Mean_T1_Blk'] - df_tour_detailed['Regular_All_Mean_T2_Blk']
# df_tour_detailed['Regular_All_Mean_PF_Diff'] = df_tour_detailed['Regular_All_Mean_T1_PF'] - df_tour_detailed['Regular_All_Mean_T2_PF']

# df_tour_detailed['Regular_All_Sum_Target_Diff'] = df_tour_detailed['Regular_All_Sum_T1_Target'] - df_tour_detailed['Regular_All_Sum_T2_Target']

### %%time
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings 
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold,RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error,log_loss,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5


# train and test
#train_df = df_tour_detailed[(df_tour_detailed['Season']<2018) & (df_tour_detailed['Season']>2010)]

train_df = df_tour_detailed[(df_tour_detailed['Season']<2019)]
test_df = df_tour_detailed[df_tour_detailed['Season']==2019]
print ('train_df', train_df.shape)
print ('test_df', test_df.shape)


drop_features=['target',]
feats = [f for f in train_df.columns if f not in drop_features 
        ]

feats = [
# 'Regular_Mean_T1_Score',
#        'Regular_Mean_T1_FGM', 'Regular_Mean_T1_FGA', 'Regular_Mean_T1_FGM3',
#        'Regular_Mean_T1_FGA3', 'Regular_Mean_T1_FTM', 'Regular_Mean_T1_FTA',
#        'Regular_Mean_T1_OR', 'Regular_Mean_T1_DR', 'Regular_Mean_T1_Ast',
#        'Regular_Mean_T1_TO', 'Regular_Mean_T1_Stl', 'Regular_Mean_T1_Blk',
#        'Regular_Mean_T1_PF', 'Regular_Mean_T2_Point_Diff',
#        'Regular_Mean_T2_Score', 'Regular_Mean_T2_FGM', 'Regular_Mean_T2_FGA',
#        'Regular_Mean_T2_FGM3', 'Regular_Mean_T2_FGA3', 'Regular_Mean_T2_FTM',
#        'Regular_Mean_T2_FTA', 'Regular_Mean_T2_OR', 'Regular_Mean_T2_DR',
#        'Regular_Mean_T2_Ast', 'Regular_Mean_T2_TO', 'Regular_Mean_T2_Stl',
#        'Regular_Mean_T2_Blk', 'Regular_Mean_T2_PF', 
# ------------- tour --------------    
'Seed_diff',#

# ------------- regular --------------     
 #   'Regular_Mean_Score_Allowed_Diff',
    'Regular_Mean_Score_Diff',
       'Regular_Mean_FGM_Diff', 
    'Regular_Mean_FGA_Diff',
       'Regular_Mean_FGM3_Diff', 
    'Regular_Mean_FGA3_Diff',
       'Regular_Mean_FTM_Diff', 
    'Regular_Mean_FTA_Diff',
       'Regular_Mean_OR_Diff', 
    'Regular_Mean_Ast_Diff',
       'Regular_Mean_TO_Diff',
    'Regular_Mean_Stl_Diff',
       'Regular_Mean_Blk_Diff', 
    'Regular_Mean_PF_Diff',
    'Regular_Sum_Target_Diff',


'Regular_Sum_Target_last1_Diff',
'Regular_Mean_Score_Ratio', 
'Regular_Mean_FGA_Ratio', 
'Regular_Mean_FTM_Ratio',
#'location',
# 'mae_prob',
    
    
]
cat_features = []#'location'

# minmaxscaler
# scaler = MinMaxScaler()
# target_array = train_df['Point_Diff'].values.reshape(-1,1)
# target_array = scaler.fit_transform(target_array)
# train_df['target_new'] = target_array
train_df['target_new'] = train_df['Point_Diff']

n_splits= 10

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=817)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
cv_list = []
print ('feats:' + str(len(feats)))

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
  
    train_x, train_y, train_y_new = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx], train_df['target_new'].iloc[train_idx]
    valid_x, valid_y, valid_y_new = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx], train_df['target_new'].iloc[valid_idx] 
    
    print("Train Index:",train_idx.shape,",Val Index:",valid_idx.shape)

    params = {
               "objective" : "binary", 
               "boosting" : "gbdt", 
               "metric" : "binary_logloss",  
               "max_depth": 8, #8
               "min_data_in_leaf": 70, #70
               "min_gain_to_split": 0.1, 
               "reg_alpha": 0.5, #0.1,
               "reg_lambda": 1, #20
               "num_leaves" : 31, #15
               "max_bin" : 255, #255
               "learning_rate" : 0.01, #0.01
               "subsample" : 1,
               "colsample_bytree" : 0.8, #0.9
    }
    

#     params = {
#                "objective" : "regression", 
#                "boosting" : "gbdt", 
#                "metric" : "mae",  
#                "max_depth": 8, #8
#                "min_data_in_leaf": 70, #70
#                "min_gain_to_split": 0.1, 
#                "reg_alpha": 0.1,
#                "reg_lambda": 20, #20
#                "num_leaves" : 15, #120
#                "max_bin" : 255, #255
#                "learning_rate" : 0.01, #0.01
#                "subsample" : 1,
#                "colsample_bytree" : 0.9, #0.25
#     }
    
    if n_fold >= 0:
        print("Fold:" + str(n_fold))
        dtrain = lgb.Dataset(
            train_x, label=train_y_new,categorical_feature=cat_features)#categorical_feature=cat_features
        dval = lgb.Dataset(
            valid_x, label=valid_y_new, reference=dtrain,categorical_feature=cat_features)
        bst = lgb.train(
            params, dtrain, num_boost_round=10000,
            valid_sets=[dval], early_stopping_rounds=200, verbose_eval=100,)#feval = evalerror
        
        new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
        for item in new_list:
            print (item) 

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        oof_cv = log_loss(valid_y,  oof_preds[valid_idx])
        #oof_cv = mean_absolute_error(valid_y,  oof_preds[valid_idx])
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += bst.predict(test_df[feats], num_iteration=bst.best_iteration) / folds.n_splits # test_df_new

cv = log_loss(train_df['target'],  oof_preds)
print('Full OOF LOGLOSS %.6f' % cv)  
# cv = mean_absolute_error(train_df['target'],  oof_preds)
# print('Full OOF MAE %.6f' % cv)  

# cv = log_loss(test_df['target'],  sub_preds)
# print('Test LOGLOSS %.6f' % cv)  
# cv = mean_absolute_error(test_df['target'],  sub_preds)
# print('Test MAE %.6f' % cv)  

#oof_df = pd.DataFrame()
# train_df['mae_prob'] = oof_preds
#oof_df[['card_id','target']].to_csv('../ensemble/womens_mae_oof_' + str(cv) + '.csv',index=False)

# test_df['mae_prob'] = sub_preds
#test_df[['card_id','target']].to_csv('../ensemble/womens_mae_pred_' + str(cv) + '.csv',index=False)


#### post
%%time

start = 0
end = 1
oof_preds_post = np.clip(oof_preds,start,end)
sub_preds_post = np.clip(sub_preds,start,end)


train_df['Pred'] = oof_preds_post

test_df['Pred'] = sub_preds_post

train_df.loc[(train_df.T1_seed==1) & (train_df.T2_seed==16), 'Pred'] = 1.0
test_df.loc[(test_df.T1_seed==1) & (test_df.T2_seed==16), 'Pred'] = 1.0

train_df.loc[(train_df.T1_seed==2) & (train_df.T2_seed==15), 'Pred'] = 1.0
test_df.loc[(test_df.T1_seed==2) & (test_df.T2_seed==15), 'Pred'] = 1.0

train_df.loc[(train_df.T1_seed==3) & (train_df.T2_seed==14), 'Pred'] = 1.0
test_df.loc[(test_df.T1_seed==3) & (test_df.T2_seed==14), 'Pred'] = 1.0

train_df.loc[(train_df.T1_seed==4) & (train_df.T2_seed==13), 'Pred'] = 1.0
test_df.loc[(test_df.T1_seed==4) & (test_df.T2_seed==13), 'Pred'] = 1.0

train_df.loc[(train_df.T1_seed==16) & (train_df.T2_seed==1), 'Pred'] = 0.0
test_df.loc[(test_df.T1_seed==16) & (test_df.T2_seed==1), 'Pred'] = 0.0

train_df.loc[(train_df.T1_seed==15) & (train_df.T2_seed==2), 'Pred'] = 0.0
test_df.loc[(test_df.T1_seed==15) & (test_df.T2_seed==2), 'Pred'] = 0.0

train_df.loc[(train_df.T1_seed==14) & (train_df.T2_seed==3), 'Pred'] = 0.0
test_df.loc[(test_df.T1_seed==14) & (test_df.T2_seed==3), 'Pred'] = 0.0

train_df.loc[(train_df.T1_seed==13) & (train_df.T2_seed==4), 'Pred'] = 0.0
test_df.loc[(test_df.T1_seed==13) & (test_df.T2_seed==4), 'Pred'] = 0.0
    
cv = log_loss(train_df['target'],  train_df['Pred'] )
print('Full OOF LOGLOSS %.6f' % cv)      

