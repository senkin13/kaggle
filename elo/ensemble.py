%%time
import numpy as np
import pandas as pd
import gc
import time
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import pickle
import time
import datetime
import warnings 
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5


##############train test
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

for df in [train, test]:
    df['first_active_month'].fillna('2017-09',inplace=True)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days

# Submodel
hist_new_oof_unique = pd.read_pickle('../feature/submodel/hist_new_oof_unique.pkl')    
hist_new_pred_unique = pd.read_pickle('../feature/submodel/hist_new_pred_unique.pkl')    
hist_oof_unique = pd.read_pickle('../feature/submodel/hist_oof_unique.pkl')    
hist1_oof_unique = pd.read_pickle('../feature/submodel/hist1_oof_unique.pkl')  
hist_pred_unique = pd.read_pickle('../feature/submodel/hist_pred_unique.pkl')    
new_oof_unique = pd.read_pickle('../feature/submodel/new_oof_unique.pkl')    
new_pred_unique = pd.read_pickle('../feature/submodel/new_pred_unique.pkl') 
hist1_pred_unique = pd.read_pickle('../feature/submodel/hist1_pred_unique.pkl')


# OOF Train
## with submodel
lgb_v2_3628287_oof = pd.read_csv('../4590/lgb_v2_oof_3.6282879167899678_4590.csv').rename(columns={'target':'lgb_v2_3628287'})
lgb_v5_3626548_oof = pd.read_csv('../4590/lgb_v5_oof_3.6265489090754697_4590.csv').rename(columns={'target':'lgb_v5_3626548'})
lgb_v6_3630270_oof = pd.read_csv('../4590/lgb_v6_oof_3.6302703316631306_4590.csv').rename(columns={'target':'lgb_v6_3630270'})
lgb_v8_3630243_oof = pd.read_csv('../4590/lgb_v8_oof_3.6302435290184274_4590.csv').rename(columns={'target':'lgb_v8_3630243'})
lgb_v9_3631594_oof = pd.read_csv('../4590/lgb_v9_oof_3.6315942885200965_4590.csv').rename(columns={'target':'lgb_v9_3631594'})
lgb_v10_3624431_oof = pd.read_csv('../4590/lgb_v10_oof_3.624431417740978_4590.csv').rename(columns={'target':'lgb_v10_3624431'})

cat_v2_3642849_oof = pd.read_csv('../4590/cat_v2_oof_3.642849396739676.csv').rename(columns={'target':'cat_v2_3642849'})
cat_v5_3638180_oof = pd.read_csv('../4590/cat_v5_oof_3.638180386973219_4590.csv').rename(columns={'target':'cat_v5_3638180'})
cat_v6_3641266_oof = pd.read_csv('../4590/cat_v6_oof_3.641266366992786_4590.csv').rename(columns={'target':'cat_v6_3641266'})
cat_v8_3641057_oof = pd.read_csv('../4590/cat_v8_oof_3.6410575908726264.csv').rename(columns={'target':'cat_v8_3641057'})
cat_v9_3644587_oof = pd.read_csv('../4590/cat_df_v9_509_oof_3.6445874094174266_4590.csv').rename(columns={'target':'cat_v9_3644587'})
cat_v10_3639065_oof = pd.read_csv('../4590/cat_df_v10_385_oof_3.6390651119491313_4590.csv').rename(columns={'target':'cat_v10_3639065'})

nn_v2_3652412_oof = pd.read_csv('../4590/nn_v2_oof_3.652412112235396.csv').rename(columns={'target':'nn_v2_3652412'})
nn_v2_3652279_oof = pd.read_csv('../4590/nn_v2_oof_3.6522797239106337.csv').rename(columns={'target':'nn_v2_3652279'})
nn_v2_3651086_oof = pd.read_csv('../4590/nn_v2_oof_3.6510864005102786.csv').rename(columns={'target':'nn_v2_3651086'})
nn_v2_3650883_oof = pd.read_csv('../4590/nn_v2_oof_3.650883053620915.csv').rename(columns={'target':'nn_v2_3650883'})
nn_v2_3650888_oof = pd.read_csv('../4590/nn_v2_oof_3.6508880670201043.csv').rename(columns={'target':'nn_v2_3650888'})

nn_v8_3652610_oof = pd.read_csv('../4590/nn_v8_oof_3.6526105845669505.csv').rename(columns={'target':'nn_v8_3652610'})
nn_v8_3652252_oof = pd.read_csv('../4590/nn_v8_oof_3.652252173250081.csv').rename(columns={'target':'nn_v8_3652252'})
nn_v8_3652129_oof = pd.read_csv('../4590/nn_v8_oof_3.652129818523328.csv').rename(columns={'target':'nn_v8_3652129'})
nn_v8_3651332_oof = pd.read_csv('../4590/nn_v8_oof_3.6513322058576043.csv').rename(columns={'target':'nn_v8_3651332'})
nn_v8_3650987_oof = pd.read_csv('../4590/nn_v8_oof_3.6509872964797325.csv').rename(columns={'target':'nn_v8_3650987'})

nn_v5_3644688_oof = pd.read_csv('../4590/nn_v5_oof_3.6446881838575633.csv').rename(columns={'target':'nn_v5_3644688'})
nn_v5_3647673_oof = pd.read_csv('../4590/nn_v5_oof_3.64767324254722.csv').rename(columns={'target':'nn_v5_3647673'})
nn_v5_3646798_oof = pd.read_csv('../4590/nn_v5_oof_3.646798945617891.csv').rename(columns={'target':'nn_v5_3646798'})
nn_v5_3647910_oof = pd.read_csv('../4590/nn_v5_oof_3.6479107278305096.csv').rename(columns={'target':'nn_v5_3647910'})
nn_v5_3645888_oof = pd.read_csv('../4590/nn_v5_oof_3.6458881166150108.csv').rename(columns={'target':'nn_v5_3645888'})

nn_v6_3651765_oof = pd.read_csv('../4590/nn_v6_oof_3.6517654640310555.csv').rename(columns={'target':'nn_v6_3651765'})
nn_v6_3649837_oof = pd.read_csv('../4590/nn_v6_oof_3.6498377327912155.csv').rename(columns={'target':'nn_v6_3649837'})
nn_v6_3650724_oof = pd.read_csv('../4590/nn_v6_oof_3.6507242738502854.csv').rename(columns={'target':'nn_v6_3650724'})
nn_v6_3651711_oof = pd.read_csv('../4590/nn_v6_oof_3.651711323142337.csv').rename(columns={'target':'nn_v6_3651711'})
nn_v6_3651451_oof = pd.read_csv('../4590/nn_v6_oof_3.651451045656479.csv').rename(columns={'target':'nn_v6_3651451'})

nn_v9_3654152_oof = pd.read_csv('../4590/nn_df_v9_509_oof_3.654152571708067_4590.csv').rename(columns={'target':'nn_v9_3654152'})
nn_v9_3659479_oof = pd.read_csv('../4590/nn_df_v9_509_oof_3.6594799207209_4590.csv').rename(columns={'target':'nn_v9_3659479'})
nn_v9_3658122_oof = pd.read_csv('../4590/nn_df_v9_509_oof_3.6581220929155713_4590.csv').rename(columns={'target':'nn_v9_3658122'})

nn_v10_3640746_oof = pd.read_csv('../4590/nn_df_v10_385_oof_3.6407460362952726_4590.csv').rename(columns={'target':'nn_v10_3640746'})
nn_v10_3641109_oof = pd.read_csv('../4590/nn_df_v10_385_oof_3.6411098416171837_4590.csv').rename(columns={'target':'nn_v10_3641109'})
nn_v10_3642849_oof = pd.read_csv('../4590/nn_df_v10_385_oof_3.642849990385437_4590.csv').rename(columns={'target':'nn_v10_3642849'})
nn_v10_3642431_oof = pd.read_csv('../4590/nn_df_v10_385_oof_3.6424314405061695_4590.csv').rename(columns={'target':'nn_v10_3642431'})
nn_v10_3642371_oof = pd.read_csv('../4590/nn_df_v10_385_oof_3.6423716161259567_4590.csv').rename(columns={'target':'nn_v10_3642371'})
nn_v10_3643881_oof = pd.read_csv('../4590/nn_df_v10_385_oof_3.643881094827809_4590.csv').rename(columns={'target':'nn_v10_3643881'})
nn_v10_3644938_oof = pd.read_csv('../4590/nn_df_v10_385_oof_3.6449385995814785_4590.csv').rename(columns={'target':'nn_v10_3644938'})
nn_v10_3643967_oof = pd.read_csv('../4590/nn_df_v10_385_oof_3.6439673248696964_4590.csv').rename(columns={'target':'nn_v10_3643967'})
nn_v10_3645736_oof = pd.read_csv('../4590/nn_df_v10_385_oof_3.6457368890755766_4590.csv').rename(columns={'target':'nn_v10_3645736'})

## without submodel
lgb_v2_3631178_oof = pd.read_csv('../4590/lgb_v2_oof_3.631178888956938_4590.csv').rename(columns={'target':'lgb_v2nosub_3631178'})
lgb_v5_3631363_oof = pd.read_csv('../4590/lgb_v5_oof_3.63136331543829_4590.csv').rename(columns={'target':'lgb_v5nosub_3631363'})
lgb_v6_3638587_oof = pd.read_csv('../4590/lgb_df_v6_nosubmodel_oof_3.638587902649418_4590.csv').rename(columns={'target':'lgb_v6nosub_3638587'})
lgb_v9_3633760_oof = pd.read_csv('../4590/lgb_v9_oof_3.633760010064895_4590.csv').rename(columns={'target':'lgb_v9nosub_3633760'})
lgb_v8_3633866_oof = pd.read_csv('../4590/lgb_v8nosub_oof_3.6338663316139477_4590.csv').rename(columns={'target':'lgb_v8nosub_3633866'})
lgb_v10_3629696_oof = pd.read_csv('../4590/lgb_v10nosub_oof_3.62969698600354_4590.csv').rename(columns={'target':'lgb_v10nosub_3629696'})

cat_v2_3643534_oof = pd.read_csv('../4590/cat_v2_oof_3.6435343738249855_4590.csv').rename(columns={'target':'cat_v2nosub_3643534'})
cat_v9_3646145_oof = pd.read_csv('../4590/cat_df_v9_nosubmodel_oof_3.646145261639896_4590.csv').rename(columns={'target':'cat_v9nosub_3646145'})
cat_v10_3641476_oof = pd.read_csv('../4590/cat_df_v10_nosubmodel_oof_3.641476799119645_4590.csv').rename(columns={'target':'cat_v10nosub_3641476'})

nn_v2_3655949_oof = pd.read_csv('../4590/nn_v2nosub_oof_3.6559497416233495_4590.csv').rename(columns={'target':'nn_v2nosub_3655949'})
nn_v2_3657138_oof = pd.read_csv('../4590/nn_v2nosub_oof_3.6571385477131586_4590.csv').rename(columns={'target':'nn_v2nosub_3657138'})
nn_v2_3659540_oof = pd.read_csv('../4590/nn_v2nosub_oof_3.659540705101139_4590.csv').rename(columns={'target':'nn_v2nosub_3659540'})
nn_v2_3658049_oof = pd.read_csv('../4590/nn_v2nosub_oof_3.6580493842640154_4590.csv').rename(columns={'target':'nn_v2nosub_3658049'})
nn_v2_3657516_oof = pd.read_csv('../4590/nn_v2nosub_oof_3.6575165286810267_4590.csv').rename(columns={'target':'nn_v2nosub_3657516'})

nn_v5_3657089_oof = pd.read_csv('../4590/nn_v5nosub_oof_3.6570892567256834_4590.csv').rename(columns={'target':'nn_v5nosub_3657089'})
nn_v5_3656940_oof = pd.read_csv('../4590/nn_v5nosub_oof_3.65694046979869_4590.csv').rename(columns={'target':'nn_v5nosub_3656940'})
nn_v5_3656590_oof = pd.read_csv('../4590/nn_v5nosub_oof_3.656590252879022_4590.csv').rename(columns={'target':'nn_v5nosub_3656590'})
nn_v5_3655499_oof = pd.read_csv('../4590/nn_v5nosub_oof_3.6554990108195367_4590.csv').rename(columns={'target':'nn_v5nosub_3655499'})
nn_v5_3655257_oof = pd.read_csv('../4590/nn_v5nosub_oof_3.655257027973143_4590.csv').rename(columns={'target':'nn_v5nosub_3655257'})

nn_v9_3658062_oof = pd.read_csv('../4590/nn_df_v9_nosubmodel_oof_3.658062964530797_4590.csv').rename(columns={'target':'nn_v9nosub_3658062'})
nn_v9_3661352_oof = pd.read_csv('../4590/nn_df_v9_nosubmodel_oof_3.661352350294397_4590.csv').rename(columns={'target':'nn_v9nosub_3661352'})

nn_v10_3650779_oof = pd.read_csv('../4590/nn_df_v10_nosubmodel_oof_3.6507796197816784_4590.csv').rename(columns={'target':'nn_v10nosub_3650779'})
nn_v10_3648925_oof = pd.read_csv('../4590/nn_df_v10_nosubmodel_oof_3.6489253259994427_4590.csv').rename(columns={'target':'nn_v10nosub_3648925'})
nn_v10_3655221_oof = pd.read_csv('../4590/nn_df_v10_nosubmodel_oof_3.6552218419130043_4590.csv').rename(columns={'target':'nn_v10nosub_3655221'})
nn_v10_3652406_oof = pd.read_csv('../4590/nn_df_v10_nosubmodel_oof_3.652406592609308_4590.csv').rename(columns={'target':'nn_v10nosub_3652406'})
nn_v10_3656174_oof = pd.read_csv('../4590/nn_df_v10_nosubmodel_oof_3.6561747638332918_4590.csv').rename(columns={'target':'nn_v10nosub_3656174'})

#--------------------------------------------------------------------------------------#
# OOF Test
## with submodel
lgb_v2_3628287_pred = pd.read_csv('../4590/lgb_v2_pred_3.6282879167899678_4590.csv').rename(columns={'target':'lgb_v2_3628287'})
lgb_v5_3626548_pred = pd.read_csv('../4590/lgb_v5_pred_3.6265489090754697_4590.csv').rename(columns={'target':'lgb_v5_3626548'})
lgb_v6_3630270_pred = pd.read_csv('../4590/lgb_v6_pred_3.6302703316631306_4590.csv').rename(columns={'target':'lgb_v6_3630270'})
lgb_v8_3630243_pred = pd.read_csv('../4590/lgb_v8_pred_3.6302435290184274_4590.csv').rename(columns={'target':'lgb_v8_3630243'})
lgb_v9_3631594_pred = pd.read_csv('../4590/lgb_v9_pred_3.6315942885200965_4590.csv').rename(columns={'target':'lgb_v9_3631594'})
lgb_v10_3624431_pred = pd.read_csv('../4590/lgb_v10_pred_3.624431417740978_4590.csv').rename(columns={'target':'lgb_v10_3624431'})


cat_v2_3642849_pred = pd.read_csv('../4590/cat_v2_pred_3.642849396739676.csv').rename(columns={'target':'cat_v2_3642849'})
cat_v5_3638180_pred = pd.read_csv('../4590/cat_v5_pred_3.638180386973219_4590.csv').rename(columns={'target':'cat_v5_3638180'})
cat_v6_3641266_pred = pd.read_csv('../4590/cat_v6_pred_3.641266366992786_4590.csv').rename(columns={'target':'cat_v6_3641266'})
cat_v8_3641057_pred = pd.read_csv('../4590/cat_v8_pred_3.6410575908726264.csv').rename(columns={'target':'cat_v8_3641057'})
cat_v9_3644587_pred = pd.read_csv('../4590/cat_df_v9_509_pred_3.6445874094174266_4590.csv').rename(columns={'target':'cat_v9_3644587'})
cat_v10_3639065_pred = pd.read_csv('../4590/cat_df_v10_385_pred_3.6390651119491313_4590.csv').rename(columns={'target':'cat_v10_3639065'})


nn_v2_3652412_pred = pd.read_csv('../4590/nn_v2_pred_3.652412112235396.csv').rename(columns={'target':'nn_v2_3652412'})
nn_v2_3652279_pred = pd.read_csv('../4590/nn_v2_pred_3.6522797239106337.csv').rename(columns={'target':'nn_v2_3652279'})
nn_v2_3651086_pred = pd.read_csv('../4590/nn_v2_pred_3.6510864005102786.csv').rename(columns={'target':'nn_v2_3651086'})
nn_v2_3650883_pred = pd.read_csv('../4590/nn_v2_pred_3.650883053620915.csv').rename(columns={'target':'nn_v2_3650883'})
nn_v2_3650888_pred = pd.read_csv('../4590/nn_v2_pred_3.6508880670201043.csv').rename(columns={'target':'nn_v2_3650888'})

nn_v8_3652610_pred = pd.read_csv('../4590/nn_v8_pred_3.6526105845669505.csv').rename(columns={'target':'nn_v8_3652610'})
nn_v8_3652252_pred = pd.read_csv('../4590/nn_v8_pred_3.652252173250081.csv').rename(columns={'target':'nn_v8_3652252'})
nn_v8_3652129_pred = pd.read_csv('../4590/nn_v8_pred_3.652129818523328.csv').rename(columns={'target':'nn_v8_3652129'})
nn_v8_3651332_pred = pd.read_csv('../4590/nn_v8_pred_3.6513322058576043.csv').rename(columns={'target':'nn_v8_3651332'})
nn_v8_3650987_pred = pd.read_csv('../4590/nn_v8_pred_3.6509872964797325.csv').rename(columns={'target':'nn_v8_3650987'})

nn_v5_3644688_pred = pd.read_csv('../4590/nn_v5_pred_3.6446881838575633.csv').rename(columns={'target':'nn_v5_3644688'})
nn_v5_3647673_pred = pd.read_csv('../4590/nn_v5_pred_3.64767324254722.csv').rename(columns={'target':'nn_v5_3647673'})
nn_v5_3646798_pred = pd.read_csv('../4590/nn_v5_pred_3.646798945617891.csv').rename(columns={'target':'nn_v5_3646798'})
nn_v5_3647910_pred = pd.read_csv('../4590/nn_v5_pred_3.6479107278305096.csv').rename(columns={'target':'nn_v5_3647910'})
nn_v5_3645888_pred = pd.read_csv('../4590/nn_v5_pred_3.6458881166150108.csv').rename(columns={'target':'nn_v5_3645888'})

nn_v6_3651765_pred = pd.read_csv('../4590/nn_v6_pred_3.6517654640310555.csv').rename(columns={'target':'nn_v6_3651765'})
nn_v6_3649837_pred = pd.read_csv('../4590/nn_v6_pred_3.6498377327912155.csv').rename(columns={'target':'nn_v6_3649837'})
nn_v6_3650724_pred = pd.read_csv('../4590/nn_v6_pred_3.6507242738502854.csv').rename(columns={'target':'nn_v6_3650724'})
nn_v6_3651711_pred = pd.read_csv('../4590/nn_v6_pred_3.651711323142337.csv').rename(columns={'target':'nn_v6_3651711'})
nn_v6_3651451_pred = pd.read_csv('../4590/nn_v6_pred_3.651451045656479.csv').rename(columns={'target':'nn_v6_3651451'})

nn_v9_3654152_pred = pd.read_csv('../4590/nn_df_v9_509_pred_3.654152571708067_4590.csv').rename(columns={'target':'nn_v9_3654152'})
nn_v9_3659479_pred = pd.read_csv('../4590/nn_df_v9_509_pred_3.6594799207209_4590.csv').rename(columns={'target':'nn_v9_3659479'})
nn_v9_3658122_pred = pd.read_csv('../4590/nn_df_v9_509_pred_3.6581220929155713_4590.csv').rename(columns={'target':'nn_v9_3658122'})

nn_v10_3640746_pred= pd.read_csv('../4590/nn_df_v10_385_pred_3.6407460362952726_4590.csv').rename(columns={'target':'nn_v10_3640746'})
nn_v10_3641109_pred= pd.read_csv('../4590/nn_df_v10_385_pred_3.6411098416171837_4590.csv').rename(columns={'target':'nn_v10_3641109'})
nn_v10_3642849_pred = pd.read_csv('../4590/nn_df_v10_385_pred_3.642849990385437_4590.csv').rename(columns={'target':'nn_v10_3642849'})
nn_v10_3642431_pred = pd.read_csv('../4590/nn_df_v10_385_pred_3.6424314405061695_4590.csv').rename(columns={'target':'nn_v10_3642431'})
nn_v10_3642371_pred = pd.read_csv('../4590/nn_df_v10_385_pred_3.6423716161259567_4590.csv').rename(columns={'target':'nn_v10_3642371'})
nn_v10_3643881_pred = pd.read_csv('../4590/nn_df_v10_385_pred_3.643881094827809_4590.csv').rename(columns={'target':'nn_v10_3643881'})
nn_v10_3644938_pred = pd.read_csv('../4590/nn_df_v10_385_pred_3.6449385995814785_4590.csv').rename(columns={'target':'nn_v10_3644938'})
nn_v10_3643967_pred = pd.read_csv('../4590/nn_df_v10_385_pred_3.6439673248696964_4590.csv').rename(columns={'target':'nn_v10_3643967'})
nn_v10_3645736_pred = pd.read_csv('../4590/nn_df_v10_385_pred_3.6457368890755766_4590.csv').rename(columns={'target':'nn_v10_3645736'})


## without submodel
lgb_v2_3631178_pred = pd.read_csv('../4590/lgb_v2_pred_3.631178888956938_4590.csv').rename(columns={'target':'lgb_v2nosub_3631178'})
lgb_v5_3631363_pred = pd.read_csv('../4590/lgb_v5_pred_3.63136331543829_4590.csv').rename(columns={'target':'lgb_v5nosub_3631363'})
lgb_v6_3638587_pred = pd.read_csv('../4590/lgb_df_v6_nosubmodel_pred_3.638587902649418_4590.csv').rename(columns={'target':'lgb_v6nosub_3638587'})
lgb_v9_3633760_pred = pd.read_csv('../4590/lgb_v9_pred_3.633760010064895_4590.csv').rename(columns={'target':'lgb_v9nosub_3633760'})
lgb_v8_3633866_pred = pd.read_csv('../4590/lgb_v8nosub_pred_3.6338663316139477_4590.csv').rename(columns={'target':'lgb_v8nosub_3633866'})
lgb_v10_3629696_pred = pd.read_csv('../4590/lgb_v10nosub_pred_3.62969698600354_4590.csv').rename(columns={'target':'lgb_v10nosub_3629696'})

cat_v2_3643534_pred = pd.read_csv('../4590/cat_v2_pred_3.6435343738249855_4590.csv').rename(columns={'target':'cat_v2nosub_3643534'})
cat_v9_3646145_pred = pd.read_csv('../4590/cat_df_v9_nosubmodel_pred_3.646145261639896_4590.csv').rename(columns={'target':'cat_v9nosub_3646145'})
cat_v10_3641476_pred = pd.read_csv('../4590/cat_df_v10_nosubmodel_pred_3.641476799119645_4590.csv').rename(columns={'target':'cat_v10nosub_3641476'})

nn_v2_3655949_pred = pd.read_csv('../4590/nn_v2nosub_pred_3.6559497416233495_4590.csv').rename(columns={'target':'nn_v2nosub_3655949'})
nn_v2_3657138_pred = pd.read_csv('../4590/nn_v2nosub_pred_3.6571385477131586_4590.csv').rename(columns={'target':'nn_v2nosub_3657138'})
nn_v2_3659540_pred = pd.read_csv('../4590/nn_v2nosub_pred_3.659540705101139_4590.csv').rename(columns={'target':'nn_v2nosub_3659540'})
nn_v2_3658049_pred = pd.read_csv('../4590/nn_v2nosub_pred_3.6580493842640154_4590.csv').rename(columns={'target':'nn_v2nosub_3658049'})
nn_v2_3657516_pred = pd.read_csv('../4590/nn_v2nosub_pred_3.6575165286810267_4590.csv').rename(columns={'target':'nn_v2nosub_3657516'})

nn_v5_3657089_pred = pd.read_csv('../4590/nn_v5nosub_pred_3.6570892567256834_4590.csv').rename(columns={'target':'nn_v5nosub_3657089'})
nn_v5_3656940_pred = pd.read_csv('../4590/nn_v5nosub_pred_3.65694046979869_4590.csv').rename(columns={'target':'nn_v5nosub_3656940'})
nn_v5_3656590_pred = pd.read_csv('../4590/nn_v5nosub_pred_3.656590252879022_4590.csv').rename(columns={'target':'nn_v5nosub_3656590'})
nn_v5_3655499_pred = pd.read_csv('../4590/nn_v5nosub_pred_3.6554990108195367_4590.csv').rename(columns={'target':'nn_v5nosub_3655499'})
nn_v5_3655257_pred = pd.read_csv('../4590/nn_v5nosub_pred_3.655257027973143_4590.csv').rename(columns={'target':'nn_v5nosub_3655257'})

nn_v9_3658062_pred = pd.read_csv('../4590/nn_df_v9_nosubmodel_pred_3.658062964530797_4590.csv').rename(columns={'target':'nn_v9nosub_3658062'})
nn_v9_3661352_pred = pd.read_csv('../4590/nn_df_v9_nosubmodel_pred_3.661352350294397_4590.csv').rename(columns={'target':'nn_v9nosub_3661352'})

nn_v10_3650779_pred = pd.read_csv('../4590/nn_df_v10_nosubmodel_pred_3.6507796197816784_4590.csv').rename(columns={'target':'nn_v10nosub_3650779'})
nn_v10_3648925_pred = pd.read_csv('../4590/nn_df_v10_nosubmodel_pred_3.6489253259994427_4590.csv').rename(columns={'target':'nn_v10nosub_3648925'})
nn_v10_3655221_pred = pd.read_csv('../4590/nn_df_v10_nosubmodel_pred_3.6552218419130043_4590.csv').rename(columns={'target':'nn_v10nosub_3655221'})
nn_v10_3652406_pred = pd.read_csv('../4590/nn_df_v10_nosubmodel_pred_3.652406592609308_4590.csv').rename(columns={'target':'nn_v10nosub_3652406'})
nn_v10_3656174_pred = pd.read_csv('../4590/nn_df_v10_nosubmodel_pred_3.6561747638332918_4590.csv').rename(columns={'target':'nn_v10nosub_3656174'})

#-------------------------------------------------------------------------------------------------#
# Merge
for v in [nn_v2_3652412_oof, nn_v2_3652279_oof, nn_v2_3651086_oof, nn_v2_3650883_oof, nn_v2_3650888_oof,
         nn_v8_3652610_oof, nn_v8_3652252_oof, nn_v8_3652129_oof, nn_v8_3651332_oof, nn_v8_3650987_oof,
         nn_v5_3644688_oof, nn_v5_3647673_oof, nn_v5_3646798_oof, nn_v5_3647910_oof, nn_v5_3645888_oof,
         nn_v6_3651765_oof, nn_v6_3649837_oof, nn_v6_3650724_oof, nn_v6_3651711_oof, nn_v6_3651451_oof, 
         lgb_v2_3628287_oof, lgb_v5_3626548_oof, lgb_v6_3630270_oof, lgb_v8_3630243_oof,lgb_v9_3631594_oof,lgb_v10_3624431_oof,
         lgb_v5_3631363_oof, lgb_v2_3631178_oof,lgb_v9_3633760_oof,lgb_v8_3633866_oof,lgb_v10_3629696_oof,lgb_v6_3638587_oof,
         cat_v2_3642849_oof, cat_v5_3638180_oof, cat_v6_3641266_oof, cat_v8_3641057_oof,  cat_v9_3644587_oof,cat_v10_3639065_oof,
         cat_v2_3643534_oof, cat_v9_3646145_oof, cat_v10_3641476_oof, 
          
         nn_v2_3655949_oof, nn_v2_3657138_oof, nn_v2_3659540_oof, nn_v2_3658049_oof, nn_v2_3657516_oof,
         nn_v5_3657089_oof, nn_v5_3656940_oof, nn_v5_3656590_oof, nn_v5_3655499_oof, nn_v5_3655257_oof,  
         nn_v10_3640746_oof, nn_v10_3641109_oof, nn_v10_3642849_oof, nn_v10_3642431_oof, nn_v10_3642371_oof,
         nn_v10_3643881_oof,nn_v10_3644938_oof,nn_v10_3643967_oof,nn_v10_3645736_oof, 
         nn_v10_3650779_oof, nn_v10_3648925_oof, nn_v10_3655221_oof, nn_v10_3652406_oof, nn_v10_3656174_oof,   
         nn_v9_3654152_oof,nn_v9_3659479_oof,nn_v9_3658122_oof,nn_v9_3658062_oof,nn_v9_3661352_oof,  
         ]:
    train = train.merge(v,on='card_id',how='left')

for v in [nn_v2_3652412_pred, nn_v2_3652279_pred, nn_v2_3651086_pred, nn_v2_3650883_pred, nn_v2_3650888_pred,
         nn_v8_3652610_pred, nn_v8_3652252_pred, nn_v8_3652129_pred, nn_v8_3651332_pred, nn_v8_3650987_pred,
         nn_v5_3644688_pred, nn_v5_3647673_pred, nn_v5_3646798_pred, nn_v5_3647910_pred, nn_v5_3645888_pred,
         nn_v6_3651765_pred, nn_v6_3649837_pred, nn_v6_3650724_pred, nn_v6_3651711_pred, nn_v6_3651451_pred,
         lgb_v2_3628287_pred, lgb_v5_3626548_pred, lgb_v6_3630270_pred, lgb_v8_3630243_pred, lgb_v9_3631594_pred,lgb_v10_3624431_pred,
         lgb_v5_3631363_pred, lgb_v2_3631178_pred, lgb_v9_3633760_pred,lgb_v8_3633866_pred,lgb_v10_3629696_pred,lgb_v6_3638587_pred,
         cat_v2_3642849_pred, cat_v5_3638180_pred, cat_v6_3641266_pred, cat_v8_3641057_pred, cat_v9_3644587_pred,cat_v10_3639065_pred,
         cat_v2_3643534_pred, cat_v9_3646145_pred, cat_v10_3641476_pred,
          
         nn_v2_3655949_pred, nn_v2_3657138_pred, nn_v2_3659540_pred, nn_v2_3658049_pred, nn_v2_3657516_pred,  
         nn_v5_3657089_pred, nn_v5_3656940_pred, nn_v5_3656590_pred, nn_v5_3655499_pred, nn_v5_3655257_pred,
         nn_v10_3640746_pred, nn_v10_3641109_pred,nn_v10_3642849_pred, nn_v10_3642431_pred, nn_v10_3642371_pred,
         nn_v10_3643881_pred,nn_v10_3644938_pred,nn_v10_3643967_pred,nn_v10_3645736_pred, 
         nn_v10_3650779_pred, nn_v10_3648925_pred, nn_v10_3655221_pred, nn_v10_3652406_pred, nn_v10_3656174_pred,   
         nn_v9_3654152_pred,nn_v9_3659479_pred,nn_v9_3658122_pred,nn_v9_3658062_pred,nn_v9_3661352_pred, 
          
         ]:
    test = test.merge(v,on='card_id',how='left')
    
    
# Level1 - LGB
%%time
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
from scipy import sparse
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.feature_selection import chi2, SelectPercentile

train_df = train.copy()
test_df = test.copy()



train_df = train_df.merge(hist_oof_unique,on='card_id',how='left')
train_df = train_df.merge(hist_new_oof_unique,on='card_id',how='left')
# train_df = train_df.merge(df_v10[feats_v10],on=['card_id'],how='left')

test_df = test_df.merge(hist_pred_unique,on='card_id',how='left')
test_df = test_df.merge(hist_new_pred_unique,on='card_id',how='left')
# test_df = test_df.merge(df_v10[feats_v10],on=['card_id'],how='left')


# outlier
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1

drop_features=['card_id', 'target', 'outliers' , 
              ]

feats = [f for f in train_df.columns if f not in drop_features]

feats = [
    #nn

         'nn_v2nosub_3659540', 'nn_v2nosub_3658049', 
         'nn_v8_3652610', 'nn_v8_3652252','nn_v8_3652129', 'nn_v8_3651332', 'nn_v8_3650987', 
         'nn_v5_3644688', 'nn_v5_3647673', 'nn_v5_3646798', 'nn_v5_3647910', 'nn_v5_3645888',
         'nn_v6_3651765', 'nn_v6_3649837', 'nn_v6_3650724', 'nn_v6_3651711', 'nn_v6_3651451', 
         'nn_v10_3640746', 'nn_v10_3641109', 'nn_v10_3642849', 'nn_v10_3642431', 'nn_v10_3642371',
         'nn_v10_3643881','nn_v10_3644938','nn_v10_3643967','nn_v10_3645736',
         'nn_v10nosub_3650779', 'nn_v10nosub_3648925', 'nn_v10nosub_3655221',
         'nn_v10nosub_3656174', 'nn_v10nosub_3652406',


     #cat    
         'cat_v2_3642849',  

     #lgb    
         'lgb_v10_3624431','lgb_v10nosub_3629696','lgb_v2_3628287','lgb_v2nosub_3631178',
         'lgb_v5nosub_3631363', 'lgb_v6_3630270','lgb_v8_3630243', 'lgb_v8nosub_3633866',
        
     #submodel    
        'hist_max_submodel','hist_sum_submodel','hist_min_submodel', 
        'hist_new_min_submodel','hist_new_max_submodel',
    
    #raw 
       'feature_1', 'feature_2', 'feature_3','year', 'month', 'elapsed_time',
    
    #df_v10
   
        ]

cat_features = []
n_splits= 5
seed = 4590
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
cv_list = []
print ('feats:' + str(len(feats)))

    
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
  
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx] 
    
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 15, #15
        'min_data_in_leaf': 40, #40
        'min_gain_to_split': 0.01,#0.01
        'reg_alpha': 1, #1,
        'reg_lambda': 20, #20        
        'learning_rate': 0.006,
        'max_depth': 4, 
        'bagging_fraction': 0.7,  
        'bagging_freq' : 5,
        'feature_fraction': 0.2,#0.2
        'verbose': 1,
    }    
    

    if n_fold >= 0:
        evals_result = {}
        dtrain = lgb.Dataset(
            train_x, label=train_y, categorical_feature=cat_features)#categorical_feature=cat_features
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain,categorical_feature=cat_features)
        bst = lgb.train(
            params, dtrain, num_boost_round=10000,
            valid_sets=[dval], early_stopping_rounds=200, verbose_eval=100,)#feval = evalerror
        
        new_list = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
        for item in new_list:
            print (item) 

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        oof_cv = rmse(valid_y,  oof_preds[valid_idx])
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += bst.predict(test_df[feats], num_iteration=bst.best_iteration) / folds.n_splits # test_df_new

cv = rmse(train_df['target'],  oof_preds)
print('Full OOF RMSE %.6f' % cv)  

# oof_df = pd.DataFrame()
# oof_df['card_id'] = train_df['card_id']
# oof_df['target'] = oof_preds
# oof_df[['card_id','target']].to_csv('../ensemble/stacking_lv1_lgb_oof_' + str(cv) + '_' + str(seed) + '.csv',index=False)

# test_df['target'] = sub_preds
# test_df[['card_id','target']].to_csv('../ensemble/stacking_lv1_lgb_pred_' + str(cv) + '_' + str(seed) + '.csv',index=False)   

# Level1 - ExtraTree
from sklearn.ensemble import ExtraTreesRegressor


def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

train_df = train.copy()
test_df = test.copy()


train_df = train_df.merge(hist_oof_unique,on='card_id',how='left')
train_df = train_df.merge(hist_new_oof_unique,on='card_id',how='left')


test_df = test_df.merge(hist_pred_unique,on='card_id',how='left')
test_df = test_df.merge(hist_new_pred_unique,on='card_id',how='left')

# outlier
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1


drop_features=['card_id', 'target', 'outliers' , 
              ]

feats = [f for f in train_df.columns if f not in drop_features]

feats = [
    #nn

         'nn_v2nosub_3659540', 'nn_v2nosub_3658049', 
         'nn_v8_3652610', 'nn_v8_3652252','nn_v8_3652129', 'nn_v8_3651332', 'nn_v8_3650987', 
         'nn_v5_3644688', 'nn_v5_3647673', 'nn_v5_3646798', 'nn_v5_3647910', 'nn_v5_3645888',
         'nn_v6_3651765', 'nn_v6_3649837', 'nn_v6_3650724', 'nn_v6_3651711', 'nn_v6_3651451', 
         'nn_v10_3640746', 'nn_v10_3641109', 'nn_v10_3642849', 'nn_v10_3642431', 'nn_v10_3642371',
         'nn_v10_3643881','nn_v10_3644938','nn_v10_3643967','nn_v10_3645736',
         'nn_v10nosub_3650779', 'nn_v10nosub_3648925', 'nn_v10nosub_3655221',
         'nn_v10nosub_3656174', 'nn_v10nosub_3652406',


     #cat    
         'cat_v2_3642849',  

     #lgb    
         'lgb_v10_3624431','lgb_v10nosub_3629696','lgb_v2_3628287','lgb_v2nosub_3631178',
         'lgb_v5nosub_3631363', 'lgb_v6_3630270','lgb_v8_3630243', 'lgb_v8nosub_3633866',
        
     #submodel    
        'hist_max_submodel','hist_sum_submodel','hist_min_submodel', 
        'hist_new_min_submodel','hist_new_max_submodel',
    
    #raw 
       'feature_1', 'feature_2', 'feature_3','year', 'month', 'elapsed_time',
    
    #df_v10
   
        ]

n_splits= 5
seed = 4590
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
oof_preds = np.zeros((train_df.shape[0]))
sub_preds = np.zeros((test_df.shape[0]))
cv_list = []

print ('feats:' + str(len(feats)))
    
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
  
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx] 

    print("Train Index:",train_idx,",Val Index:",valid_idx)
    
    clf = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, min_samples_split=30,min_samples_leaf=40, max_depth=10, max_features=0.5,
                            )
    clf.fit(train_x, train_y)
    
    oof_preds[valid_idx] = clf.predict(valid_x)
    oof_cv = rmse(valid_y,  oof_preds[valid_idx])
    cv_list.append(oof_cv)
    print (cv_list)
    
    sub_preds  += clf.predict(test_df[feats]) / folds.n_splits

        
cv = rmse(train_df['target'],  oof_preds)
print('Full OOF RMSE %.6f' % cv)  
    
# oof_df = pd.DataFrame()
# oof_df['card_id'] = train_df['card_id']
# oof_df['target'] = oof_preds
# oof_df[['card_id','target']].to_csv('../ensemble/stacking_lv1_et_oof_' + str(cv) + '_' + str(seed) + '.csv',index=False)

# test_df['target'] = sub_preds
# test_df[['card_id','target']].to_csv('../ensemble/stacking_lv1_et_pred_' + str(cv) + '_' + str(seed) + '.csv',index=False)           

# Level1 -NN
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import optimizers
from keras.optimizers import Adam
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.layers.advanced_activations import PReLU
from keras import layers
from keras.layers import LeakyReLU
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers import Dropout, Embedding, Permute, Concatenate, Flatten, Reshape
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D, PReLU
from keras.layers.advanced_activations import LeakyReLU, PReLU,ELU
from keras.models import Model
from keras.layers import CuDNNGRU, CuDNNLSTM, Bidirectional, Lambda
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,LearningRateScheduler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from keras.regularizers import l1, l2, l1_l2
from keras import optimizers
import tensorflow as tf
import random
from sklearn.metrics import roc_auc_score
import copy
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler
from sklearn.preprocessing import StandardScaler

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

def preprocess(train_df,test_df,feats):
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(0) 

    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.fillna(0)
    
    scaler = StandardScaler()
    train_df[feats] = scaler.fit_transform(train_df[feats])
    test_df[feats] = scaler.transform(test_df[feats])
    
    return train_df[feats], test_df[feats]


def nn(input_shape):
    model = Sequential()
    model.add(Dense(512, input_dim = input_shape, init='he_normal', activation='relu'))
    model.add(Dropout(0.25))    
    model.add(BatchNormalization())
    model.add(Dense(256, init='he_normal', activation='relu'))#random_uniform he_normal
    model.add(BatchNormalization())
    model.add(Dropout(0.25)) 
    model.add(Dense(32, init='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))      
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam') #Adam(lr=0.001, decay=0.0001)
    return model

train_df = train.copy()
test_df = test.copy()


train_df = train_df.merge(hist_oof_unique,on='card_id',how='left')
train_df = train_df.merge(hist_new_oof_unique,on='card_id',how='left')


test_df = test_df.merge(hist_pred_unique,on='card_id',how='left')
test_df = test_df.merge(hist_new_pred_unique,on='card_id',how='left')

# outlier
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1


drop_features=['card_id', 'target', 'outliers' , 
              ]

feats = [f for f in train_df.columns if f not in drop_features]

feats = [
    #nn

         'nn_v2nosub_3659540', 'nn_v2nosub_3658049', 
         'nn_v8_3652610', 'nn_v8_3652252','nn_v8_3652129', 'nn_v8_3651332', 'nn_v8_3650987', 
         'nn_v5_3644688', 'nn_v5_3647673', 'nn_v5_3646798', 'nn_v5_3647910', 'nn_v5_3645888',
         'nn_v6_3651765', 'nn_v6_3649837', 'nn_v6_3650724', 'nn_v6_3651711', 'nn_v6_3651451', 
         'nn_v10_3640746', 'nn_v10_3641109', 'nn_v10_3642849', 'nn_v10_3642431', 'nn_v10_3642371',
         'nn_v10_3643881','nn_v10_3644938','nn_v10_3643967','nn_v10_3645736',
         'nn_v10nosub_3650779', 'nn_v10nosub_3648925', 'nn_v10nosub_3655221',
         'nn_v10nosub_3656174', 'nn_v10nosub_3652406',


     #cat    
         'cat_v2_3642849',  

     #lgb    
         'lgb_v10_3624431','lgb_v10nosub_3629696','lgb_v2_3628287','lgb_v2nosub_3631178',
         'lgb_v5nosub_3631363', 'lgb_v6_3630270','lgb_v8_3630243', 'lgb_v8nosub_3633866',
        
     #submodel    
        'hist_max_submodel','hist_sum_submodel','hist_min_submodel', 
        'hist_new_min_submodel','hist_new_max_submodel',
    
    #raw 
       'feature_1', 'feature_2', 'feature_3','year', 'month', 'elapsed_time',
    
    #df_v10
   
        ]

# preprocessing
train_df[feats], test_df[feats] = preprocess(train_df,test_df,feats)

n_splits= 5
seed = 4590
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
oof_preds = np.zeros((train_df.shape[0],1))
sub_preds = np.zeros((test_df.shape[0],1))
cv_list = []

print ('feats:' + str(len(feats)))
root_mean_squared_error_mean = 0
    
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
  
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx] 
    
    print("Train Index:",train_idx,",Val Index:",valid_idx)
    
    model = nn(train_x.shape[1])
    
    filepath = str(n_fold) + "_nn_best_model.hdf5" 
    es = EarlyStopping(patience=5, mode='min', verbose=1) #monitor=root_mean_squared_error, 
    checkpoint = ModelCheckpoint(filepath=filepath, save_best_only=True,mode='auto') #monitor=root_mean_squared_error
    reduce_lr_loss = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=2, verbose=1, epsilon=1e-4, mode='min')

    hist = model.fit([ train_x], train_y, batch_size=512, epochs=30, validation_data=(valid_x, valid_y), callbacks=[es, checkpoint, reduce_lr_loss], verbose=1)

    model.load_weights(filepath)
    _oof_preds = model.predict(valid_x, batch_size=1024,verbose=1)
    oof_preds[valid_idx] = _oof_preds.reshape((-1,1))

    oof_cv = rmse(valid_y,  oof_preds[valid_idx])
    oof_std = np.std(oof_preds[valid_idx])
    cv_list.append(oof_cv)

    print (cv_list)

    sub_preds += model.predict(test_df[feats] , batch_size=1024).reshape((-1,1)) / folds.n_splits # test_df_new

        
cv = rmse(train_df['target'],  oof_preds)
print('Full OOF RMSE %.6f' % cv)  
    
# oof_df = pd.DataFrame()
# oof_df['card_id'] = train_df['card_id']
# oof_df['target'] = oof_preds
# oof_df[['card_id','target']].to_csv('../ensemble/stacking_lv1_nn_oof_' + str(cv) + '_' + str(seed) + '.csv',index=False)

# test_df['target'] = sub_preds
# test_df[['card_id','target']].to_csv('../ensemble/stacking_lv1_nn_pred_' + str(cv) + '_' + str(seed) + '.csv',index=False)           

# Level1 - Linear Regression
from sklearn.linear_model import Ridge
from keras.models import Sequential, Model


def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5

train_df = train.copy()
test_df = test.copy()


train_df = train_df.merge(hist_oof_unique,on='card_id',how='left')
train_df = train_df.merge(hist_new_oof_unique,on='card_id',how='left')


test_df = test_df.merge(hist_pred_unique,on='card_id',how='left')
test_df = test_df.merge(hist_new_pred_unique,on='card_id',how='left')

# outlier
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1


drop_features=['card_id', 'target', 'outliers' , 
              ]

feats = [f for f in train_df.columns if f not in drop_features]

feats = [
    #nn

         'nn_v2nosub_3659540', 'nn_v2nosub_3658049', 
         'nn_v8_3652610', 'nn_v8_3652252','nn_v8_3652129', 'nn_v8_3651332', 'nn_v8_3650987', 
         'nn_v5_3644688', 'nn_v5_3647673', 'nn_v5_3646798', 'nn_v5_3647910', 'nn_v5_3645888',
         'nn_v6_3651765', 'nn_v6_3649837', 'nn_v6_3650724', 'nn_v6_3651711', 'nn_v6_3651451', 
         'nn_v10_3640746', 'nn_v10_3641109', 'nn_v10_3642849', 'nn_v10_3642431', 'nn_v10_3642371',
         'nn_v10_3643881','nn_v10_3644938','nn_v10_3643967','nn_v10_3645736',
         'nn_v10nosub_3650779', 'nn_v10nosub_3648925', 'nn_v10nosub_3655221',
         'nn_v10nosub_3656174', 'nn_v10nosub_3652406',


     #cat    
         'cat_v2_3642849',  

     #lgb    
         'lgb_v10_3624431','lgb_v10nosub_3629696','lgb_v2_3628287','lgb_v2nosub_3631178',
         'lgb_v5nosub_3631363', 'lgb_v6_3630270','lgb_v8_3630243', 'lgb_v8nosub_3633866',
        
     #submodel    
        'hist_max_submodel','hist_sum_submodel','hist_min_submodel', 
        'hist_new_min_submodel','hist_new_max_submodel',
    
    #raw 
       'feature_1', 'feature_2', 'feature_3','year', 'month', 'elapsed_time',
    
    #df_v10
   
        ]


n_splits= 5
seed = 4590
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
oof_preds = np.zeros((train_df.shape[0]))
sub_preds = np.zeros((test_df.shape[0]))
cv_list = []
test_df_ridge = test_df.copy()
test_df_ridge = test_df_ridge[feats].values

print ('feats:' + str(len(feats)))
    
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
  
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx] 
    train_x = train_x.values
    valid_x = valid_x.values
    print("Train Index:",train_idx,",Val Index:",valid_idx)
    
    clf = Ridge(solver='auto', fit_intercept=False, alpha=10, max_iter=100, normalize=False)#, tol=0.01
    clf.fit(train_x, train_y)
    
    oof_preds[valid_idx] = clf.predict(valid_x)
    oof_cv = rmse(valid_y,  oof_preds[valid_idx])
    cv_list.append(oof_cv)
    print (cv_list)
    
    sub_preds  += clf.predict(test_df_ridge) / folds.n_splits

        
cv = rmse(train_df['target'],  oof_preds)
print('Full OOF RMSE %.6f' % cv)  
    
# oof_df = pd.DataFrame()
# oof_df['card_id'] = train_df['card_id']
# oof_df['target'] = oof_preds
# oof_df[['card_id','target']].to_csv('../ensemble/stacking_lv1_ridge_oof_' + str(cv) + '_' + str(seed) + '.csv',index=False)

# test_df['target'] = sub_preds
# test_df[['card_id','target']].to_csv('../ensemble/stacking_lv1_ridge_pred_' + str(cv) + '_' + str(seed) + '.csv',index=False)           

# Level2 - Blending
%%time
train = pd.read_csv('../input/train.csv',usecols=['card_id','target'])
test = pd.read_csv('../input/test.csv',usecols=['card_id'])

ridge_oof = pd.read_csv('../ensemble/stacking_lv1_ridge_oof_3.6193833297566522_4590.csv').rename(columns={'target':'ridge'})
et_oof = pd.read_csv('../ensemble/stacking_lv1_et_oof_3.618337011149574_4590.csv').rename(columns={'target':'et'})
nn_oof = pd.read_csv('../ensemble/stacking_lv1_nn_oof_3.616778452963284_4590.csv').rename(columns={'target':'nn'})
lgb_oof = pd.read_csv('../ensemble/stacking_lv1_lgb_oof_3.616281093121384_4590.csv').rename(columns={'target':'lgb'})

ridge_pred = pd.read_csv('../ensemble/stacking_lv1_ridge_pred_3.6193833297566522_4590.csv').rename(columns={'target':'ridge'})
et_pred = pd.read_csv('../ensemble/stacking_lv1_et_pred_3.618337011149574_4590.csv').rename(columns={'target':'et'})
nn_pred = pd.read_csv('../ensemble/stacking_lv1_nn_pred_3.616778452963284_4590.csv').rename(columns={'target':'nn'})
lgb_pred = pd.read_csv('../ensemble/stacking_lv1_lgb_pred_3.616281093121384_4590.csv').rename(columns={'target':'lgb'})

# Merge
for v in [ridge_oof, et_oof, nn_oof, lgb_oof, 
         ]:
    train = train.merge(v,on='card_id',how='left')

for v in [ridge_pred, et_pred, nn_pred, lgb_pred, 
         ]:
    test = test.merge(v,on='card_id',how='left')
    
without_oof = pd.read_csv('../nooutliers/lgb_v10_oof_1.5488890386510492.csv').reset_index(drop=True).rename(columns={'target':'without'})
without_oof['is_outliers'] = 0
without_outliers_oof = pd.read_csv('../nooutliers/lgb_v10_outliersoof_1.5488890386510492.csv').reset_index(drop=True).rename(columns={'target':'without'})
without_outliers_oof['is_outliers'] = 1
without_oof = pd.concat([without_oof,without_outliers_oof],axis=0).reset_index(drop=True)
without_pred = pd.read_csv('../nooutliers/lgb_v10_pred_1.5488890386510492.csv').reset_index(drop=True).rename(columns={'target':'without'})

rank_oof = pd.read_csv('../outliers/outliers_lgb_oof_0.9093378579729009.csv').reset_index(drop=True)#.rename(columns={'pred':'rank'}).drop(['Unnamed: 0','target'],axis=1)
rank_pred = pd.read_csv('../outliers/outliers_lgb_pred_0.9093378579729009.csv').reset_index(drop=True)#.rename(columns={'pred':'rank'}).drop(['Unnamed: 0'],axis=1)

train = train.merge(without_oof,on='card_id',how='left')
train = train.merge(rank_oof,on='card_id',how='left')

test = test.merge(without_pred,on='card_id',how='left')
test = test.merge(rank_pred,on='card_id',how='left')    


%%time
from sklearn.isotonic import IsotonicRegression

#################################################
# lgb
###############################################
cv = rmse(train['target'],  train['lgb'])
print('LGB RMSE %.6f' % cv) 

ir_lgb = IsotonicRegression(out_of_bounds='clip') 
ir_lgb.fit(train['lgb'], train['target']),

ir_pred = ir_lgb.predict(train['lgb'])
train['lgb_isotonic'] = ir_pred
cv = rmse(train['target'],  train['lgb_isotonic'])
print('LGB IsotonicRegression OOF RMSE %.6f' % cv) 

test['lgb_isotonic'] = ir_lgb.predict(test['lgb'])

#################################################
# nn
###############################################
cv = rmse(train['target'],  train['nn'])
print('NN RMSE %.6f' % cv) 

ir_nn = IsotonicRegression(out_of_bounds='clip') 
ir_nn.fit(train['nn'], train['target']),

ir_pred = ir_nn.predict(train['nn'])
train['nn_isotonic'] = ir_pred
cv = rmse(train['target'],  train['nn_isotonic'])
print('NN IsotonicRegression OOF RMSE %.6f' % cv) 

test['nn_isotonic'] = ir_nn.predict(test['nn'])

#################################################
# extra tree
###############################################
cv = rmse(train['target'],  train['et'])
print('ET RMSE %.6f' % cv) 

ir_et = IsotonicRegression(out_of_bounds='clip') 
ir_et.fit(train['et'], train['target']),

ir_pred = ir_et.predict(train['et'])
train['et_isotonic'] = ir_pred
cv = rmse(train['target'],  train['et_isotonic'])
print('ET IsotonicRegression OOF RMSE %.6f' % cv) 

test['et_isotonic'] = ir_et.predict(test['et'])

train['blend'] = train['et_isotonic']*0.3 + train['nn_isotonic']*0.35  + train['lgb_isotonic']*0.35
cv = rmse(train['target'],  train['blend'])
print('Full OOF RMSE %.6f' % cv)  

test['blend'] = test['et_isotonic']*0.3 + test['nn_isotonic']*0.35 + test['lgb_isotonic']*0.35
test['target'] = test['et_isotonic']*0.3 + test['nn_isotonic']*0.35 + test['lgb_isotonic']*0.35

# Replace Nooutliers
%%time
#################################################
# without outliers
###############################################
cv = rmse(train[train['is_outliers']==0]['target'],  train[train['is_outliers']==0]['without'])
print('Only Without outliers RMSE %.6f' % cv) 

cv = rmse(train['target'],  train['without'])
print('Without outliers RMSE %.6f' % cv) 
###############################################
ir_without = IsotonicRegression(out_of_bounds='clip') 
ir_without.fit(train[train['is_outliers']==0]['without'], train[train['is_outliers']==0]['target']),

ir_pred = ir_without.predict(train['without'])
train['without_isotonic'] = ir_pred

cv = rmse(train[train['is_outliers']==0]['target'],  train[train['is_outliers']==0]['without_isotonic'])
print('Only Without outliers IsotonicRegression OOF RMSE %.6f' % cv) 

cv = rmse(train['target'],  train['without_isotonic'])
print('Without outliers IsotonicRegression OOF RMSE %.6f' % cv) 

# without outliers
###############################################
test['without_isotonic'] = ir_without.predict(test['without'])

# ------------------------------------------------------------------------------------------#

#####################train Replace Without Outliers############################
without_cv_list = []
# for threshold in range(4000, 20000, 1000):
for threshold in [14]:    
    threshold = threshold / 2000
    def postprocess_without(row):
        if row['rank']<threshold:
            return row['without_isotonic']
        else:
            return row['blend']
    
    post = train.copy()
    print ('threshold:',threshold)
    post['final'] = post.apply(postprocess_without, axis=1)
    cv = rmse(post['target'],  post['final'])
    without_cv_list.append(cv)
    print('Replace Without Outliers OOF RMSE %.6f' % cv)  

####################test Replace Without Outliers####################
for threshold in [14]:    
    threshold = threshold / 2000
    def postprocess_without(row):
        if row['rank']<threshold:
            return row['without_isotonic']
        else:
            return row['blend']
    
    print ('threshold:',threshold)
    test['target'] = test.apply(postprocess_without, axis=1)
