from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

df_train = pd.read_csv(
    'input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 80735413)  # from 2016-05-31
)

hol = pd.read_csv(
    "input/holidays_events.csv", parse_dates=["date"]
)

holiday = hol[(hol['locale'] == 'National') & (hol['transferred'] == False)].drop(['type','locale','locale_name','description','transferred'], axis=1)

holiday['isholiday']=True

train = pd.merge(df_train, holiday, how='left', on=['date'])

hol_2017_train = train.set_index(
    ["store_nbr", "item_nbr", "date"])[["isholiday"]].unstack(
        level=-1).fillna(False)
