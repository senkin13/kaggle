import numpy as np
from fbprophet import Prophet
from datetime import datetime

def wr(s,i):
    df = store[store['item_nbr'] == i].drop(['store_nbr', 'item_nbr'], axis=1)
    df.rename(columns = {'dt':'ds'}, inplace=True)
    df.rename(columns = {'unit_sales':'y'}, inplace=True)
    df['y'] = np.log1p(df['y'])
    startdate = datetime.strptime(df.iloc[df.shape[0]-1].ds, "%Y-%m-%d")
    enddate = datetime.strptime("2017-08-31", "%Y-%m-%d")
    periods = (enddate - startdate).days
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)import pandas as pd

    forecast['store_nbr'] = s
    forecast['item_nbr'] = i
    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast[['ds','store_nbr','item_nbr','yhat']].iloc[-16:].to_csv('output/' + str(s) + '/' + str(i) + '.csv', index=False, header=False)
    
    
from multiprocessing import Pool

pool = Pool(16)

for s in range(1,25):
  store = pd.read_csv("store/" + str(s))
  for i in store['item_nbr'].unique():
    pool.apply_async(func=wr, args=(s,i),) 

pool.close()
pool.join()    
