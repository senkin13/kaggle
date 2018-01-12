df_oil = pd.read_csv(
    "input/oil.csv",
    parse_dates=["date"] 
)

df_oil_train = pd.merge(df2017, df_oil, how='left', on=['date'])
df_oil_test = pd.merge(df_test, df_oil, how='left', on=['date'])

df_oil_train = df_oil_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["dcoilwtico"]].unstack(
        level=-1).fillna(method='bfill')
df_oil_train.columns = df_oil_train.columns.get_level_values(1)
df_oil_test = df_oil_test.set_index(
    ["store_nbr", "item_nbr", "date"])[["dcoilwtico"]].unstack(
        level=-1).fillna(method='bfill')
df_oil_test.columns = df_oil_test.columns.get_level_values(1)
df_oil_test = df_oil_test.reindex(df_oil_train.index).fillna(bfill)
df_oil = pd.concat([df_oil_train, df_oil_test], axis=1)




        "oil_day_1_2017": get_timespan(df_oil, t2017, 1, 1).values.ravel(), 
        "oil_day_7_2017": get_timespan(df_oil, t2017, 7, 1).values.ravel(),  
        "oil_day_14_2017": get_timespan(df_oil, t2017, 14, 1).values.ravel(),  
        "oil_day_21_2017": get_timespan(df_oil, t2017, 21, 1).values.ravel(),          
        "oil_mean_3_2017": get_timespan(df_oil, t2017, 3, 3).mean(axis=1).values, 
        "oil_mean_7_2017": get_timespan(df_oil, t2017, 7, 7).mean(axis=1).values,   
        "oil_mean_14_2017": get_timespan(df_oil, t2017, 14, 14).mean(axis=1).values,   
        "oil_mean_28_2017": get_timespan(df_oil, t2017, 28, 28).mean(axis=1).values,   

    for i in range(16):
        X["oil_{}".format(i)] = df_oil[
            t2017 + timedelta(days=i)].values.astype(np.float32)     
