import pandas as pd

test = pd.read_csv('../input/test.csv')
test['click_time'] = pd.to_datetime(test['click_time'])
Y_dup_not_last = test.loc[test.duplicated(subset=['ip','device','os','app','channel','click_time'],keep='last'), :]
Y_dup_not_last.index

sub = pd.read_csv('../models/minmax_final.csv')
sub.loc[Y_dup_not_last.index,'is_attributed'] = 0

sub.to_csv('../models/lgb_minmax_postprocess.csv.gz', index=False, float_format='%.9f', compression='gzip')
