from sklearn.metrics import confusion_matrix
import itertools
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline

target_col = "target"

##### plot compraision
plt.figure(figsize=(12,6))
plt.plot(zhan_wk['wk_idnt'],zhan_wk['zhan_mape'],c='red',label='zhan')
plt.plot(zhan_wk['wk_idnt'],zhan_wk['zhan2_mape'],c='blue',label='zhan2')

plt.legend(loc='upper left')
plt.xlabel('wk_idnt')
plt.ylabel('mape mean')
plt.title('benchmark')
plt.show

##### scatter
plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train[target_col].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.show()

##### Histogram
plt.figure(figsize=(12,8))
sns.distplot(train[target_col].values, bins=50, kde=False, color="red")
plt.title("Histogram of Target")
plt.xlabel('Target', fontsize=12)
plt.show()

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
for col in [
max_tmprt_max_yearmonth', 
]:
    plt.ion()
    fig, ax = plt.subplots(figsize=(12,8))
    plt.title('Correlation between ' + str(col) + ' and sales')
    ax2 = ax.twinx()
  
    dates = mpl.dates.date2num(df['day_dt'])
    ax2.plot_date(dates, np.sqrt(df['target']+1), '-', color='tab:red',label='target', alpha=0.3)
 
    ax.plot_date(dates, df[col], '.', color='tab:blue', label=col)
    ax.set_ylabel(col); ax2.set_ylabel('target')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.ioff()    
    plt.show()

##### Scatter plot between target and prediction
reg_cd = 63
region_dense_region = region_dense[region_dense['reg_cd']==reg_cd]

def plot_oof_preds(llim, ulim):
        plt.figure(figsize=(6,6))
        sns.scatterplot(x='mape_zhan',y='mape_zhan2',
                        data=region_dense_region[['mape_zhan', 'mape_zhan2']],s=200);
        plt.xlim((llim, ulim))
        plt.ylim((llim, ulim))
        plt.plot([llim, ulim], [llim, ulim])
        plt.xlabel('mape_zhan')
        plt.ylabel('mape_zhan2')
        plt.title('Region ' + str(reg_cd) + ' Mape Benchmark', fontsize=18)
        plt.show()

plot_oof_preds(0, 6)

##### Scatter plot the real time to failure vs predicted (Testing Set)
plt.figure(figsize=(6, 6))
plt.scatter(y_test.values.flatten(), predictions)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.title('Testing Set')
plt.xlabel('time to failure', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

##### displot
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
ax = sns.distplot(train["target"].values, bins=200, kde=False)
ax.set_xlabel('target', fontsize=15)
ax.set_ylabel('target', fontsize=15)
ax.set_title("target Histogram", fontsize=20)

##### group barplot 
plt.figure(figsize=(16, 10))
sns.barplot(x="wk_idnt", hue="fr_ord_main_reg_cd", y="prd_sls_ratio", data=region_dense)
plt.show()

##### barplot
cnt_srs=train.groupby("feature_1").target.mean()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('target', fontsize=12)
plt.xlabel('feature_1', fontsize=12)
plt.show()


cnt_srs = train_tr['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in train set")
plt.show()

cnt_srs = valid_tr['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in test set")
plt.show()

##### venn2
plt.figure(figsize=(20,16))
venn2([set(train.card_id.unique()), set(test.card_id.unique())], set_labels = ('Train set', 'Test set') )
plt.title("Number of card_id in train and test", fontsize=15)
plt.show()

##### confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
cnf_matrix = confusion_matrix(train_df['outliers'], (oof_preds>0.5))# np.round

plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=[0,1],
                      title='Confusion matrix')
