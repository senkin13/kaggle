from sklearn.metrics import confusion_matrix
import itertools
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline

target_col = "target"

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
