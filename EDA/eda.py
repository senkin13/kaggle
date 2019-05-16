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
