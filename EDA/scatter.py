import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline

target_col = "target"

plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train[target_col].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.show()
