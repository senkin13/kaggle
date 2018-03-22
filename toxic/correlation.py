import pandas as pd
from scipy.stats import ks_2samp
import os
import glob

#path to the validation files
#os.chdir('path to the validation directory')
#list of files for which you want to do comparison
files=glob.glob('../sub/*.csv')

def corr(first_file, second_file):
    # assuming first column is `class_name_id`
    print('\n Relation Between : %s and %s' % (first_file,second_file))
    first_df = pd.read_csv(first_file, index_col=0)
    second_df = pd.read_csv(second_file, index_col=0)
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    #class_names = ['identity_hate']

    for class_name in class_names:

        # all correlations
        print('\n Class: %s' % class_name)
        print(' Pearson\'s correlation score: %0.6f' %
              first_df[class_name].corr(
                  second_df[class_name], method='pearson'))
        #print(' Kendall\'s correlation score: %0.6f' %
        #      first_df[class_name].corr(
        #          second_df[class_name], method='kendall'))
        #print(' Spearman\'s correlation score: %0.6f' %
        #      first_df[class_name].corr(
        #          second_df[class_name], method='spearman'))
        #ks_stat, p_value = ks_2samp(first_df[class_name].values,
        #                            second_df[class_name].values)
        #print(' Kolmogorov-Smirnov test:    KS-stat = %.6f    p-value = %.3e\n'
        #      % (ks_stat, p_value))

#loop through the files and compare the files one by one with each other
k=0
for i in range(len(files)-1):
    for j in range(i+1,len(files)):
        k=k+1
        corr(files[i],files[j])
print('total number of comparison: %d'%k)
