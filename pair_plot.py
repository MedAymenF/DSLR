#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('dataset_train.csv', index_col='Index')
sns.pairplot(df, hue='Hogwarts House', diag_kind='hist', height=1, plot_kws=dict(s=7))
plt.show()
