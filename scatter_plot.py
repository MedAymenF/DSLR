#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('dataset_train.csv')
df.plot.scatter('Astronomy', 'Defense Against the Dark Arts')
plt.show()
