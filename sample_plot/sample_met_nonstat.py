import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import seaborn as sns; sns.set()

files = pd.read_csv('Data/nonstat_ar.csv')
start, end = 0, 2000

names = list(files)
files_Timmer1 = [f for f in names if 'Timmer1' in f]
files_Timmer2 = [f for f in names if 'Timmer2' in f]
files_Timmer3 = [f for f in names if 'Timmer3' in f]
All_Files = [files_Timmer1,files_Timmer2,files_Timmer3]

files_sample = []
size = 4
for i in range(3):
    r = np.random.RandomState(999-i)
    files_Timmer_samp = r.choice(All_Files[i],size)
    files_sample.append(files_Timmer_samp)

for i in range(3):
    a = files.loc[:,files_sample[i]]
    plt.plot(scale(a.iloc[start:end,:]))
    plt.savefig('sample_plot/nonstat_Timmer' + str(i+1) + '.pdf', bbox_inches='tight', dpi=900)
    plt.close()
