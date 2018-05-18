import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dat = pd.read_excel('Accuracy/accuracies.xlsx')
dat.isnull().values.any() #False

file_name = list(dat.File)
file_name = ['_'.join(item.split('_')[:-4]) for item in file_name]
dat.iloc[:,0] = file_name
dat1 = dat.iloc[:,1:]
dat1.index = dat.iloc[:,0]


ax = sns.heatmap(dat1,vmin=0, vmax=1,yticklabels=False,cmap='RdYlGn',linewidths=0.01)
plt.tight_layout()
plt.savefig('Accuracy/acc_heat.pdf',dpi=900)