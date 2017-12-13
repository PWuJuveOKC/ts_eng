import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import seaborn as sns; sns.set()

#### Sample Plots for Illustration (3 for each type within 20 samples)
files = os.listdir("Data/All_time_series_website/Web_Time_Series_Files_7")
path = "Data/All_time_series_website/Web_Time_Series_Files_7"

files_ECG = [f for f in files if 'ECG' in f] #456
files_ART = [f for f in files if 'ART' in f] #132

files_CO2 = [f for f in files if 'CO2' in f] #130
files_CVP = [f for f in files if 'CVP' in f] #85

files_PAP = [f for f in files if 'PAP' in f] #120

start, end = 0, 600

AllFiles = [files_ECG, files_ART, files_CO2, files_PAP]

size = 20
AllTS = []
ts_names =[]
for thefiles in AllFiles:
    for i in range(size):
        ts = pd.read_table(path + "/" + thefiles[i], header=None)
        ts_names.append(thefiles[i])
        AllTS.append(np.array((ts[start:end])).reshape(-1))

IND = []
r = np.random.RandomState(999)
for i in range(4):
    IND.append(r.choice(np.array(range(20*i,20*(i+1))),3,replace=False))
IND = np.reshape(IND,-1)

###Dataframe
ts_dat = pd.DataFrame(AllTS)
ts_dat.index = ts_names
ts_dat.to_csv('sample_med/Medical_sample80_raw.csv')

types = ['ECG','ART','CO2','PAP']

for i in range(len(types)):
    for j in range(3):
        plt.plot(AllTS[IND[3*i+j]])
        #print(IND[3*i+j])
        plt.savefig('sample_med/'+types[i]+'.jpg',dpi=900)
    plt.close()