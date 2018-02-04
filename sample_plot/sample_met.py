import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import seaborn as sns; sns.set()

files = os.listdir("Data/All_time_series_website/Web_Time_Series_Files_4")
path = "Data/All_time_series_website/Web_Time_Series_Files_4"

files_Air = [f for f in files if f[0:5] == 'CM_ai'] #264
files_Prep = [f for f in files if f[0:8] == 'CM_prate'] #466
files_Hum = [f for f in files if f[0:7] == 'CM_rhum'] #271
files_Slp = [f for f in files if f[0:6] == 'CM_slp'] #271

start, end = 0, 1500

AllFiles = [files_Air, files_Prep, files_Hum, files_Slp]

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
ts_dat.to_csv('sample_plot/Climate_sample80_raw.csv')

types = ['Temperature','Precipitation','Humidity','Pressure']

for i in range(len(types)):
    for j in range(3):
        plt.plot(scale(AllTS[IND[3*i+j]]))
        #print(IND[3*i+j])
        plt.savefig('sample_plot/'+types[i]+'.pdf',bbox_inches='tight',dpi=900)
    plt.close()