import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns; sns.set()

files = os.listdir("Data/All_time_series_website/Web_Time_Series_Files_4")
path = "Data/All_time_series_website/Web_Time_Series_Files_4"

files_Air = [f for f in files if f[0:5] == 'CM_ai'] #264
files_Prep = [f for f in files if f[0:8] == 'CM_prate'] #466
files_Hum = [f for f in files if f[0:7] == 'CM_rhum'] #271
files_Slp = [f for f in files if f[0:6] == 'CM_slp'] #271

start, end = 0, 600

AllFiles = [files_Air, files_Prep, files_Hum, files_Slp]

size = 20
AllTS = []
for thefiles in AllFiles:
    for i in range(size):
        ts = pd.read_table(path + "/" + thefiles[i], header=None)
        AllTS.append(np.array(scale(ts[start:end])).reshape(-1))


## Euclidean
dist_mat = euclidean_distances(AllTS,AllTS)

## Fast DTW
def FastDTW(x,y):
    distanceDTW, mypath = fastdtw(x, y, dist=euclidean)
    return distanceDTW


dist_mat2 = np.zeros((size*4, size*4))
start = time.time()
for i in xrange(size*4):
    for j in xrange(size*4):
        lati= AllTS[i]
        latj =AllTS[j]
        dist_mat2[i, j] = FastDTW(lati,latj)
        dist_mat2[j, i] = dist_mat2[i, j]
print(time.time()-start)
#
#

dist_mat2_pd = pd.DataFrame(dist_mat2)
dist_mat2_pd.to_csv('Output/dist_dtw_temp2.csv',header=None,index=None)

#
### Visualization
def distance_matrix(df,metric):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df, interpolation="nearest", cmap=cmap)
    ax1.grid(False)
    #plt.title(metric+' Metric')
    labels=['Temperature', 'Precipitation', 'Humidity', 'Pressure']
    ax1.set_xticklabels(labels,fontsize=10)
    ax1.set_xticks(np.arange(size*0.5,size*4.5,size))
    ax1.set_yticklabels(labels,fontsize=10)
    ax1.set_yticks(np.arange(size*0.5,size*4.5,size))
    fig.colorbar(cax)
    plt.show()

distance_matrix(dist_mat, 'Euclidean')
plt.savefig("Output/dist_EU2.pdf",dpi=900)
distance_matrix(dist_mat2, 'DTW')
plt.savefig("Output/dist_DTW2.pdf",dpi=900)