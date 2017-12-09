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
dist_mat2_pd.to_csv('Output/dist_dtw_temp.csv',header=None,index=None)

### Visualization
def distance_matrix(df,metric):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('hot', 30)
    cax = ax1.imshow(df, interpolation="nearest", cmap=cmap)
    ax1.grid(False)
    #plt.title(metric+' Metric')
    labels=['ECG', 'ART', 'CO2', 'PAP']
    ax1.set_xticklabels(labels,fontsize=10)
    ax1.set_xticks(np.arange(size*0.5,size*4.5,size))
    ax1.set_yticklabels(labels,fontsize=10)
    ax1.set_yticks(np.arange(size*0.5,size*4.5,size))
    fig.colorbar(cax)
    plt.show()

distance_matrix(dist_mat, 'Euclidean')
plt.savefig("Output/dist_EU.jpg",dpi=900)
distance_matrix(dist_mat2, 'DTW')
plt.savefig("Output/dist_DTW.jpg",dpi=900)