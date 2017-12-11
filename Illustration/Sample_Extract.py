import os
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import time
import numpy as np
import random
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

files = os.listdir("../Data/All_time_series_website/Web_Time_Series_Files_7")
path = "../Data/All_time_series_website/Web_Time_Series_Files_7"

files_ECG = [f for f in files if 'ECG' in f] #456
files_ART = [f for f in files if 'ART' in f] #132

files_CO2 = [f for f in files if 'CO2' in f] #130
files_CVP = [f for f in files if 'CVP' in f] #85

files_PAP = [f for f in files if 'PAP' in f] #120

start, end = 0, 600

AllFiles = [files_ECG, files_ART, files_CO2, files_PAP]

size = 20
## format the dataframe for tsfresh use
AllFrames = []
ts_names = []
for myfiles in AllFiles:
    frames = []
    k = 0
    for i in range(size):
        k += 1
        ts = pd.read_table(path +  "/" + myfiles[i], header=None)
        ts['id'] = k
        ts['time'] = ts.index
        ts.columns = ['value', 'id', 'time']
        ts['value'] = scale(ts.value)
        ts = ts.iloc[start:end, :]
        ts_names.append(myfiles[i])
        frames.append(ts)
    dat = pd.concat(frames)
    extracted_features = extract_features(dat, column_id="id", column_sort="time")
    impute_features = impute(extracted_features)
    #norm_features = scale(impute_features)
    AllFrames.append(impute_features)

Alldat = pd.concat(AllFrames)
Alldat['name'] = ts_names
Alldat['Label'] = [0] * size + [1] * size + [2] * size + [3] * size

Alldat.to_csv('../Data/sampleTS_Med_feature.csv',index=None)


### Drop columns with all same value and scale the features
def DropSame(df0):
    df = df0.copy();
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
    return df

Alldat2 = DropSame(Alldat)
norm_features = scale(Alldat2.iloc[:,:-2])
norm_features = pd.DataFrame(norm_features)
norm_features.columns = list(Alldat2)[:-2]
norm_features['name'] = ts_names
norm_features['Label'] = [0] * size + [1] * size + [2] * size + [3] * size

norm_features.to_csv('../Data/sampleTS_Med_feature_norm.csv',index=None)