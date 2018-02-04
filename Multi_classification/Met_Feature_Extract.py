import os
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import time
import numpy as np
import random
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

files = os.listdir("Data/All_time_series_website/Web_Time_Series_Files_4")
path = "Data/All_time_series_website/Web_Time_Series_Files_4"

files_Air = [f for f in files if f[0:5] == 'CM_ai'] #264
files_Prep = [f for f in files if f[0:8] == 'CM_prate'] #466
files_Hum = [f for f in files if f[0:7] == 'CM_rhum'] #271
files_Slp = [f for f in files if f[0:6] == 'CM_slp'] #271

start, end = 0, 1500

AllFiles = [files_Air, files_Prep, files_Hum, files_Slp]

## format the dataframe for tsfresh use
AllFrames = []
ts_names = []
for myfiles in AllFiles:
    frames = []
    k = 0
    for i in range(len(myfiles)):
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

Alldat = pd.concat(AllFrames);
Alldat['name'] = ts_names;
Alldat['Label'] = [0] * len(files_Air) + [1] * len(files_Prep) + [2] * len(files_Hum) + \
                  [3] * len(files_Slp);

Alldat.to_csv('Data/Met_feature.csv',index=None)


### Drop columns with all same value and scale the features
def DropSame(df0):
    df = df0.copy();
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
    return df

Alldat2 = DropSame(Alldat);
norm_features = scale(Alldat2.iloc[:,:-2]);
norm_features = pd.DataFrame(norm_features);
norm_features.columns = list(Alldat2)[:-2];
norm_features['name'] = ts_names;
norm_features['Label'] = [0] * len(files_Air) + [1] * len(files_Prep) + [2] * len(files_Hum) + \
                  [3] * len(files_Slp);

norm_features.to_csv('Data/Met_feature_norm.csv',index=None)