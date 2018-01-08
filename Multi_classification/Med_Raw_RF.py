import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time

files = os.listdir("Data/All_time_series_website/Web_Time_Series_Files_7")
path = "Data/All_time_series_website/Web_Time_Series_Files_7"

files_ECG = [f for f in files if 'ECG' in f] #456
files_ART = [f for f in files if 'ART' in f] #132

files_CO2 = [f for f in files if 'CO2' in f] #130
files_CVP = [f for f in files if 'CVP' in f] #85

files_PAP = [f for f in files if 'PAP' in f] #120

start, end = 0, 600


sampleFiles = [files_ECG, files_ART, files_CO2, files_PAP]
Allts = []
ts_names = []
for myfiles in sampleFiles:
    frames = []
    for i in range(len(myfiles)):
        ts = pd.read_table(path +  "/" + myfiles[i], header=None)
        ts = np.array(ts).reshape(-1).astype(float)
        ts = scale(ts)[start:end]
        Allts.append(ts)
        ts_names.append(myfiles[i])

X = np.vstack(Allts)
y = np.array([0] * len(files_ECG) + [1] * len(files_ART) + [2] * len(files_CO2) + [3] * len(files_PAP))



clf = RandomForestClassifier(n_estimators=50,random_state=0)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

param_grid = {"max_depth": [3, 4, 5],
              "max_features": ['auto','log2'],
              "min_samples_split": [2, 3, 5,10],
              "min_samples_leaf": [1, 3, 10],
              "criterion": ["gini", "entropy"]}

# run grid search
start = time.time()
grid_search = GridSearchCV(clf, cv=3,param_grid=param_grid)
grid_search.fit(X, y)
grid_search.best_score_

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time.time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

print(grid_search.best_params_)
