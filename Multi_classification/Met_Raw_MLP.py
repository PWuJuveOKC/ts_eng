import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import time


files = os.listdir("Data/All_time_series_website/Web_Time_Series_Files_4")
path = "Data/All_time_series_website/Web_Time_Series_Files_4"

files_Air = [f for f in files if f[0:5] == 'CM_ai'] #264
files_Prep = [f for f in files if f[0:8] == 'CM_prate'] #466
files_Hum = [f for f in files if f[0:7] == 'CM_rhum'] #271
files_Slp = [f for f in files if f[0:6] == 'CM_slp'] #271

start, end = 0, 1500


sampleFiles = [files_Air, files_Prep, files_Hum, files_Slp]
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
y = np.array([0] * len(files_Air) + [1] * len(files_Prep) + [2] * len(files_Hum) + [3] * len(files_Slp))

# use a full grid over all parameters
param_grid = {"learning_rate": ['constant', 'invscaling', 'adaptive'],
              "alpha": [1e-2,0.1,0.2,0.5],
              "activation": ['logistic', 'tanh', 'relu'],
              "hidden_layer_sizes": [(100,),(50,),(100,100),(50,50)]}


nn = MLPClassifier(random_state=0,max_iter=1000,warm_start=True)

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

# run grid search
start = time.time()
grid_search = GridSearchCV(nn, cv=3,param_grid=param_grid)
grid_search.fit(X, y)
grid_search.best_score_

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time.time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
print(grid_search.best_params_)