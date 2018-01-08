import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import time
from sklearn.preprocessing import scale


Alldat = pd.read_csv('Data/Med_feature_norm.csv')
y = np.array(Alldat['Label'])
X = np.array(Alldat.iloc[:,:-2])

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

param_grid = {"learning_rate": ['constant', 'invscaling', 'adaptive'],
              "alpha": [1e-2,0.1,0.2,0.5],
              "activation": ['logistic', 'tanh', 'relu'],
              "hidden_layer_sizes": [(100,),(50,),(100,100),(50,50)]}


# run grid search
start = time.time()
grid_search = GridSearchCV(nn, cv=3,param_grid=param_grid)
grid_search.fit(X, y)
grid_search.best_score_

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time.time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
grid_search.best_params_
