import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import time


Alldat = pd.read_csv('Data/Med_feature_norm.csv')
y = np.array(Alldat['Label'])
X = np.array(Alldat.iloc[:,:-2])

clf = AdaBoostClassifier(n_estimators=50,random_state=0)

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

param_grid = {"learning_rate": [0.05, 0.1, 0.2, 0.3, 0.5, 1]}

# run grid search
start = time.time()
grid_search = GridSearchCV(clf, cv=3,param_grid=param_grid)
grid_search.fit(X, y)
grid_search.best_score_

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time.time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
grid_search.best_params_

clf_Ada = AdaBoostClassifier(learning_rate=0.3,random_state=0)
clf_Ada.fit(X,y)
clf_Ada.score(X,y)
print(np.array(list(Alldat))[np.where(clf_Ada.feature_importances_ > 0.02)])