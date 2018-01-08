import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time


Alldat = pd.read_csv('Data/Med_feature_norm.csv')
y = np.array(Alldat['Label'])
X = np.array(Alldat.iloc[:,:-2])

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
            print("------------------------------")

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
grid_search.best_params_

clf_RF = RandomForestClassifier(n_estimators=50,criterion='gini',max_depth=3,max_features='log2',
                                min_samples_leaf=10,min_samples_split=10,random_state=0)
clf_RF.fit(X,y)
print(clf_RF.score(X,y))
print(np.array(list(Alldat))[np.where(clf_RF.feature_importances_ > 0.02)])
print "Features sorted by their score:"
print np.array(sorted(zip(map(lambda x: round(x, 4),clf_RF.feature_importances_),list(Alldat)),
             reverse=True))
