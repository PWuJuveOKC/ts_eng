import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from collections import Counter
import multiprocessing
import time
import os
import sys

num_cores = multiprocessing.cpu_count()
file_name_train_list = os.listdir("../Data/feat_train")
file_name_test_list = os.listdir("../Data/feat_test")


# Utility function to report best scores
def report(results, n_top=15):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("------------------------------")

fold = 5
##################################  Random Forest ##################################
param_grid_rf = {"max_depth": [3, 4, 5],
              "max_features": ['auto','log2'],
              "min_samples_split": [2, 3, 5,10],
              "min_samples_leaf": [1, 3, 10],
              "criterion": ["gini", "entropy"]}
clf_rf = RandomForestClassifier(n_estimators=50,random_state=0,n_jobs=-1)


####### classification and prediction on test data
def classification_task(clf,param_grid):
    # run grid search
    start = time.time()
    grid_search = GridSearchCV(clf, cv=fold,param_grid=param_grid, n_jobs= (num_cores - 1))
    grid_search.fit(X_train, y_train)
    param = grid_search.best_params_
    clf = RandomForestClassifier(criterion=param['criterion'], random_state=0, max_features=param['max_features'],
                                    max_depth=param['max_depth'],min_samples_leaf=param['min_samples_leaf'],
                                    min_samples_split=param['min_samples_split'],
                                    n_estimators=50)
    clf.fit(X_train,y_train)
    imp_array = np.array(sorted(zip(map(lambda x: round(x, 2), clf.feature_importances_),
                                    list(dat_train)),reverse=True))

    return imp_array

acc_list = []
### Output to txt file
orig_stdout = sys.stdout
f = open('../Output/imp.txt', 'w')
sys.stdout = f


for i in range(2):
    dat_train = pd.read_csv('../Data/feat_train/' + file_name_train_list[i])
    dat_train.fillna(0,inplace=True)
    dat_test = pd.read_csv('../Data/feat_test/' + file_name_test_list[i])
    dat_test.isnull().values.any()
    dat_test.fillna(0,inplace=True)

    train_label = dat_train['_signal'].values
    test_label = dat_test['_signal'].values

    y_train = np.array([int(item.split('_')[-3]) for item in train_label])
    X_train = dat_train.iloc[:,1:].values
    y_test = np.array([int(item.split('_')[-3]) for item in test_label])
    X_test = dat_test.iloc[:,1:].values
    Counter(y_train);Counter(y_test);
    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    IMP_arr = classification_task(clf_rf,param_grid_rf)
    IMP_arr_10 = IMP_arr[:10][:,1]

    print(file_name_train_list[i], IMP_arr_10[0], IMP_arr_10[1], IMP_arr_10[2],
          IMP_arr_10[3], IMP_arr_10[4], IMP_arr_10[5], IMP_arr_10[6], IMP_arr_10[7],
          IMP_arr_10[8], IMP_arr_10[9], '\n',file=f,flush=True)

sys.stdout = orig_stdout
f.close()




