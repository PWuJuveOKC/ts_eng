import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

##################################  SVM ###########################################
param_grid_svm = {"kernel": ['linear','rbf'],
              "C": [1e-5,1e-4,1e-3,0.01,0.1,1,2,5,10,100,1000,10000],
              "gamma": [1e-5,1e-4,1e-3,0.01,0.1,1,2,5,10,100,1000,1000]}
clf_svm = SVC()

##################################  Decision Tree ###########################################
param_grid_tree = {"max_depth": [3, 4, 5],
              "max_features": ['auto','log2'],
              "min_samples_split": [2, 3, 5,10],
              "min_samples_leaf": [1, 3, 10],
              "criterion": ["gini", "entropy"]}
clf_tree = DecisionTreeClassifier(random_state=0)

##################################  KNN ###########################################
param_grid_knn = {"n_neighbors": [1,2,5,10],
                 "leaf_size": [5,10,20,30,50],
              "p": [1,2,3,4],
              "weights": ['uniform','distance']}
clf_knn = KNeighborsClassifier(algorithm='auto', n_jobs=-1)

##################################  Logistic ########################################
param_grid_log = {"C": [1e-5,1e-4,1e-3,0.01,0.1,1,2,5,10,100,1000,10000]}
clf_log = LogisticRegression(penalty='l1', random_state=0)

#######################################  ADA ########################################
param_grid_ada = {"learning_rate": [0.001,0.01,0.05, 0.1, 0.2, 0.3, 0.5, 1],
                  "n_estimators": [20,50,100]}
clf_ada = AdaBoostClassifier(random_state=0)

####### classification and prediction on test data
def classification_task(clf,param_grid):
    # run grid search
    start = time.time()
    grid_search = GridSearchCV(clf, cv=fold,param_grid=param_grid, n_jobs= (num_cores - 1))
    grid_search.fit(X_train, y_train)

    #print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
        #% (time.time() - start, len(grid_search.cv_results_['params'])))
    #report(grid_search.cv_results_)
    #grid_search_rf.best_params_
    my_accuracy = grid_search.score(X_test, y_test)
    #print('Accuracy: ', my_accuracy)
    #print('Prediction:', Counter(grid_search.predict(X_test)))

    return my_accuracy

acc_list = []
### Output to txt file
orig_stdout = sys.stdout
f = open('../Output/accuracies.txt', 'w')
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


    ada_acc = classification_task(clf_ada,param_grid_ada) #Ada Boost

    tree_acc = classification_task(clf_tree,param_grid_tree) #Tree

    knn_acc = classification_task(clf_knn,param_grid_knn) #KNN

    log_acc = classification_task(clf_log,param_grid_log) #Logistic regression

    rf_acc = classification_task(clf_rf,param_grid_rf) #RF

    svm_acc = classification_task(clf_svm,param_grid_svm) #SVM

    #acc_list.append([file_name_train_list[i],ada_acc,tree_acc,knn_acc,log_acc,rf_acc,svm_acc])
    print(file_name_train_list[i], np.round(ada_acc,3),np.round(tree_acc,3),np.round(knn_acc,3),
          np.round(log_acc, 3),np.round(rf_acc,3),np.round(svm_acc,3),'\n',file=f,flush=True)

sys.stdout = orig_stdout
f.close()

#dat_acc = pd.DataFrame(acc_list,columns=['file_name','ADA','Decision Tree', 'KNN', 'Logistic','RF','SVM'])

#dat_acc.to_csv('../Output/tsfresh_accuracies.csv',index=None)

### ANN
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
#
# def build_clf(optimizer):
#     clf = Sequential()
#     clf.add(Dense(units = 100, kernel_initializer = 'normal', activation = 'relu', input_dim = 92))
#     clf.add(Dropout(0.2))
#     clf.add(Dense(units = 50, kernel_initializer = 'normal', activation = 'relu'))
#     clf.add(Dropout(0.2))
#     clf.add(Dense(units = len(np.unique(y_train)), kernel_initializer = 'normal',activation='softmax'))
#     clf.compile(optimizer = optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
#     return clf
#
#
# clf_ann = KerasClassifier(build_fn = build_clf, epochs=100)
# param_grid_ann = {'batch_size': [25, 50, 50],
#               'optimizer': ['adam', 'rmsprop']}
#






