import pandas as pd
import numpy as np
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
##################################  Neural Networks ##################################

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras import metrics

batch_size_ls =  [10, 25, 32, 50]
optimizer_ls = ['adam', 'rmsprop']
dropout_ls = [0.1,0.2,0.4]

i=0
dat_train = pd.read_csv('../Data/feat_train/' + file_name_train_list[i])
dat_train.fillna(0, inplace=True)
dat_test = pd.read_csv('../Data/feat_test/' + file_name_test_list[i])
dat_test.isnull().values.any()
dat_test.fillna(0, inplace=True)

train_label = dat_train['_signal'].values
test_label = dat_test['_signal'].values

y_train = np.array([int(item.split('_')[-3]) for item in train_label])
X_train = dat_train.iloc[:, 1:].values
y_test = np.array([int(item.split('_')[-3]) for item in test_label])
X_test = dat_test.iloc[:, 1:].values
Counter(y_train);
Counter(y_test);
# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


y_train_bin = to_categorical(y_train)
y_test_bin = to_categorical(y_test)
max_acc = 0
for batch_size in batch_size_ls:
    for optimizer in optimizer_ls:
        for dropout in dropout_ls:

            clf = Sequential()
            clf.add(Dense(units = 100, kernel_initializer = 'normal', activation = 'relu', input_dim = 92))
            clf.add(Dropout(dropout))
            clf.add(Dense(units = 50, kernel_initializer = 'normal', activation = 'relu'))
            clf.add(Dropout(dropout))
            clf.add(Dense(units = len(np.unique(y_train)) + 1, kernel_initializer = 'normal',activation='softmax'))
            clf.compile(optimizer = optimizer, loss = 'categorical_crossentropy',metrics=[metrics.categorical_accuracy])

            clf.fit(X_train, y_train_bin, batch_size=batch_size, epochs=100)
            eval = clf.evaluate(X_train,y_train_bin)
        if eval[1] > max_acc:
            max_acc = eval[1]
            param_est = (batch_size,dropout,optimizer)
            print (max_acc, param_est)


####### classification and prediction on test data
def classification_task(PARAM):
    clf = Sequential()
    clf.add(Dense(units=100, kernel_initializer='normal', activation='relu', input_dim=92))
    clf.add(Dropout(PARAM[1]))
    clf.add(Dense(units=50, kernel_initializer='normal', activation='relu'))
    clf.add(Dropout(PARAM[1]))
    clf.add(Dense(units=len(np.unique(y_train)) + 1, kernel_initializer='normal', activation='softmax'))
    clf.compile(optimizer=PARAM[2], loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])

    clf.fit(X_train, y_train_bin, batch_size=PARAM[0], epochs=100)
    my_accuracy = clf.evaluate(X_test,y_test_bin)[1]

    return my_accuracy

classification_task(param_est)

acc_list = []
### Output to txt file
orig_stdout = sys.stdout
f = open('../Output/accuracies.txt', 'w')
sys.stdout = f

for i in range(2):



sys.stdout = orig_stdout
f.close()









