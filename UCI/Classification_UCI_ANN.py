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


def data_prepare(dat_train,dat_test):
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

    return X_train, X_test, y_train, y_test, y_train_bin, y_test_bin

n_epoch = 100
####### model training and tuning
def training_model(X_train, y_train_bin):
    max_acc = 0
    for batch_size in batch_size_ls:
        for optimizer in optimizer_ls:
            for dropout in dropout_ls:

                clf = Sequential()
                clf.add(Dense(units = 100, kernel_initializer = 'normal', activation = 'relu', input_dim = 92))
                clf.add(Dropout(dropout))
                clf.add(Dense(units = 50, kernel_initializer = 'normal', activation = 'relu'))
                clf.add(Dropout(dropout))
                clf.add(Dense(units = y_train_bin.shape[1], kernel_initializer = 'normal',activation='softmax'))
                clf.compile(optimizer = optimizer, loss = 'categorical_crossentropy',metrics=[metrics.categorical_accuracy])

                clf.fit(X_train, y_train_bin, batch_size=batch_size, verbose=False, epochs=n_epoch)
                eval = clf.evaluate(X_train,y_train_bin)
                if eval[1] > max_acc:
                    max_acc = eval[1]
                    param = (batch_size,dropout,optimizer)

    return max_acc, param



####### classification and prediction on test data
def classification_task(PARAM,X_train,X_test,y_train_bin,y_test_bin):
    clf = Sequential()
    clf.add(Dense(units=100, kernel_initializer='normal', activation='relu', input_dim=92))
    clf.add(Dropout(PARAM[1]))
    clf.add(Dense(units=50, kernel_initializer='normal', activation='relu'))
    clf.add(Dropout(PARAM[1]))
    clf.add(Dense(units=y_train_bin.shape[1], kernel_initializer='normal', activation='softmax'))
    clf.compile(optimizer=PARAM[2], loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])

    clf.fit(X_train, y_train_bin, batch_size=PARAM[0], verbose=False, epochs=n_epoch)
    my_accuracy = clf.evaluate(X_test,y_test_bin)[1]

    return my_accuracy

orig_stdout = sys.stdout
f = open('../Output/accuracies_ann.txt', 'w')
sys.stdout = f

for i in range(2):
    dat_train = pd.read_csv('../Data/feat_train/' + file_name_train_list[i])
    dat_train.fillna(0, inplace=True)
    dat_test = pd.read_csv('../Data/feat_test/' + file_name_test_list[i])
    dat_test.isnull().values.any()
    dat_test.fillna(0, inplace=True)
    X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = data_prepare(dat_train, dat_test)
    train_acc, train_param = training_model(X_train, y_train_bin)

    accuracy = classification_task(train_param, X_train, X_test, y_train_bin, y_test_bin)
    print(file_name_train_list[i], np.round(accuracy, 3), '\n', file=f, flush=True)

sys.stdout = orig_stdout
f.close()














