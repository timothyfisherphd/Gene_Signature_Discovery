#!/usr/bin/python3
# coding: utf-8

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import svm, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (brier_score_loss, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.decomposition import PCA

myfile = np.genfromtxt('data_alllabel.csv', delimiter=",", dtype=int) # cols: label, ID, race, mirnas...
data = myfile[1:, 3:]  # data: n_samples * n_features array; LAST COL OF CSV IS nan (after the last comma)
labels = myfile[1:, 0] # asian, black, white
num_label = 3
# data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0], [10, 2], [5, 5]])

# estimators = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]
# [2, 3, 4, 5, 6, 7, 8]

estimators = [('logstic_regression', LogisticRegression(solver='lbfgs', n_jobs=2, multi_class='auto',max_iter=1000)),
              ('SVC_poly3', svm.SVC(kernel='poly', degree=3, gamma='auto', max_iter=1000)),
              ('SVC_linear', svm.SVC(kernel='linear', gamma='auto', max_iter=1000)),
              ('RandomForest_equalweight', RandomForestClassifier(n_estimators=10)),
              ('Neural_network', MLPClassifier(alpha=1,max_iter=1000))]
                # ('SVC_poly3', svm.SVC(kernel='poly', degree=3, gamma='auto')),
                # ('RandomForest_balanceweight', RandomForestClassifier(n_estimators=10, class_weight='balanced_subsample')),
pca_sets = [('PCA, var_ratio_acc = 0.5', PCA(n_components=0.5, whiten=False)),
            ('PCA, var_ratio_acc = 0.9', PCA(n_components=32, whiten=False)),
            ('PCA, var_ratio_acc = 0.98', PCA(n_components=0.98, whiten=False)),]

for name_pca, method_pca in pca_sets:
    print("@@@@@@@@@@@@@@  %s  @@@@@@@@@@@@@@" % name_pca)
    pca = method_pca
    pca.fit(data)
    data_pca = pca.transform(data)
    print('n_components = %d' % pca.n_components_)

    for i in range(1):
        print("******************  %d   ******************" % i)
        data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.25, random_state=i, shuffle=True)

        count_train = [0]*num_label
        count_test = [0]*num_label
        for k in label_train:
            count_train[k] += 1
        for k in label_test:
            count_test[k] += 1
        # print("total sample: %d" % (label_train.shape[0]+label_test.shape[0]))
        # sz1 = label_train.shape[0]/(label_train.shape[0]+label_test.shape[0])
        # sz2 = label_test.shape[0]/(label_train.shape[0]+label_test.shape[0])
        # print("training dataset: %.2f, testing dataset: %.2f" % (sz1, sz2))
        for k in range(num_label):
            sz0 = (count_train[k]+count_test[k])/(label_train.shape[0]+label_test.shape[0])
            sz1 = count_train[k]/label_train.shape[0]
            sz2 = count_test[k]/label_test.shape[0]
            print("label %d in total dataset: %.2f" % (k, sz0))
            print("label %d in training dataset: %.2f, in testing dataset: %.2f" % (k, sz1, sz2))
            print()
        # ------------------------------------------------------
        for name, est in estimators:
            est.fit(data_train, label_train)
            label_pred = est.predict(data_test)
            print ('================== %s (%d labels) ===================' % (name,num_label))
            # print("Accuracy: %1.2f" % accuracy_score(label_test, label_pred))
            # print("Precision:\n%s" % precision_score(label_test, label_pred, average=None))
            # print("Recall:\n%s" % recall_score(label_test, label_pred, average=None))
            # print("F1:\n%s" % f1_score(label_test, label_pred, average=None))
            cm = confusion_matrix(label_test, label_pred)
            # print("Confusion matrix:\n%s" % cm)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized Confusion matrix:\n%s" % cm)
            # print()
