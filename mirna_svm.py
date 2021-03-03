#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import svm, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


myfile = np.genfromtxt ('resultCSV.csv', delimiter=",", dtype=int) # cols: label, ID, race, mirnas...
data = myfile[1:, 3:-1]  # data: n_samples * n_features array; LAST COL OF CSV IS nan (after the last comma)
labels = myfile[1:, 0]

sz1 = data.shape
sz2 = labels.shape
print(f'data: {sz1}, labels: {sz2} ')
print()
# type1 = data.dtype
# type2 = labels.dtype
# print(f'data type: {type1}, labels: {type2} ')
#
# max_data = np.amax(data)
# max_labels = np.amax(labels)
# print(f'data MAX: {max_data}, label MAX: {max_labels} ')
#
# indx_max = np.argmax(data)
# print(f'data MAX index: {indx_max}')
#
# print(data[0,-1])
# print(data[1,-1])
# print(data[:,0])

num_test = 7

data_train = data[:-num_test]
label_train = labels[:-num_test]
data_test = data[-num_test:]
label_test = labels[-num_test:]
#================= Linear Regression ================
lr = LinearRegression().fit(data_train, label_train)
# reg.score(X, y)
# reg.coef_
# reg.intercept_

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
lr_score = lr.score(data_test, label_test)

print ('Linear Regression')
print(f'variance score: {lr_score}')
print()

#================ Logistic Regression ==================
logr = LogisticRegression(solver='lbfgs')
logr.fit(data_train, label_train)
predicted = logr.predict(data_test)
# logr.predict_proba(X[:2, :])
logr_score = logr.score(data_test, label_test)

print ('Logistic Regression')
print(f'score: {logr_score}')
print()

#================== SVM =============================
# classifier = svm.SVC(kernel = 'linear')
#classifier = svm.SVC(kernel = 'rbf')
classifier = svm.SVC(kernel = 'poly', degree = 3)

classifier.fit(data_train, label_train)

predicted = classifier.predict(data_test)
# expected = labels[-num_test:]

print ('SVC')
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(label_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(label_test, predicted))
print(f'expected: {label_test}')
print(f'predicted: {predicted}')
print()

#======================= Random Forest =========================
clf = RandomForestClassifier(n_estimators= 10, max_depth=None,min_samples_split=2, random_state=0)
#clf = RandomForestClassifier(n_estimators= 10)
clf = clf.fit(data_train, label_train)
predicted = clf.predict(data_test)
rf_score = clf.score(data_test, label_test)

print ('Random Forest')
print(f'expected: {label_test}')
print(f'predicted: {predicted}')
print(f'score: {rf_score}')
print()

#================== nearest-neighbor classifier =============
# knn = KNeighborsClassifier()
# knn.fit(data_train, label_train)
# predicted = knn.predict(label_test)
#
# print ('KNN')
# print(f'true: {label_test}')
# print(f'predicted: {predicted}')
# print()

#===================================================

#print(data)
#print(labels)

# data = myfile[:,1]
# third = csv[:,2]
#     myfile = np.array(list(csv.reader(csv_file, delimiter=','))) # cols: label, ID, race, mirnas...
#     print(myfile)
#row = np.size(myfile,0)


# data = myfile[1:, 3:]
# labels = myfile[1:,0]
# sz_label = len(labels)
# print(f'label: {sz_label}')
# numrows = len(data)    # 3 rows in your example
# numcols = len(data[0])
# print(f'data: row = {numrows}, col = {numcols}')






# get 2D array size
    # numrows = len(data)    # 3 rows in your example
    # numcols = len(data[0])
    # print(f'row = {numrows}, col = {numcols}')
