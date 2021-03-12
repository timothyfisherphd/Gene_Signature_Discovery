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

with open('/home/hju/advanceBio/largedata/data_alllabel.csv','r') as f:
    gene_name = f.readline().rstrip('\n').split(',')
    gene_name = gene_name[3:]
print('gene_name done!')
num_feature = len(gene_name)

dict_num_gene = {}
nnarray = np.arange(num_feature, dtype=int)
for NN, GG in zip(nnarray, gene_name):
    dict_num_gene[NN] = GG
print('dict_num_gene done!')
#-----------------------------------------------
myfile = np.genfromtxt('/home/hju/advanceBio/largedata/data_alllabel.csv', delimiter=",", dtype=int) # cols: label, ID, race, mirnas...
data = myfile[1:, 3:]
num_feature = data.shape[1]
print('data done!')
# print('num_feature: %d' % num_feature)
# data = myfile[1:, 1:]  # data: n_samples * n_features array; LAST COL OF CSV IS nan (after the last comma)
labels_surv = myfile[1:, 1] # alive or dead
labels_stage = myfile[1:, 2] # stage i, ii, iii, iv
labels_race = myfile[1:, 0] # asian:0, black:1, white:2

tot_sample = 335
tot_alive = 298
tot_dead = 37
tot_i = 63
tot_ii = 178
tot_iii = 81
tot_iv = 13
tot_asian = 10
tot_black = 50
tot_white = 275

estimators = [('SVC_linear', svm.SVC(kernel='linear', gamma='auto'))]
for num_cluster in [2, 3, 4, 5, 6, 7, 8]:
# for num_cluster in [2]:
    # print("num_cluster = %d" % num_cluster)
    # estimators = [('SVC_linear', svm.SVC(kernel='linear', gamma='auto'))]
                  # ('RandomForest_equalweight', RandomForestClassifier(n_estimators=10)),
                  # ('Neural_network', MLPClassifier(alpha=1))
                  # ('logstic_regression', LogisticRegression(solver='lbfgs', n_jobs=2, multi_class='auto')),
                  # ('SVC_poly3', svm.SVC(kernel='poly', degree=3, gamma='auto')),
                  # ('RandomForest_balanceweight', RandomForestClassifier(n_estimators=10, class_weight='balanced_subsample')),
    print("#####################   kmeans n_cluster = %d     #####################" % num_cluster)
    clf = KMeans(n_clusters=num_cluster, n_init=10, n_jobs=2).fit(data)
    label_kmeans = clf.labels_
    np.savetxt("totcluster%d_label_new.csv" % num_cluster, label_kmeans, fmt="%d", delimiter=",")
    #--------------------------------------------------------------------
    print('======== Label Distribution in Clusters ==========')
    count_asian = np.zeros(num_cluster)
    count_black = np.zeros(num_cluster)
    count_white = np.zeros(num_cluster)
    count_alive = np.zeros(num_cluster)
    count_dead = np.zeros(num_cluster)
    count_i = np.zeros(num_cluster)
    count_ii = np.zeros(num_cluster)
    count_iii = np.zeros(num_cluster)
    count_iv = np.zeros(num_cluster)
    tot_cluster = np.zeros(num_cluster)
    for j in range(len(label_kmeans)):
        labl = label_kmeans[j]
        tot_cluster[labl] += 1
        #---------------------------------------------
        race = labels_race[j]
        if  race == 0:
            count_asian[labl] += 1
        elif race == 1:
            count_black[labl] += 1
        else:
            count_white[labl] += 1
        #---------------------------------------------
        surv = labels_surv[j]
        if  surv == 1:                 # dead
            count_dead[labl] += 1
        else:
            count_alive[labl] += 1
        #---------------------------------------------
        stage = labels_stage[j]
        if stage == 0:
            count_i[labl] += 1
        elif stage == 1:
            count_ii[labl] += 1
        elif stage == 2:
            count_iii[labl] += 1
        elif stage == 3:
            count_iv[labl] += 1
    for k in range(num_cluster):
        print('cluster %d has %d samples, %1.2f in total, (intra-cluster below)' % (k, tot_cluster[k], tot_cluster[k]/tot_sample))
        print('asian: %.2f, black: %.2f, white: %.2f' % (count_asian[k]/tot_cluster[k], count_black[k]/tot_cluster[k], count_white[k]/tot_cluster[k]))
        print('alive: %.2f, dead: %.2f' % (count_alive[k]/tot_cluster[k], count_dead[k]/tot_cluster[k]))
        print('stage i: %.2f, stage ii: %.2f, stage iii: %.2f, stage iv: %.2f\n'
              % (count_i[k]/tot_cluster[k], count_ii[k]/tot_cluster[k], count_iii[k]/tot_cluster[k], count_iv[k]/tot_cluster[k]))
        # print('cluster %d (inter-cluster)' % k)
        # print('asian: %.2f, black: %.2f, white: %.2f' % (count_asian[k]/tot_asian, count_black[k]/tot_black, count_white[k]/tot_white))
        # print('alive: %.2f, dead: %.2f' % (count_alive[k]/tot_alive, count_dead[k]/tot_dead))
        # print('stage i: %.2f, stage ii: %.2f, stage iii: %.2f, stage iv: %.2f'
        #       % (count_i[k]/tot_i, count_ii[k]/tot_ii, count_iii[k]/tot_iii, count_iv[k]/tot_iv))
    #---------------------------------------------------------------------------------------------------------------------------------------
    for i in range(1):
        print("===============  supervised (%d labels): %d ===========" % (num_cluster,i))
        data_train, data_test, label_train, label_test = train_test_split(data, label_kmeans, test_size=0.25, random_state=i, shuffle=True)

        count_train = [0]*num_cluster
        count_test = [0]*num_cluster
        for k in label_train:
            count_train[k] += 1
        for k in label_test:
            count_test[k] += 1
        for k in range(num_cluster):
            sz0 = (count_train[k]+count_test[k])/(label_train.shape[0]+label_test.shape[0])
            sz1 = count_train[k]/label_train.shape[0]
            sz2 = count_test[k]/label_test.shape[0]
            print("label %d in total dataset: %.2f" % (k, sz0))
            print("label %d in training dataset: %.2f, in testing dataset: %.2f" % (k, sz1, sz2))
            print()
        #-------------------------------------------------------------
        with open('SVMlinear_coef_cluster%d.csv' % num_cluster, "w") as f:
            f.write('================================== cluster %d ==============================\n' % num_cluster)
            for name, est in estimators:
                est.fit(data_train, label_train)
                label_pred = est.predict(data_test)
                print ('------ %s (%d labels) ------' % (name,num_cluster))
                f.write('***** %s ******\n' % name)
                # print("Accuracy: %1.2f" % accuracy_score(label_test, label_pred))
                # print("Precision:\n%s" % precision_score(label_test, label_pred, average=None))
                # print("Recall:\n%s" % recall_score(label_test, label_pred, average=None))
                # print("F1:\n%s" % f1_score(label_test, label_pred, average=None))
                cm = confusion_matrix(label_test, label_pred)
                # print("Confusion matrix:\n%s" % cm)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized Confusion matrix:\n%s" % cm)
                #------------ save top 10 genes ---------------
                num_top = 10
                # coef_matrix = np.array(est.coef_)
                coef_matrix = est.coef_
                count_line = 0
                for aa in range(num_cluster):
                    for bb in range(aa+1,num_cluster):
                        if (tot_cluster[aa] > 4) and (tot_cluster[bb] > 4):
                            tmp = -abs(coef_matrix[count_line,:])
                            sort_gene = tmp.argsort()
                            sort_coef = abs(tmp[sort_gene])
                            f.write('------------------- c%d, c%d ----------------\n' % (aa,bb))
                            f.write('%20s,%10s\n' % ('gene','coef'))
                            for indx in range(num_top):
                                f.write('%20s,' % dict_num_gene[sort_gene[indx]])
                                f.write('%10.2e\n' % sort_coef[indx])
                        count_line += 1

            # print()

    #----------------------- calculate mean, var -----------------
    #   gene              cluster 0, cluster 1,...
    # feature 0 mean,
    # std,
    # feature 1 mean,
    # std,
    # with open("/home/hju/advanceBio/largedata/totcluster%d_meanstd_new.csv" % num_cluster, 'w') as ff:
    #     writer = csv.writer(ff)
    #     writer.writerows([["gene"]+list(range(num_cluster))])
    #     for i in range(num_feature):
    #         avg = np.zeros(num_cluster)
    #         var = np.zeros(num_cluster)
    #         for j in range(num_cluster):
    #             tmp = data[label_kmeans==j, i]
    #             avg[j] = np.mean(tmp)
    #             var[j] = np.std(tmp,ddof=1)
    #             # print(gene_name[i])
    #         writer.writerows([[gene_name[i]+"_mean"]+list(avg)])
    #         writer.writerows([[gene_name[i]+"_std"]+list(var)])
    # # print("stats saved")
    # # print(label_kmeans)
    #-------------------------------------------------------------

# with open('/home/hju/advanceBio/largedata/data_alllabel.csv','r') as f:
#     gene_name = f.readline().rstrip('\n').split(',')
#     gene_name = gene_name[3:]

# print(gene_name)
# print(gene_name[0])
# print(gene_name[-1])
# print('num_gene_name: %d' % len(gene_name))
