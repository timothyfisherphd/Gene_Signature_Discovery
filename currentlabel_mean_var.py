#!/usr/bin/python3
import numpy as np
import csv

with open('data_alllabel.csv','r') as f:
    gene_name = f.readline().rstrip('\n').split(',')
    gene_name = gene_name[3:]

myfile = np.genfromtxt('data_alllabel.csv', delimiter=",", dtype=int) # cols: label, ID, race, mirnas...
data = myfile[1:, 3:]
num_feature = data.shape[1]
labels_surv = myfile[1:, 1] # alive or dead
labels_stage = myfile[1:, 2] # stage i, ii, iii, iv
labels_race = myfile[1:, 0] # asian:0, black:1, white:2

#------------------------ surv ----------------------------------
num_cluster = 2  ##
with open("surv_meanstd.csv", 'w') as ff:
    writer = csv.writer(ff)
    writer.writerows([["gene"]+list(range(num_cluster))])
    for i in range(num_feature):
        avg = np.zeros(num_cluster)
        var = np.zeros(num_cluster)
        for j in range(num_cluster):
            tmp = data[labels_surv==j, i]   ##
            avg[j] = np.mean(tmp)
            var[j] = np.std(tmp)
            # print(gene_name[i])
        writer.writerows([[gene_name[i]+"_mean"]+list(avg)])
        writer.writerows([[gene_name[i]+"_std"]+list(var)])

#------------------------ stage ----------------------------------
num_cluster = 4   ##
with open("stage_meanstd.csv", 'w') as ff:
    writer = csv.writer(ff)
    writer.writerows([["gene"]+list(range(num_cluster))])
    for i in range(num_feature):
        avg = np.zeros(num_cluster)
        var = np.zeros(num_cluster)
        for j in range(num_cluster):
            tmp = data[labels_stage==j, i]    ##
            avg[j] = np.mean(tmp)
            var[j] = np.std(tmp)
            # print(gene_name[i])
        writer.writerows([[gene_name[i]+"_mean"]+list(avg)])
        writer.writerows([[gene_name[i]+"_std"]+list(var)])

#------------------------ race ----------------------------------
num_cluster = 3  ##
with open("race_meanstd.csv", 'w') as ff:
    writer = csv.writer(ff)
    writer.writerows([["gene"]+list(range(num_cluster))])
    for i in range(num_feature):
        avg = np.zeros(num_cluster)
        var = np.zeros(num_cluster)
        for j in range(num_cluster):
            tmp = data[labels_race==j, i]    ##
            avg[j] = np.mean(tmp)
            var[j] = np.std(tmp)
            # print(gene_name[i])
        writer.writerows([[gene_name[i]+"_mean"]+list(avg)])
        writer.writerows([[gene_name[i]+"_std"]+list(var)])
