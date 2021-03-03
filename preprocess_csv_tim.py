#!/usr/bin/python3
import os
import csv
dict_ID = {}
id = 0

dir = '/Volumes/Timothy_Backup/Advanced_Bioinformatics/asian/alive/mirna/'
filelist = os.listdir(dir)

mRNAlist = []
with open(dir+filelist[0], "r") as f:
    next(f)
    for line in f:
        tmp = line.split()
        mRNAlist.append(tmp[0])
        #count.append(tmp[1])

w = csv.writer(open("result.csv", "w"))
for key, val in dict_ID.items():
    w.writerow([key, val])

with open('result.csv', "w") as f:
    f.write('{:<15s}'.format('label'))
    f.write('{:<15s}'.format('ID'))
    f.write('{:<15s}'.format('race'))
    # write mRNA names
    for i in mRNAlist:
        f.write('{:<15s}'.format(i))
    f.write('\n')
    # ==============================
    for j in filelist:
        race, label, other = j.split('_')
        #---------------
        if label == "alive":
            label = 0
        else:
            label = 1
        #---------------
        lst = other.split('-')
        ID = lst[0]
        dict_ID[ID] = id
        ID = id
        id = id + 1
        #---------------
        count = []
        #-------------------------------
        with open(dir+j, "r") as f_data:
            print (j)
            f.write('{:<15d}'.format(label))
            f.write('{:<15d}'.format(ID))
            f.write('{:<15s}'.format(race))
            #-------------------------------
            next(f_data)
            for line in f_data:
                tmp = line.split()
                count.append(tmp[1])
        #-------------------------------
        for i in count:
            f.write('{:<15s}'.format(i))
        f.write('\n')

#==============================
w = csv.writer(open("dict_ID_mirna_csv.csv", "w"))
for key, val in dict_ID.items():
    w.writerow([key, val])
