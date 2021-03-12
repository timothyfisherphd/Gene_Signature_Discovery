#!/usr/bin/python3
import os
import csv

dir_mirna = '/home/hju/advanceBio/largedata/mirna_all/'
dir_rna = '/home/hju/advanceBio/largedata/rnaseq_all/'
# filelist = os.listdir(dir)
# ---------- get miRNA names --------------------------
mRNAlist = []
filelist_mirna = os.listdir(dir_mirna)
# print(filelist_mirna[0])
with open(dir_mirna + filelist_mirna[0], "r") as f:
    next(f)
    for line in f:
        tmp = line.split()
        mRNAlist.append(tmp[0])
        #count.append(tmp[1])

# ---------- get RNA names --------------------------
rnalist = []
filelist_rnaseq = os.listdir(dir_rna)
with open(dir_rna + filelist_rnaseq[0], "r") as f:
    line_count = 1
    for line in f:
        if line_count < 60484:
            tmp = line.split()
            rnalist.append(tmp[0])
            line_count += 1
        else:
            break
#=========== read in dict_ID_all ============
dict_ID_all = {}
# 'case_ID', 'race', 'surv', 'mirna_f', 'rna_f'
with open('dict_ID_alllabel.csv', "r") as f:
    for line in f:
        case_ID, race, surv, stage, mirna_f, rna_f = line.split(',')
        dict_ID_all[case_ID] = [race, surv, stage, mirna_f, rna_f]

# ============================================
with open('data_alllabel.csv', "w") as f:
    # f.write('label,')
    # f.write('case_ID,')
    f.write('race,surv,stage,')
    # write mRNA names
    for i in mRNAlist:
        f.write(i+',')
    # write rna names
    cc = 1
    for i in rnalist:
        if cc < 60483:
            f.write(i+',')
        else:
            f.write(i+'\n')
        cc = cc+1
    # ----------------------------------------
    for ID in dict_ID_all:
        print(ID)
        race, surv, stage, mirna_f, rna_f = dict_ID_all[ID]
        # print(mirna_f)
        # print(rna_f)
        if (mirna_f in filelist_mirna) and (rna_f[:-1] in filelist_rnaseq):
            print("open mirna rna")
            # print(mirna_f)
            # f.write('%s,' % surv) # label
            f.write(race+","+surv+","+stage+",")
            #------------- miRNA ------------------
            with open(dir_mirna + mirna_f, "r") as f_data:
                next(f_data)
                for line in f_data:
                    tmp = line.split()
                    f.write('%d,' % int(tmp[1]))
            #-------------- rna -----------------
            with open(dir_rna + rna_f[:-1], "r") as f_data:
                line_count = 1
                for line in f_data:
                    if line_count < 60484:
                        tmp = line.split()
                        # print("rna value")
                        # print(tmp)
                        if line_count != 60483:
                            f.write('%d,' % int(tmp[1]))
                        else:
                            f.write('%d\n' % int(tmp[1]))
                        line_count +=1
                    else:
                        break
        else:
            print(mirna_f)
            print(rna_f)
