#!/usr/bin/python3
import os
import csv

# surv_yrs = [] # survived years after diagnosed
# age_at_rcrd = [] # age at recording
# age_at_diag = [] # age at diagnosed
yr_record = 2019
ww = csv.writer(open("/home/hju/advanceBio/largedata/survival_years.csv", "w"))
ww.writerow(["race", "surv", "age_at_diag", "surv_yrs"])
for i in ["asian","black","white"]:
    for j in ["alive","dead"]:
        print("*************** %s-%s *************** " % (i,j))
        with open("/home/hju/advanceBio/largedata/" + i + "/" + j + "/other/clinical.tsv","r") as f:
            next(f)
            for line in f:
                tmp = line.split('\t')
                # print(tmp)
                if tmp[12] != "--":
                    # print(tmp)
                    t0 = int(tmp[12])/365
                    t1 = yr_record - int(tmp[4])
                    # age_at_diag.append(t0)
                    # age_at_rcrd.append(t1)
                    # surv_yrs.append(t1 - t0)
                    # print("age_at_diag: %d" % t0)
                    # print("surv_yrs: %d" % (t1 - t0))
                    ww.writerow([i, j, int(t0), int(t1 - t0)])
