#!/usr/bin/python3
import os
import csv

# alive:0, dead: 1; asian:0, black:1, white:2; stage i:0, ii:1, iii:2, iv:3
dict_ID_mirna = {}
dict_ID_rna = {}
dict_ID_surv = {}
dict_ID_stage = {}
# dict_mirna_race_surv = {}
dict_mirna_race = {}
race_dict = {"asian":0, "black":1, "white":2}
newdict = {}

# dict_ID_all = {}

#-------- dict_ID_mirna/rna ----------
dir = "/home/hju/advanceBio/largedata/sample_sheet/"
#os.chdir(dir)
filelist = os.listdir(dir)
for ftsv in filelist:
    with open(dir + ftsv, "r") as f:
        print("sample_sheet")
        print(ftsv)
        next(f)
        # print(ftsv)
        for line in f:
            tmp = line.split('\t')
            print(tmp)
            case_ID = tmp[5]
            print("case_ID:")
            print(case_ID)
            #------------------------
            fname = tmp[1]
            print("fname:")
            print(fname)
            if fname[-9:] == "counts.gz":
                fname = fname[:-3]
                dict_ID_rna[case_ID] = fname
                print("rnaseq data")
            elif fname[-25:] == "mirnas.quantification.txt":
                dict_ID_mirna[case_ID] = fname
                print("mirna data")
#--------- dict_ID_surv --------------
yr_record = 2013
# ww = csv.writer(open("/home/hju/advanceBio/largedata/survival_years_2013.csv", "w"))
# ww.writerow(["race", "surv", "age_at_diag", "surv_yrs"])
count_asian = 0
count_black = 0
count_white = 0
count_alive = 0
count_dead = 0
count_i = 0
count_ii = 0
count_iii = 0
count_iv = 0
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
                    case_ID = tmp[1]
                    stage = tmp[11].split(' ')[1]
                    # print(stage)
                    t0 = int(tmp[12])/365        # age_at_diag.append(t0)
                    t1 = yr_record - int(tmp[4]) # age_at_rcrd.append(t1)
                #---------------------------------------------
                    if  j == "dead":                 # "dead"
                        if int(t1 - t0) < 5:
                            # count_dead += 1
                            dict_ID_surv[case_ID] = 1
                        else:
                            # count_alive += 1
                            dict_ID_surv[case_ID] = 0
                    elif int(t1 - t0) >= 5:          # "alive"
                        # count_alive += 1
                        dict_ID_surv[case_ID] = 0
                #---------------------------------------------
                    if stage == "i" or stage == "ia" or stage == "ib" or stage == "ic":
                        dict_ID_stage[case_ID] = 0
                        # count_i += 1
                    elif stage == "ii" or stage == "iia" or stage == "iib" or stage == "iic":
                        dict_ID_stage[case_ID] = 1
                        # count_ii += 1
                    elif stage == "iii" or stage == "iiia" or stage == "iiib" or stage == "iiic":
                        dict_ID_stage[case_ID] = 2
                        # count_iii += 1
                    elif stage == "iv" or stage == "iva" or stage == "ivb" or stage == "ivc":
                        dict_ID_stage[case_ID] = 3
                        # count_iv += 1
                    else:
                        print("stage missing")

#-------- dict_mirna_race ----------
print("MAKE dict_mirna_race")
for i in ["asian","black","white"]:
    for j in ["alive","dead"]:
        dir = "/home/hju/advanceBio/largedata/" + i + "/" + j + "/mirna/"
        #os.chdir(dir)
        print(dir)
        filelist = os.listdir(dir)
        for k in filelist:
            dict_mirna_race[k] = race_dict[i]

#-------- dict_ID for all 4 items ----------------
print("combine 5 dict")
for key in dict_ID_mirna:
    if (key in dict_ID_rna) and (key in dict_ID_surv) and (key in dict_ID_stage):
        # print("item")
        mirna_f = dict_ID_mirna[key]
        rna_f = dict_ID_rna[key]
        surv = dict_ID_surv[key]
        stage = dict_ID_stage[key]
        race = dict_mirna_race[mirna_f]
        newdict[key] = [race, surv, stage, mirna_f, rna_f]
        #---------------------------------------------
        if  race == 0:
            count_asian += 1
        elif race == 1:
            count_black += 1
        else:
            count_white += 1
        #---------------------------------------------
        #---------------------------------------------
        if  surv == 1:                 # dead
            count_dead += 1
        else:
            count_alive += 1
        #---------------------------------------------
        if stage == 0:
            count_i += 1
        elif stage == 1:
            count_ii += 1
        elif stage == 2:
            count_iii += 1
        elif stage == 3:
            count_iv += 1

tot_stage = count_i+count_ii+count_iii+count_iv
print("tot_stage: %d" % tot_stage)
print("tot_surv: %d" % (count_alive+count_dead))
print("count_alive: %d, %1.2f" % (count_alive, float(count_alive)/float(count_alive+count_dead)))
print("count_dead: %d, %1.2f" % (count_dead, float(count_dead)/float(count_alive+count_dead)))
print("count_stage i: %d, %1.2f" % (count_i, float(count_i)/float(tot_stage)))
print("count_stage ii: %d, %1.2f" % (count_ii, float(count_ii)/float(tot_stage)))
print("count_stage iii: %d, %1.2f" % (count_iii, float(count_iii)/float(tot_stage)))
print("count_stage iv: %d, %1.2f" % (count_iv, float(count_iv)/float(tot_stage)))
print("count_asian: %d, %1.2f" % (count_asian, float(count_asian)/float(count_asian+count_black+count_white)))
print("coun_black: %d, %1.2f" % (count_black, float(count_black)/float(count_asian+count_black+count_white)))
print("count_white: %d, %1.2f" % (count_white, float(count_white)/float(count_asian+count_black+count_white)))
# print(filelist)
#==============================
w = csv.writer(open("dict_ID_alllabel.csv", "w"))
# w.writerow(['case_ID', 'race', 'surv','stage', 'mirna_f', 'rna_f'])
for key, val in newdict.items():
    w.writerow([key, val[0], val[1], val[2], val[3], val[4]])
print("csv done")
