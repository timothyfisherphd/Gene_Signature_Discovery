#!/usr/bin/python3
import os
import csv

dict_myclass = {'surv':2,'stage':4,'race':3}
for myclass in ['surv','stage','race']:
    num_cluster = dict_myclass[myclass]
    print('============== %s ==============' % myclass)
    print('gene_name,type,type,dist_mean/sum_std' )
    print('gene_name,mean1,mean2,std1,std2' )
    with open(myclass + "_meanstd.csv","r") as f:
        next(f)
        cc = 0
        for line in f:
            tmp = line.split(',')
            # print(tmp)
            if cc%2 == 0:
                gene_name = tmp[0].split('_')[0]
                mean_list = [float(qq) for qq in tmp[1:]]
            else:
                var_list = [float(qq) for qq in tmp[1:]]
                for i in range(num_cluster-1):
                    for j in range(i+1, num_cluster):
                        try:
                            pp = abs(mean_list[i]-mean_list[j])/(var_list[i]+var_list[j])
                            if pp > 1:
                                print('%s,%d,%d,%1.2f' % (gene_name,i,j,pp))
                                print('%s,%1.2f,%1.2f,%1.2f,%1.2f' % (gene_name,mean_list[i],mean_list[j],var_list[i],var_list[j]))
                        except:
                            pass
            cc += 1
