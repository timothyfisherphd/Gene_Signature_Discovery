#!/usr/bin/python3
import os
import csv
import scipy.stats as stats
import numpy as np

with open('/home/hju/advanceBio/largedata/data_alllabel.csv','r') as f:
    gene_name = f.readline().rstrip('\n').split(',')
    gene_name = gene_name[3:]
print('gene_name done!')

myfile = np.genfromtxt('/home/hju/advanceBio/largedata/data_alllabel.csv', delimiter=",", dtype=int) # cols: label, ID, race, mirnas...
data = myfile[1:, 3:]
num_feature = data.shape[1]
print('data done!')

dict_num_gene = {}
nnarray = np.arange(num_feature, dtype=int)
for NN, GG in zip(nnarray, gene_name):
    dict_num_gene[NN] = GG
print('dict_num_gene done!')
#--------------------------------------------------------------------
for num_cluster in [2,3,4,5,6,7,8]:
# for num_cluster in [2]:
    print('============= num_cluster %d ==============' % num_cluster)
    label_kmeans = np.loadtxt('totcluster%d_label_new.csv' % num_cluster)
    tot_in_cluster = np.zeros(num_cluster)
    for vv in range(num_cluster):
        tot_in_cluster[vv] = label_kmeans[label_kmeans == vv].shape[0]
    with open('sort_rr_cluster%d_cc.csv' % num_cluster, "w") as f:
        f.write('==================== cluster %d =================\n' % num_cluster)
        #------- calculate tt, rr for each pair of clusters ------------
        for aa in range(num_cluster):
            if tot_in_cluster[aa] > 4:
                for bb in range(aa+1, num_cluster):
                    if tot_in_cluster[bb] > 4:
                        cc = 0
                        mystats = np.zeros([num_feature, 11])
                        for i in range(num_feature):
                            #------- calculate mean, std -----------
                            data_aa = data[label_kmeans==aa,i]
                            avg_aa = np.mean(data_aa)
                            var_aa = np.std(data_aa)
                            data_bb = data[label_kmeans==bb,i]
                            avg_bb = np.mean(data_bb)
                            var_bb = np.std(data_bb)
                            if (var_aa+var_bb) == 0:
                                rr = np.nan
                            else:
                                rr = abs(avg_aa-avg_bb)/(var_aa+var_bb)
                            try:
                                tt, p = stats.ttest_ind(data_aa, data_bb)
                            except:
                                tt = np.nan
                                # 'gene_name,cluster,cluster,dist_mean/sum_std,t-value,mean1,mean2,std1,std2,tot1,tot2'
                            mystats[cc,:] = [i, aa, bb, rr, tt, avg_aa, avg_bb, var_aa, var_bb,tot_in_cluster[aa],tot_in_cluster[bb]]
                            cc += 1
                        #---------------- SORT according to rr  ---------------
                        sort_mystats = mystats[mystats[:,3].argsort()]
                        cc_new = np.isnan(sort_mystats[:,3]).argmax(axis=0)
                        if cc_new==0:
                            print('c%d c%d generates no file' % (aa,bb))
                        else:
                            # print('check nan')
                            # print(sort_mystats[cc_new-1,:])
                            # print(sort_mystats[cc_new,:])
                            f.write('==================== c%d, c%d =================\n' % (aa,bb))
                            f.write('%20s,%8s,%8s,%10s,%10s,%10s,%10s,%10s,%10s,%5s,%5s\n' % ('gene_name','cluster','cluster','rr','t-value','mean1','mean2','std1','std2','tot1','tot2'))
                            # f.write('gene_name           , cluster, cluster, rr       , t-value  , mean1    , mean2    , std1     , std2     , tot1, tot2\n')
                            #------------- save top 10 genes ----------------
                            for kk in range(cc_new-1, cc_new-1-10, -1):
                                f.write('%20s,' % dict_num_gene[sort_mystats[kk,0]])
                                for pp in sort_mystats[kk,1:3]:
                                    f.write('%8d,' % pp)
                                for pp in sort_mystats[kk,3:5]:
                                    f.write('%10.2f,' % pp)
                                for pp in sort_mystats[kk,5:-2]:
                                    f.write('%10.2E,' % pp)
                                f.write('%5d,' % sort_mystats[kk,-2])
                                f.write('%5d\n' % sort_mystats[kk,-1])
