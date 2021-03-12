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
# for num_cluster in [4,5,6,7,8]:
# for num_cluster in [2]:
    print('============= num_cluster %d ==============' % num_cluster)
    label_kmeans = np.loadtxt('totcluster%d_label_new.csv' % num_cluster)
    mystats = np.zeros([int(num_feature*num_cluster*(num_cluster-1)/2), 11])
    # print(mystats.shape)
    cc = 0
    # mystats = np.zeros(num_feature, dtype='|S25,int8,int8,float32,float32,float32,float32,float32,float32'))
    # print('gene_name,cluster,cluster,dist_mean/sum_std,t-value,mean1,mean2,std1,std2,tot1,tot2' )
    for i in range(num_feature):
        #------- calculate mean, std for each cluster ------------
        avg = np.zeros(num_cluster)
        var = np.zeros(num_cluster)
        tot_in_cluster = np.zeros(num_cluster)
        for j in range(num_cluster):
            tmp = data[label_kmeans==j, i]
            avg[j] = np.mean(tmp)
            var[j] = np.std(tmp)
            tot_in_cluster[j] = tmp.shape[0]
        #------- calculate tt, rr for each pair of clusters ------------
        for aa in range(num_cluster):
            if tot_in_cluster[aa] > 4:
                data_aa = data[label_kmeans==aa, i]
                for bb in range(aa+1, num_cluster):
                    if tot_in_cluster[bb] > 4:
                        data_bb = data[label_kmeans==bb, i]
                        if (var[aa]+var[bb]) == 0:
                            rr = np.nan
                        else:
                            rr = abs(avg[aa]-avg[bb])/(var[aa]+var[bb])
                        try:
                            tt, p = stats.ttest_ind(data_aa, data_bb)
                        except:
                            tt = np.nan
                            # 'gene_name,cluster,cluster,dist_mean/sum_std,t-value,mean1,mean2,std1,std2,tot1,tot2'
                        mystats[cc,:] = [i, aa, bb, rr, tt, avg[aa], avg[bb], var[aa], var[bb],tot_in_cluster[aa],tot_in_cluster[bb]]
                        cc += 1
    #---------------- SORT according to rr & tt ---------------
    if cc == 0:
        print('cluster %d generates no file' % num_cluster)
    else:
        with open('sort_rr_cluster%d.csv' % num_cluster, "w") as f:
            f.write('==================== cluster %d =================\n' % num_cluster) #dist_mean/sum_std
            f.write('%20s,%8s,%8s,%10s,%10s,%10s,%10s,%10s,%10s,%5s,%5s\n' % ('gene_name','cluster','cluster','rr','t-value','mean1','mean2','std1','std2','tot1','tot2'))
            #------------- save genes with rr > 1.0 ----------------
            for kk in range(cc):
                if mystats[kk,3] > 1.0:
                    f.write('%20s,' % dict_num_gene[mystats[kk,0]])
                    for pp in mystats[kk,1:3]:
                        f.write('%8d,' % pp)
                    for pp in mystats[kk,3:5]:
                        f.write('%10.2f,' % pp)
                    for pp in mystats[kk,5:-2]:
                        f.write('%10.2E,' % pp)
                    f.write('%5d,' % mystats[kk,-2])
                    f.write('%5d\n' % mystats[kk,-1])

        # sort_mystats = mystats[mystats[:,4].argsort()]
        # cc_new = np.isnan(sort_mystats[:,4]).argmax(axis=0)
        # sort_mystats = sort_mystats[:cc_new,:]
        # with open('sort_tt_cluster%d.csv' % num_cluster, "w") as f:
        #     f.write('==================== cluster %d =================\n' % num_cluster) #dist_mean/sum_std
        #     f.write('gene_name           , cluster, cluster, rr       , t-value  , mean1    , mean2    , std1     , std2     , tot1, tot2\n')
        #     for kk in reversed(range(cc_new)):
        #         f.write('%-20s,' % dict_num_gene[sort_mystats[kk,0]])
        #         for pp in sort_mystats[kk,1:3]:
        #             f.write('%8d,' % pp)
        #         for pp in sort_mystats[kk,3:-2]:
        #             f.write('%10.2f,' % pp)
        #         f.write('%5d,' % sort_mystats[kk,-2])
        #         f.write('%5d\n' % sort_mystats[kk,-1])



        # w = csv.writer(open('/home/hju/advanceBio/largedata/sort_rr_cluster%d.csv' % num_cluster, "w"))
        # w.writerow(['gene_name','cluster','cluster','dist_mean/sum_std','t-value','mean1','mean2','std1','std2'])
        # for kk in reversed(range(cc)):
        #     w.writerow([dict_num_gene[mystats[kk,0]], mystats[kk,1:]])
