#!/usr/bin/python3

num_topgene = 10
period = 2+num_topgene

for i in range(2,9):
# for i in [3]:
    f_ratio = open("sort_rr_cluster%d_cc.csv" % i, "r")
    f_svm = open("SVMlinear_coef_cluster%d.csv" % i, "r")
    next(f_ratio)
    next(f_svm)
    next(f_svm)
    for j in range(0,100):
        tmp = f_ratio.readline()
        if tmp == '':
            break
        else:
            tmp = tmp.split(' ')
            # print(tmp)
            clusterA = tmp[1].strip()[:-1]  # discard comma
            clusterB = tmp[2].strip()
            ff = open("overlap_cluster%d_%s%s.csv" % (i,clusterA,clusterB), "w")
            next(f_ratio)
            next(f_svm)
            next(f_svm)
            gene_ratio = []
            gene_svm = []
            for kk in range(0,num_topgene):
                tmp = f_ratio.readline().split(',')[0].strip()
                if tmp[:4] == 'ENSG':
                    gene_ratio.append(tmp)
            for kk in range(0,num_topgene):
                tmp = f_svm.readline().split(',')[0].strip()
                # print(tmp)
                if tmp[:4] == 'ENSG':
                    gene_svm.append(tmp)
            # print('c%s c%s' % (clusterA,clusterB))
            # print(gene_ratio)
            # print(gene_svm)
            for mygene in gene_ratio:
                if mygene in gene_svm:
                    ff.write('%s\n' % mygene)
            ff.close()
