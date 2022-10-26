import networkx as nx
# from python_paris import paris
from scipy.io import loadmat, savemat
# import matplotlib.pyplot as plt
import numpy as np
# from sklearn.cluster import AgglomerativeClustering
from sknetwork.clustering import Louvain, modularity
from sknetwork.hierarchy import Paris, tree_sampling_divergence, dasgupta_score, cut_straight

from scipy.sparse import csr_matrix
import sys
import mat73



def fit_louvain():

    X = dat_in['po']

    # n,foo,nsmp = X.shape
    nsmp = X.shape[0]

    # loop through each sample
    M = []
    LAB = []
    for ismp in range(0,nsmp):
        if ismp % np.ceil(nsmp*0.1) == 0:
            print('%d %%' % (ismp/nsmp*100))
        # xtmp = X[:,:,ismp]
        xtmp = X[ismp]
        sz=xtmp.shape

        # reformat
        row = []
        col = []
        data = []
        for ii in range(0,sz[0]):
            for jj in range(0,sz[1]):
                x = xtmp[ii, jj]
                if x > 0:
                    row.append(ii)
                    col.append(jj)
                    data.append(x)
                # edges.append((ii + 1, jj + 1, 1))
        C = csr_matrix((data, (row, col)), shape=sz)

        # fit
        louvain = Louvain()
        # adjacency = karate_club()
        labels = louvain.fit_transform(C)

        # some stats
        m = modularity(C, labels)

        # store
        M.append(m)
        LAB.append(labels)


    # save
    out = {
        'labels':LAB,
        'modularity':M,
    }
    savemat(name_out,out)


def fit_paris():
    foo=1

    X = dat_in['po']
    nsmp = len(X)

    # loop through each sample
    dend = []
    TSD = []
    DGS = []
    CUTS_lab = [];
    CUTS_n = [];
    MOD = []

    for ismp in range(0, nsmp):
        if ismp % np.ceil(nsmp * 0.1) == 0:
            print('%d %%' % (ismp / nsmp * 100))
        xtmp = X[ismp]
        sz = xtmp.shape

        # reformat
        row = []
        col = []
        data = []
        for ii in range(0, sz[0]):
            for jj in range(0, sz[1]):
                x = xtmp[ii, jj]
                if x > 0:
                    row.append(ii)
                    col.append(jj)
                    data.append(x)
                # edges.append((ii + 1, jj + 1, 1))
        C = csr_matrix((data, (row, col)), shape=sz)

        # paris
        paris = Paris()
        d = paris.fit_transform(C)
        
        # make multiple cuts
        ncut = range(2,sz[0]);
        cutlab = [];
        mod = [];
        for nc in ncut:
            # make a cut
            lab = cut_straight(d,n_clusters=nc)
            cutlab.append(lab)

            # modularity
            m = modularity(C, lab)
            mod.append(m)

        # store
        dend.append(d)
        TSD.append(tree_sampling_divergence(C,d))
        DGS.append(dasgupta_score(C,d))
        CUTS_lab.append(cutlab)
        CUTS_n.append(ncut)
        MOD.append(mod)
        

    # save
    out = {
        'dendrogram':dend,
        'tree_sampling_divergence':TSD,
        'dasgupta_score':DGS,
        'modularity':MOD,
        'ncut': CUTS_n,
        'labels_cut':CUTS_lab
    }
    savemat(name_out, out)


def get_metrics():
    # init
    out = {}

    # which metrics?
    theseMetrics = dat_in['metrics']

    # prep
    if any(x in ['dgs','tds'] for x in theseMetrics):
        # prep adjancency
        X = dat_in['po']
        n, foo, nsmp = X.shape

        A = []
        for ismp in range(0, nsmp):
            if ismp % np.ceil(nsmp * 0.1) == 0:
                print('%d %%' % (ismp / nsmp * 100))
            xtmp = X[:, :, ismp]

            # reformat
            row = []
            col = []
            data = []
            for ii in range(0, n):
                for jj in range(0, n):
                    x = xtmp[ii, jj]
                    if x > 0:
                        row.append(ii)
                        col.append(jj)
                        data.append(x)
                    # edges.append((ii + 1, jj + 1, 1))
            a = csr_matrix((data, (row, col)), shape=(n, n))
            A.append(a)

        # prep dendrogram
        D = dat_in['dendrogram']


    # TDS
    if 'tds' in theseMetrics:
        TDS = []
        d=1
        for ii in range(0,nsmp):
            t=tree_sampling_divergence(A[ii], D[:,:,ii])
            TDS.append(t)
        out['tree_sampling_divergence'] = TDS
        foo=1

    # dasgupta
    if 'dgs' in theseMetrics:
        DGS = []
        d=1
        for ii in range(0,nsmp):
            t=dasgupta_score(A[ii], D[:,:,ii])
            DGS.append(t)
        out['dasgupta_score'] = DGS
        foo=1

    # modularity

    # save
    savemat(name_out, out)

if __name__ == '__main__':
    # laod data
    name_in = sys.argv[1]
    name_out = sys.argv[2]

    dat_in = mat73.loadmat(name_in)

    func = dat_in['func']

    # call
    print('data: {}'.format(name_in))
    if func == 'fit_louvain':
        print('fitting with Louvain...')
        fit_louvain()
    elif func == 'fit_paris':
        print('fitting with Paris...')
        fit_paris()
    elif func == 'get_metrics':
        print('getting metrics...')
        get_metrics()
