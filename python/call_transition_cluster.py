import networkx as nx
# from python_paris import paris
from scipy.io import loadmat, savemat
# import matplotlib.pyplot as plt
import numpy as np
# from sklearn.cluster import AgglomerativeClustering
from sknetwork.clustering import Louvain, modularity
from sknetwork.hierarchy import Paris, tree_sampling_divergence, dasgupta_score

from scipy.sparse import csr_matrix
import sys

# import mat73



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
    tmp = X.shape
    nsmp = tmp[0]
    # n = tmp[0]
    # if len(tmp)==2:
    #     nsmp = 1
    # else:
    #     nsmp = tmp[2]

    # loop through each sample
    dend = []
    TSD = []
    DGS = []
    for ismp in range(0, nsmp):
        if ismp % np.ceil(nsmp * 0.1) == 0:
            print('%d %%' % (ismp / nsmp * 100))
        # xtmp = X[:, :, ismp]
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
        # from IPython.display import SVG
        # from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph, svg_dendrogram
        paris = Paris()
        d = paris.fit_transform(C)
        # louvain_hierarchy = LouvainHierarchy()
        # d = louvain_hierarchy.fit_transform(C)

        # store
        dend.append(d)
        TSD.append(tree_sampling_divergence(C,d))
        DGS.append(dasgupta_score(C,d))

    # save
    out = {
        'dendogram':dend,
        'tree_sampling_divergence':TSD,
        'dasgupta_score':DGS
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
    if True:
        func = sys.argv[1]
        dataname = sys.argv[2]
        savepath = sys.argv[3]
    else: #debugging
        # func = 'fit_louvain'
        # dataname = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/modularity/po_obs'
        func = 'fit_paris'
        dataname = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/modularity/po_obs_hier'
        savepath = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/modularity'

    # format in and out data names
    name_in = '{}_in.mat'.format(dataname)
    name_out = '{}_out.mat'.format(dataname)

    dat_in = loadmat(name_in, squeeze_me=True)
    #dat_in = mat73.loadmat(name_in)

    # call
    print('data: {}'.format(dataname))
    if func == 'fit_louvain':
        print('fitting with Louvain...')
        fit_louvain()
    elif func == 'fit_paris':
        print('fitting with Paris...')
        fit_paris()
    elif func == 'get_metrics':
        print('getting metrics...')
        get_metrics()
