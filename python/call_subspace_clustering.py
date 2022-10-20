import numpy as np
from scipy.io import loadmat, savemat
import sys
import time
import mat73

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/mnt/scratch/git/oms_internal/bhv_cluster/matlab/python/subspace-clustering')

from cluster.selfrepresentation import ElasticNetSubspaceClustering

if __name__ == '__main__':
    name_in=sys.argv[1]
    name_out=sys.argv[2]

    st=time.time()

    # load the data
    print('loading data...')
    # dat_in = loadmat(name_in,squeeze_me=True)
    dat_in = mat73.loadmat(name_in)

    X = dat_in['X']

    #X = np.ascontiguousarray(X).astype(np.float32)
    
    # do the clustering
    clf = ElasticNetSubspaceClustering(n_clusters=3,algorithm='spams',gamma=100)
    clf.fit(X)

    print('fit time: {}'.format(time.time()-st))

    # save
    print('saving: ' + name_out)
    out = {
        'labels':clf.labels_,
        'csr':clf.representation_matrix_
    }
    savemat(name_out,out)

foo=1


