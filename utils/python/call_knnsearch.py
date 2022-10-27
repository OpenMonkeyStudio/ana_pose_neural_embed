import numpy as np
from scipy.io import loadmat, savemat
import sys
import time
import faiss
import mat73

if __name__ == '__main__':
    name_in=sys.argv[1]
    name_out=sys.argv[2]

    st=time.time()

    # load the data
    print('loading data...')
    # dat_in = loadmat(name_in,squeeze_me=True)
    dat_in = mat73.loadmat(name_in)

    X = dat_in['X']
    Xq = dat_in['Xq']
    K = int(dat_in['K'])
    metric = dat_in['metric']
    useGPU = dat_in['useGPU']
    
    X = np.ascontiguousarray(X).astype(np.float32)

    if not isinstance(Xq,list):
        Xq = np.ascontiguousarray(Xq).astype(np.float32)


    # do the search
    print('setup...')

    d = X.shape[1]
    if metric=='cosine':
        print('metric: {}'.format(metric))
        faiss.normalize_L2(X)
        index1 = faiss.index_factory(d, "Flat",faiss.METRIC_INNER_PRODUCT)
    else:
        index1 = faiss.index_factory(d, "Flat")

    # CPU or gpu index?
    if useGPU > 0:
        if useGPU==1:
            print('single GPU...')
            res = faiss.StandardGpuResources()  # use a single GPU
            index = faiss.index_cpu_to_gpu(res, 0, index1)
        else:
            print('all GPU...')
            index = faiss.index_cpu_to_all_gpus(index1)
    else:
        print('CPU...')
        index = index1
        index.parallel_mode = 1


    print('training')
    index.train(X)
    index.add(X)

    print('searching')
    if isinstance(Xq,list):
        distances = []
        neighbors = []
        ii=-1
        for s in Xq:
            print(s)

            ii=ii+1
            print('batch {}: {}'.format(ii,s))
            #xin = mat73.loadmat(s)
            xin = loadmat(s)
            print(xin.keys())

            if "Xq" in xin:
                xq = xin['Xq']
            elif "X_feat" in xin:
                xq = xin['X_feat']
            else:
                print("not sure what data to look for: {}".format(s))
                exit

            xq = np.ascontiguousarray(xq).astype(np.float32)
            if metric=='cosine':
                faiss.normalize_L2(xq)
            dd, nn = index.search(xq, K)
            distances.append(dd)
            neighbors.append(nn)
    else:
        distances, neighbors = index.search(Xq, K)

    print('fit time: {}'.format(time.time()-st))

    # save
    print('saving: ' + name_out)
    out = {
        'distances':distances,
        'neighbors':neighbors,
    }
    savemat(name_out,out)

foo=1


