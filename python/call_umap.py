import numpy as np
import umap
import umap.aligned_umap
from scipy.io import loadmat, savemat
import pickle
import sys
import time
import numba
#import hdf5storage as hd
import mat73

def hamming(x,y):
    return np.count_nonzero(x != y)


@numba.njit()
def shd(a, b):
    n = len(a)
    D = [np.count_nonzero(a != np.roll(b,ii))/n for ii in range(0,n)]
    return min(D)

@numba.njit()
def shd_dur(a, b):
    ntot = len(a)
    n = int(ntot/2)

    s1 = a[0:n]
    dur1 = a[n:ntot]
    s2 = b[0:n]
    dur2 = b[n:ntot]

    D=[]
    for ii in range(0,n):
        # shd distance
        c = np.count_nonzero(s1 != np.roll(s2, ii)) / n
        dur = np.sqrt( np.sum((dur1 - np.roll(dur2,ii))**2) ) / n
        d = c**2 + dur**2
        D.append(d)
    return min(D)

@numba.njit()
def shd_dur_mult(a, b):
    ntot = len(a)
    n = int(ntot/2)

    s1 = a[0:n]
    dur1 = a[n:ntot]
    s2 = b[0:n]
    dur2 = b[n:ntot]

    D=[]
    for ii in range(0,n):
        # shd distance
        c = s1 == np.roll(s2, ii)
        dur = (dur1 - np.roll(dur2,ii))
        #d = c**2 + dur**2
        d = np.sqrt(np.sum( (c * dur)**2 )) / n + np.count_nonzero(not c)
        D.append(d)
    return min(D)

@numba.njit()
def shd_dur_sim2(a, b):
    ntot = len(a)
    n = int(ntot / 4)

    s1 = a[0:n]
    dur1 = a[n:n * 2]
    s2 = b[0:n]
    dur2 = b[n:n * 2]
    xy1 = a[n * 2:ntot]
    xy2 = b[n * 2:ntot]

    D = []
    for ii in range(0, n):
        c = np.count_nonzero(s1 != np.roll(s2, ii)) / n # hamming
        dur = np.sqrt(np.sum((dur1 - np.roll(dur2, ii)) ** 2)) / n # duration
        sim = np.sqrt(np.sum((xy1 - np.roll(xy2, ii*2)) ** 2)) / n # 2D similarity
        d = c ** 2 + dur ** 2 + sim ** 2
        D.append(d)
    return min(D)

@numba.njit()
def sdur_sim(a, b):
    ntot = len(a)
    n = int(ntot / 3)

    D = []
    for ii in range(0, n):
        d = np.sqrt(np.sum((a - np.roll(b, ii*3)) ** 2)) / n
        D.append(d)
    return min(D)

@numba.njit()
def ssim(a, b):
    ntot = len(a)
    n = int(ntot / 2)

    D = []
    for ii in range(0, n):
        d = np.sqrt(np.sum((a - np.roll(b, ii*2)) ** 2)) / n
        D.append(d)
    return min(D)

# convert umap object to something matlab can read
def umap2mat(u_in):
    u_out = u_in.__dict__
    # 'graph_'
    delThis = ['_input_distance_func',
               '_inverse_distance_func',
               '_output_distance_func',
                'graph_',
                'precomputed_knn'
               ]

    # metric
    m = u_in.metric
    if not isinstance(m,str):
        u_out['metric'] = m.__name__

    # cant convert these
    for d in delThis:
        try:
            u_out.pop(d)
        finally:
            foo=1

    #print(u_out['n_jobs'])
    for k,v in u_out.items():
        #print('{}: {}'.format(k,type(v)))
        if v is None:
            #print('changing...')
            u_out[k]='__None__'
        elif type(v) is dict:
            u_out[k]='__dict__'
            
    #for k,v in u_out.items():
    #    print('{}: {}'.format(k,type(v)))

    return u_out


def fit():
    print('umap fit...')

    #u = umap.UMAP(verbose=True,parallel=True)
    u = umap.umap_.UMAP(verbose=True)

    # unpack further
    X_train=dat_in['X_train']

    if 'y' in dat_in.items():
        doSupervised = True
        targ = dat_in['target']
    else:
        doSupervised = False

    # set attributes that can be set
    for k,v in dat_in.items():
        if k in ['__header__','__version__','__globals__','X_train']:
            continue

        if 'custom_metric' in k: # use a custom metric
            if v == 'shiftedHamming':
                dfunc = shd
            elif v == 'shiftedHammingDur':
                dfunc = shd_dur
            elif v == 'shiftedHammingDurSim2':
                dfunc = shd_dur_sim2
            elif v == 'shiftedDurSim':
                dfunc = sdur_sim
            elif v == 'shiftedSim':
                dfunc = ssim
            u.__setattr__('metric', dfunc)
        elif k == 'n_epochs':
            u.__setattr__('n_epochs', int (dat_in[k]))
        elif k in u.__dict__.keys(): # generic variable
            print('setting {}'.format(k))
            try:
                v2=type(u.__dict__[k]) (dat_in[k])
            except:
                v2=dat_in[k]
            u.__setattr__(k,v2)

        else:
            print("not a umap attribute: {}".format(k))


    # call umap
    st=time.time()
    if doSupervised:
        reducer=u.fit(X_train,y=targ)
    else:
        reducer=u.fit(X_train)

    print('fit time: {}'.format(time.time()-st))

    # save umap object, for python and later matlab
    print('saving to {}...'.format(savepath))

    sname2='{}/umap_train.mat'.format(savepath)
    savemat(sname2,umap2mat(reducer))

    # dont save python for now, takes too long
    #sname1='{}/umap_train.py'.format(savepath)
    #pickle.dump(reducer, open(sname1, 'wb'), protocol=4)


    return reducer
    foo=1

def transform(reducer):
    print('umap transform...')
    # load test data
    loadpath=savepath

    X_test=dat_in['X_test']

    # load back reducer if necessary
    if reducer is None:
        reducer=pickle.load(open('{}/umap_train.py'.format(loadpath),'rb'))

    # re-embed
    embedding_test_ = reducer.transform(X_test)

    # save
    print('saving to {}...'.format(savepath))
    # sname1='{}/umap_test.py'.format(savepath)
    # pickle.dump(embedding_test_, open(sname1, 'wb'))

    sname2='{}/umap_test.mat'.format(savepath)
    savemat(sname2,{'embedding_test_': embedding_test_})

    foo=1


if __name__ == '__main__':
    # # inputs
    # func='fit'
    # name_in='/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/embed_seq_shiftDurSim/umap_data_train.mat'
    # savepath = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/embed_seq_shiftDurSim'
    func=sys.argv[1]
    name_in=sys.argv[2]
    savepath=sys.argv[3]

    # load the data
    # dat_in = loadmat(name_in,squeeze_me=True)
    #dat_in = hd.loadmat(name_in)
    dat_in = mat73.loadmat(name_in)

    #if 'nparallel' in dat_in.keys():
    #    if dat_in['nparallel'] > 1:
    #        dat_in['parallel']=True
    #        numba.set_num_threads(dat_in['parallel'])
    #else:
    #    dat_in['parallel']=False

    # calls
    if func == 'fit':
        fit()
    elif func == 'transform':
        transform(None)
    elif func == 'fit_transform':
        reducer = fit()
        transform(reducer)
    else:
        raise ValueError('func to call not recognized')



foo=1


