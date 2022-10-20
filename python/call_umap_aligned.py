
import umap
import umap.aligned_umap
from scipy.io import loadmat, savemat
import pickle
import sys
import time


# convert umap object to something matlab can read
def umap2mat(u_in):
    u_out = u_in.__dict__
    delThis = ['_input_distance_func',
               '_inverse_distance_func',
               '_output_distance_func',
               'mappers_'
               ]

    # cant convert these
    for d in delThis:
        try:
            u_out.pop(d)
        finally:
            foo=1

    for k,v in u_out.items():
        if v is None:
            u_out[k]='__None__'

    return u_out


def fit():
    print('aligned umap fit...')

    # reformat relations
    rlt=dat_in['relations']
    relations=[]
    for r in rlt:
        tmp={}
        for a in r:
            tmp[a[0]]=a[1]
        relations.append(tmp)

    foo=1

    u = umap.AlignedUMAP(verbose=True)
    dat_in.pop('relations')

    # unpack further
    X_train=dat_in['X_train']

    # set attributes that can be set
    for k,v in dat_in.items():
        if k in ['__header__','__version__','__globals__','X_train']:
            continue

        if k in u.__dict__.keys():
            v2=type(u.__dict__[k]) (dat_in[k])
            u.__setattr__(k,v2)
        else:
            print("not a umap attribute: {}".format(k))


    # call umap
    st=time.time()
    r=u.fit(X_train,relations=relations)
    print('fit time: {}'.format(time.time()-st))

    # save umap object, for python and later matlab
    print('saving to {}...'.format(savepath))

    # sname1='{}/umap_train.py'.format(savepath)
    # pickle.dump(r, open(sname1, 'wb'))
    # sname2='{}/umap_train.mat'.format(savepath)
    # savemat(sname2,umap2mat(r))


    b = []
    for x in r.embeddings_:
        b.append(x)
    sname2 = '{}/umap_train.mat'.format(savepath)
    savemat(sname2,{'embeddings_':b})

    for ii in range(len(X_train)):
        tmp = r.mappers_[ii]
        sname1='{}/umap_train{}.py'.format(savepath,ii)
        pickle.dump(tmp, open(sname1, 'wb'))

    foo=1

def transform():
    print('umap transform...')
    # load test data
    X_test=dat_in['X_test']

    loadpath=savepath

    r=pickle.load(open('{}/umap_train.py'.format(loadpath),'rb'))

    # re-embed
    embedding_test_ = r.transform(X_test)

    # save
    print('saving to {}...'.format(savepath))
    sname1='{}/umap_test.py'.format(savepath)
    pickle.dump(embedding_test_, open(sname1, 'wb'))

    sname2='{}/umap_test.mat'.format(savepath)
    savemat(sname2,{'embedding_test_': embedding_test_})

    foo=1


if __name__ == '__main__':
    # # inputs
    # func='fit'
    # name_in='/Volumes/DATA_bg/ana/ana_test_indivAlign2/umap_data_train.mat'
    # savepath = '/Volumes/DATA_bg/ana/ana_test_indivAlign2'
    func=sys.argv[1]
    name_in=sys.argv[2]
    savepath=sys.argv[3]

    # load the data
    dat_in = loadmat(name_in,squeeze_me=True)

    # calls
    if func == 'fit':
        fit()
    elif func == 'transform':
        transform()
    else:
        raise ValueError('func to call not recognized')


foo=1


