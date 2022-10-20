import sys
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from scipy.io import loadmat, savemat

import warnings
warnings.filterwarnings("ignore")

# where the data lives
# name_in = sys.argv[1]
# name_out = sys.argv[2]
# name_in = '/users/ben/desktop/test.mat'
# name_out = '/users/ben/desktop/test_new.mat'
name_in='/private/var/folders/dc/6xmmkjg1163gsfv0l93lncgh0000gn/T/tp3fd4d036_8438_439c_9a47_d85f93bf9ae5_in.mat'
name_out='/private/var/folders/dc/6xmmkjg1163gsfv0l93lncgh0000gn/T/tp3fd4d036_8438_439c_9a47_d85f93bf9ae5_out.mat'

def main():
    # load the data
    dat = loadmat(name_in)

    X = dat['X']
    y = np.ravel(dat['y'])
    n_jobs=dat['njobs'][0][0]
    nrand=dat['nrand'][0][0]

    penalty='elasticnet'
    max_iter = 2000
    multi_class = 'ovr'
    solver='saga'
    Cs=[0.01, 0.1, 1.0, 10.0, 100.0]
    l1_ratios = np.arange(0,1.1,0.1)
    if n_jobs<2:
        n_jobs=None


    # run code
    print('fitting observations')
    clf=LogisticRegressionCV(cv=5,
                             n_jobs=n_jobs,
                             max_iter=max_iter,
                             penalty=penalty,
                             multi_class=multi_class,
                             solver=solver,
                             l1_ratios=l1_ratios,
                             Cs=Cs
                             ).fit(X,y)

    a=[]
    for k in clf.scores_:
        a.append(clf.scores_[k])


    # randomization
    print('fitting randomizations')
    ar=[]
    clfr=[]
    for ir in range(nrand):
        print("rand ", ir)
        yr = sklearn.utils.shuffle(y)
        c=LogisticRegressionCV(cv=5,
                                 n_jobs=n_jobs,
                                 max_iter=max_iter,
                                 penalty=penalty,
                                 multi_class=multi_class,
                                 solver=solver,
                                 l1_ratios=l1_ratios,
                                 Cs=Cs
                                 ).fit(X,yr)
        tmpar=[]
        for k in c.scores_:
            tmpar.append(c.scores_[k])

        ar.append(tmpar)
        clfr.append(c)

    # feature importance

    # final
    br = []
    for ii in range(nrand):
        br.append(clfr[ii].coef_)

    out={
        'acc':a,
        'acc_rand':ar,
        'b':clf.coef_,
        'b_rand':br,
        'C':clf.Cs_,
        'l1_ratios':clf.l1_ratios_
        # 'clf':clf
    }

    # save
    savemat(name_out,out)


if __name__ == '__main__':
    main()

foo=1
