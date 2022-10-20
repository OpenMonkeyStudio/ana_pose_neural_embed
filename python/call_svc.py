from scipy.io import loadmat, savemat
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
from sklearn import metrics as skm
from sklearn.inspection import permutation_importance

import time
import sys
import mat73

# main fit function
def run_clf(Xin,yin):
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=nfold)
    clf = svm.SVC(kernel=kernel, C=1, class_weight='balanced')

    F1=[]
    ACC=[]
    COEF=[]
    #P=[]
    for i, (train, test) in enumerate(cv.split(Xin, yin)):
        #print('fold {}'.format(i))

        # train
        clf.fit(Xin[train,:], yin[train])

        # test
        y_pred = clf.predict(Xin[test,:])

        # metrics
        f1 = skm.f1_score(yin[test],y_pred,average='weighted')
        acc = skm.balanced_accuracy_score(yin[test],y_pred)
        if kernel == 'linear':
            coef = clf.coef_
        else:
            coef = np.empty((1,Xin.shape[1],))
            coef.fill(np.nan)
        #r = permutation_importance(clf, Xin[test,:], yin[test],n_repeats = 30)

        # store
        F1.append(f1)
        ACC.append(acc)
        COEF.append(coef)
        #P.append(r)

    # output
    out = {
        'accuracy': ACC,
        'F1': F1,
        'coef': COEF
    }
    return out
    foo=1


if __name__ == '__main__':
    # inputs
    if True:
        name_in = sys.argv[1]
        name_out = sys.argv[2]
    else: #debugging
        foo=1;

    start_time = time.time()

    # laod data
    #dat_in = loadmat(name_in, squeeze_me=True)
    dat_in = mat73.loadmat(name_in);
    X = dat_in['X']
    y = dat_in['y']
    nfold = int(dat_in['nfold'])
    if dat_in.__contains__('nrand'):
        nrand = int(dat_in['nrand'])
    else:
        nrand = 0
    if dat_in.__contains__('kernel'):
        kernel = dat_in['kernel']
    else:
        kernel = 'linear'

    # run observed
    print('observed fit...')
    Xo=np.ndarray.copy(X)
    yo=np.ndarray.copy(y)
    print('obersved fit..')
    res_obs = run_clf(Xo,yo)

    # run random
    res_tmp = []
    if (nrand > 0):
        print('random fit..')
        for ir in range(0,nrand):
            print('{}'.format(ir))
            Xr= np.ndarray.copy(X)
            yr = np.ndarray.copy(y)
            yr = np.random.permutation(yr)

            res = run_clf(Xr,yr)
            res_tmp.append(res)

        res_rand = {k: [] for k in res_tmp[0].keys()}

        for ii in range(0,nrand):
            for k in res_rand.keys():
                res_rand[k].append(res_tmp[ii][k])

    foo=1


    # finish
    end_time = time.time()
    print("elapsed time: {} sec",end_time-start_time)

    res = {}
    for k in res_obs.keys():
        k2 = k+'_obs'
        res[k2] = res_obs[k]

    if (nrand > 0):
        for k in res_rand.keys():
            k2 = k + '_rand'
            res[k2] = res_rand[k]

    # save
    savemat(name_out, res)

foo=1

def get_metrics(y,y_pred):
    foo=1