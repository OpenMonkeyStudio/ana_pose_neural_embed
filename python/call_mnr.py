from scipy.io import loadmat, savemat
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import sklearn as sk

#from sklearn.model_selection import train_test_split, StratifiedKFold
#from sklearn import svm
#from sklearn import metrics as skm
#from sklearn.inspection import permutation_importance
#from sklearn.utils import class_weight
#from sklearn.utils.class_weight import compute_sample_weight

#import xgboost as xgb
#import lightgbm as lgbm
#import /home/auser/miniconda3/envs/bv/lib/python3.7/site-packages/xgboost as xgb
#import xgboost; print(xgboost.__version__)
#from xgboost import XGBClassifier
#print(xgb.__file__)


import time
import sys

import warnings
#warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings(action="ignore", message=r".*\nStarting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.")
 
def LogLikelihood(residual,n,k):
     ll = -(n * 1/2) * (1 + np.log(2 * np.pi)) - (n / 2) * np.log(residuals.dot(residuals) / n)

     return ll

  
def AIC_BIC(ll,n,k):
    k = k + 1

    AIC = (-2 * ll) + (2 * k)
    BIC = (-2 * ll) + (k * np.log(n))

    return AIC, BIC


# main fit function
def run_clf(Xin,yin):

    #w=sample_weight=compute_sample_weight("balanced", yin)
    #clf = LogisticRegression(
    #            multi_class='multinomial', 
    #            solver='lbfgs',
    #            class_weight='balanced'
    #            )

    clf = LogisticRegression(
                solver='liblinear',
                class_weight='balanced'
                )

    # train
    clf.fit(Xin, yin)

    # stuff
    ll=-2*sk.metrics.log_loss(y, clf.predict_proba(Xin), normalize=False) # log-likelihood    
    nobs=Xin.shape[0];
    nfeat=Xin.shape[1]
    a,b=AIC_BIC(ll,nobs,nfeat)
    #print('aic={},bic={}'.format(a,b))

    # output
    out = {
        'coef': clf.coef_,
        'aic':a,
        'bic':b
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

    # load data
    dat_in = loadmat(name_in, squeeze_me=True)
    X = dat_in['X']
    y = dat_in['y']
    if dat_in.__contains__('nrand'):
        nrand = dat_in['nrand']
    else:
        nrand = 0

    # run observed
    print('observed fit...')
    Xo=np.ndarray.copy(X)
    yo=np.ndarray.copy(y)
    res_obs = run_clf(X,y)
    print("elapsed time: {} sec",time.time()-start_time)

    # run random
    res_tmp = []
    if (nrand > 0):
        print('random fit..')
        Xr= np.ndarray.copy(X)
        yr = np.ndarray.copy(y)
        for ir in range(0,nrand):
            print('{}'.format(ir))
            yr = np.random.permutation(yr)

            res = run_clf(Xr,yr)
            res_tmp.append(res)

        res_rand = {k: [] for k in res_tmp[0].keys()}

        for ii in range(0,nrand):
            for k in res_rand.keys():
                res_rand[k].append(res_tmp[ii][k])
    print("elapsed time: {} sec",time.time()-start_time)

    foo=1

    # prep output
    res = {}
    for k in res_obs.keys():
        k2 = k+'_obs'
        res[k2] = res_obs[k]

    if (nrand > 0):
        for k in res_rand.keys():
            k2 = k + '_rand'
            res[k2] = res_rand[k]

    # finish
    end_time = time.time()
    print("elapsed time: {} sec",end_time-start_time)


    # save
    savemat(name_out, res)

foo=1
