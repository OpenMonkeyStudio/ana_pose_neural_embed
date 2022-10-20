from scipy.io import loadmat, savemat
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import sklearn as sk
from sklearn.utils.class_weight import compute_sample_weight

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
import mat73

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
    # init
    clf = LogisticRegression(
                solver='liblinear',
                class_weight='balanced'
                )

    # loop over models
    out={}
    for ii in range(0,2):
        if ii == 0:
            X2 = np.ndarray.copy(Xin)[:,1:Xin.shape[0]] 
        else:
            X2 = np.ndarray.copy(Xin) 

        # train
        clf.fit(X2, yin)

        # stuff
        nobs=X2.shape[0];
        nfeat=X2.shape[1]
        yd = sk.preprocessing.OneHotEncoder().fit_transform(y.reshape(-1,1)).toarray()
        y_pred = clf.predict_proba(X2)
        w=sample_weight=compute_sample_weight("balanced", yin)

        ll=-2*sk.metrics.log_loss(y, y_pred, normalize=False) # log-likelihood    
        a,b=AIC_BIC(ll,nobs,nfeat)
        #print('aic={},bic={}'.format(a,b))
        
        r2 = sk.metrics.r2_score(yd,y_pred,sample_weight=w) #multioutput='variance_weighted'
        mse = sk.metrics.mean_squared_error(yd,y_pred,sample_weight=w)
        evar = sk.metrics.explained_variance_score(yd,y_pred,sample_weight=w)

        # output
        out['coef{}'.format(ii)]= clf.coef_
        out['aic{}'.format(ii)]= a
        out['bic{}'.format(ii)]= b
        out['r2{}'.format(ii)]= r2
        out['mse{}'.format(ii)]= mse
        out['expvar{}'.format(ii)]= evar

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
    #dat_in = loadmat(name_in, squeeze_me=True)
    dat_in = mat73.loadmat(name_in)
    X = dat_in['X']
    y = dat_in['y']
    if dat_in.__contains__('dim_rate'):
        dim_rate = dat_in['dim_rate']
        print(type(dim_rate))
        print(dim_rate)
        dim_rate = dim_rate.astype(int)
        print(dim_rate)

        if dim_rate.size==1:
            dim_rate = [dim_rate]
    else:
        dim_rate=[0]

    if dat_in.__contains__('nrand'):
        nrand = int(dat_in['nrand'])
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
        nsmp = len(yr)
        mn=np.ceil(nsmp*0.25)
        mx=np.max(nsmp*0.75)
        for ir in range(0,nrand):
            print('{}'.format(ir))
            #yr = np.random.permutation(yr)
            
            for ix in dim_rate:
                r=np.random.randint(mn,mx)
                idx = np.roll(np.arange(nsmp),r)
                Xr[:,ix] = X[idx,ix]
            #yr=np.roll(yr,r)
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
    res['models'] = np.array(['pose','pose+rate'], dtype=np.object)
    res['Xdim'] = X.shape
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
