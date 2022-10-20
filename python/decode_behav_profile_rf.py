import sys
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from scipy.io import loadmat, savemat
import sklearn.metrics as smet
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import time

import warnings
warnings.filterwarnings("ignore")

# where the data lives
name_in = sys.argv[1]
name_out = sys.argv[2]
# name_in=    '/private/var/folders/dc/6xmmkjg1163gsfv0l93lncgh0000gn/T/tp722a5415_668b_4c50_b34f_768aeab1648e_in.mat'
# name_out=    '/private/var/folders/dc/6xmmkjg1163gsfv0l93lncgh0000gn/T/tp722a5415_668b_4c50_b34f_768aeab1648e_out.mat'

# name_in='/private/var/folders/dc/6xmmkjg1163gsfv0l93lncgh0000gn/T/tp6759f7bc_5bc8_4bc4_b9bd_c6eee5e0388a_in.mat'
# name_out='/private/var/folders/dc/6xmmkjg1163gsfv0l93lncgh0000gn/T/tp6759f7bc_5bc8_4bc4_b9bd_c6eee5e0388a_out.mat'

def main():
    # load the data
    dat = loadmat(name_in)

    X_train = dat['X_train']
    X_test = dat['X_test']
    y_train = np.ravel(dat['y_train'])
    y_test = np.ravel(dat['y_test'])
    multi_class = dat['multi_class']

    nfold = dat['nfolds'][0][0]
    n_jobs=dat['njobs'][0][0]
    nrand=dat['nrand'][0][0]

    if n_jobs<2:
        n_jobs=None

    # grid for tuning
    # n_estimators = [int(x) for x in np.linspace(start=10, stop=210, num=5)]
    n_estimators = [2,3,6,9]
    max_features = [None, 'sqrt']
    max_depth = [int(x) for x in np.linspace(2, 9, num=6)]
    max_depth.append(None)
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'bootstrap': bootstrap}

    # train observed
    st=time.time()
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=400, cv=nfold, verbose=1,n_jobs=-1)
    rf_random.fit(X_train, y_train)
    clf = rf_random.best_estimator_
    acc,cm,f1=get_metrics(clf, X_test, y_test)

    print("time: {}".format(time.time()-st))
    b=clf.feature_importances_
    best_params=rf_random.best_params_

    # feature importance
    print('feature importance')
    feat_imp = permutation_importance(clf, X_test, y_test,
                               n_repeats = nrand,
                               random_state = 0)
    # randomization
    print('fitting randomizations')

    acc_rand = []
    cm_rand = []
    f1_rand = []
    b_rand = []
    feat_rand=[]
    # best_params_rand=[]
    for ir in range(nrand):
        print("rand ", ir)
        yr = sklearn.utils.shuffle(y_train)

        rfr = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=nfold, verbose=0,n_jobs=-1)
        rfr.fit(X_train, yr)
        c = rfr.best_estimator_

        accr, cmr, f1r = get_metrics(c, X_test, y_test)

        acc_rand.append(accr)
        cm_rand.append(cmr)
        f1_rand.append(f1r)
        b_rand.append(c.feature_importances_)
        # best_params_rand.append(rfr.best_params_)

        f = permutation_importance(c, X_test, y_test, n_repeats=nrand)
        feat_rand.append(f)

    # final
    out={
        'acc':acc,
        'acc_rand':acc_rand,
        'confusion':cm,
        'confusion_rand':cm_rand,
        'f1':f1,
        'f1_rand':f1_rand,
        'b':b,
        'b_rand':b_rand,
        'feature_importance':feat_imp,
        'feature_importance_rand': feat_rand,
        'best_params': best_params,
        # 'best_params_rand': best_params_rand
    }

    # save
    savemat(name_out,out)


def get_metrics(clf,x_test,y_test):
    probas_ = clf.predict_proba(x_test)
    y_pred = probas_.argmax(axis=1)+1

    acc=smet.accuracy_score(y_test,y_pred)
    cm = smet.confusion_matrix(y_test,y_pred)
    f1=smet.f1_score(y_test,y_pred,average='weighted')


    return acc,cm,f1

if __name__ == '__main__':
    main()

foo=1
