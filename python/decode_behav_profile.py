import sys
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from scipy.io import loadmat, savemat
import sklearn.metrics as smet
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")

# where the data lives
name_in = sys.argv[1]
name_out = sys.argv[2]
# name_in = '/private/var/folders/dc/6xmmkjg1163gsfv0l93lncgh0000gn/T/tp510da2e6_fdc7_414d_a697_48d268d8968c_in.mat'
# name_out = '/private/var/folders/dc/6xmmkjg1163gsfv0l93lncgh0000gn/T/tp510da2e6_fdc7_414d_a697_48d268d8968c_out.mat'

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


    max_iter = 2000
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
    l1_ratios = np.arange(0, 1.1, 0.1)

    if multi_class=='ovr':
        penalty='elasticnet'
        #multi_class = 'ovr'
        solver='saga'
    elif multi_class=='multinomial':
        penalty='elasticnet'
        #multi_class='multinomial'
        solver='saga'
    else:
        raise ValueError('multi_class not recognized')

    if n_jobs<2:
        n_jobs=None


    # run code
    print('fitting observations')
    clf=LogisticRegressionCV(cv=nfold,
                             n_jobs=n_jobs,
                             max_iter=max_iter,
                             penalty=penalty,
                             multi_class=multi_class,
                             solver=solver,
                             l1_ratios=l1_ratios,
                             Cs=Cs
                             ).fit(X_train,y_train)

    acc,cm,f1=get_metrics(clf, X_test, y_test)
    b=clf.coef_

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
    for ir in range(nrand):
        print("rand ", ir)
        yr = sklearn.utils.shuffle(y_train)
        c=LogisticRegressionCV(cv=nfold,
                                 n_jobs=n_jobs,
                                 max_iter=max_iter,
                                 penalty=penalty,
                                 multi_class=multi_class,
                                 solver=solver,
                                 l1_ratios=l1_ratios,
                                 Cs=Cs
                                 ).fit(X_train,yr)

        accr, cmr, f1r = get_metrics(c, X_test, y_test)

        acc_rand.append(accr)
        cm_rand.append(cmr)
        f1_rand.append(f1r)
        b_rand.append(c.coef_)

        f = permutation_importance(c, X_test, y_test,
                                          n_repeats=nrand,
                                          random_state=0)
        feat_rand.append(f)



    # final
    br = []


    out={
        'acc':acc,
        'acc_rand':acc_rand,
        'confusion':cm,
        'confusion_rand':cm_rand,
        'f1':f1,
        'f1_rand':f1_rand,
        'b':b,
        'b_rand':b_rand,
        'C':clf.Cs_,
        'l1_ratios':clf.l1_ratios_,
        'feature_importance':feat_imp,
        'feature_importance_rand': feat_rand,

        # 'clf':clf
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

    #fpr, tpr, thresholds = smet.roc_curve(y_test, probas_[:, ilabel])
    #prec, rec, th2 = smet.precision_recall_curve(y_test, probas_[:, ilabel])

    #acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    #f1 = smet.f1_score(y_test, y_pred)
    #roc_auc = smet.auc(fpr, tpr)
    #pr_auc = smet.average_precision_score(y_test, y_pred)

if __name__ == '__main__':
    main()

foo=1
