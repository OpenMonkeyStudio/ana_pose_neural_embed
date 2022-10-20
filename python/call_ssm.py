from scipy.io import loadmat, savemat
# import matplotlib.pyplot as plt
import numpy as np

import sys
# from autohmm import ar

# ssmpath='/Users/Ben/Documents/git/oms_internal/bhv_cluster/toolboxes/ssm/ssm'
# from application.app.folder.file import func_name
# import ssmpath as ssm

import ssm

# load data
# format in and out data names
dataname = '/Users/Ben/Desktop/test_ssm/data'
name_in = '{}_in.mat'.format(dataname)
name_out = '{}_out.mat'.format(dataname)

dat_in = loadmat(name_in, squeeze_me=True)
# dat_in = mat73.loadmat(name_in)

c = dat_in['c2']
c=c-1
K = 2 #dat_in['nstate2']
D = dat_in['nstate2']
C = dat_in['nstate2']

# lds = ssm.LDS(K, D, emissions="gaussian")
# elbos, q = lds.fit(c, method="laplace_em", num_iters=10)

# arhmm = ssm.HMM(K=K, D=D, observations="ar")
# arhmm.fit(c)

# model = ar.ARTHMM(n_unique=K)

# x=np.transpose(np.asmatrix(c))
# y=np.ones((len(c),1))
# c2 = np.concatenate((x,y),axis=1)

arhmm = ssm.HMM(K=K, D=D, C=C, observations="categorical")
arhmm.fit(c)

zhat = arhmm.most_likely_states(c)
# tmp = arhmm.log_likelihood(c)

z=arhmm.most_likely_states(c,(len(c),))

t = arhmm.transitions.transition_matrix

foo=1