from ace import model
from scipy.io import loadmat, savemat

from ace.samples import wang04
wang04.run_wang04()

exit
datapath='/Users/Ben/Desktop/test_limb/ccn.mat'
dat_in = loadmat(datapath,squeeze_me=True)
x=dat_in['x']
y=dat_in['y']

myace = model.Model()
myace.build_model_from_xy(x, y)
myace.eval([0.1, 0.2, 0.5, 0.3, 0.5])