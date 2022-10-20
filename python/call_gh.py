
import networkx as nx
import GraphHierarchy as gh
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# load
name_in = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/modularity/po_obs_hier_in.mat'
dat_in = loadmat(name_in,squeeze_me=True)
X = dat_in['po']
X = X[1]
n = X.shape[0]

# create graph
G = nx.DiGraph()
# G.add_nodes_from([ii+1 for ii in range(0,n)])

edges = []
for ii in range(0,n):
    for jj in range(0,n):
        x = X[ii, jj]
        if x > 0:
            edges.append((ii, jj, x))
            # edges.append((ii + 1, jj + 1, 1))
G.add_weighted_edges_from(edges)

# plot
# graph = nx.gnr_graph(20, 0.4)
plt.subplot(121)
nx.draw_networkx(G)
plt.show()

foo=1