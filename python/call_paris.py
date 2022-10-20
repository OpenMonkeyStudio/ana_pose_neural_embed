import networkx as nx
from python_paris import paris
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sknetwork.clustering import Louvain
from sknetwork.data import karate_club

from scipy.sparse import csr_matrix

# laod data
name_in = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/po1.mat'

dat_in = loadmat(name_in,squeeze_me=True)
X = dat_in['po']

n = X.shape[0]

# DG = nx.DiGraph()
# DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
# plt.subplot(121)
# nx.draw(DG, with_labels=True, font_weight='bold')

# create graph
# G = nx.DiGraph()
# # G.add_nodes_from([ii+1 for ii in range(0,n)])
#
# edges = []
# for ii in range(0,n):
#     for jj in range(0,n):
#         x = X[ii, jj]
#         if x > 0:
#             edges.append((ii, jj, x))
#             # edges.append((ii + 1, jj + 1, 1))
# G.add_weighted_edges_from(edges)

# plt.subplot(121)
# nx.draw(graph, with_labels=True, font_weight='bold')
# plt.show()


row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])

row = []
col = []
data = []
for ii in range(0,n):
    for jj in range(0,n):
        x = X[ii, jj]
        if x > 0:
            row.append(ii)
            col.append(jj)
            data.append(x)
            # edges.append((ii + 1, jj + 1, 1))
C = csr_matrix((data, (row, col)), shape=(n, n))


louvain = Louvain()
# adjacency = karate_club()
labels = louvain.fit_transform(C)

foo=1

# pcc_longueurs=list(nx.all_pairs_shortest_path_length(G))
# pcc_longueurs=list(nx.all_pairs_dijkstra_path_length(G))
# pcc_longueurs=list(nx.all_pairs_bellman_ford_path_length(G))
# distances=np.zeros((n,n))
# # distances[i, j] is the length of the shortest path between i and j
# for i in range(n):
#     for j in range(n):
#         distances[i, j] = pcc_longueurs[i][1][j]
# clustering = AgglomerativeClustering(n_clusters=12,linkage='complete',affinity='precomputed').fit_predict(distances)

# out = nx.clustering(graph)
# dendrogram = paris(G)

foo=1

