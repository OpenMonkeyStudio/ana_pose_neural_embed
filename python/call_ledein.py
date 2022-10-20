from scipy.io import loadmat, savemat
import leidenalg as la
import igraph as ig
import cairocffi

name_in = '/Users/ben/Desktop/test_ledein/PO.mat'
name_out = '/Users/ben/Desktop/test_ledein/PO_out.mat'

dat_in = loadmat(name_in, squeeze_me=True)

PO = dat_in['PO']
nstate=PO[0].shape[0]

Gs = []
for po in PO:
    g = ig.Graph.Adjacency((po > 0).tolist())

    # Add edge weights and node labels.
    g.es['weight'] = po[po.nonzero()]
    g.vs['id'] = ['{}'.format(x+1) for x in range(0,nstate)]  # or a.index/a.columns
    g.vs['label'] = ['{}'.format(x+1) for x in range(0,nstate)]  # or a.index/a.columns

    Gs.append(g)

# membership, improv = la.find_partition_multiplex(
#                        Gs,
#                        la.ModularityVertexPartition);
#
# membership, improv = la.find_partition_temporal(
#                        Gs,
#                        la.ModularityVertexPartition);

membership, improvement = la.find_partition_temporal(
                            Gs,
                            la.ModularityVertexPartition,
                            interslice_weight=0.01
                            )

out={'labels':membership}
savemat(name_out, out)

res = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]
tmp=[]
for r in res:
    membership, improvement = la.find_partition_temporal(
                                Gs,
                                la.CPMVertexPartition,
                                interslice_weight=0.1,
                                resolution_parameter=r)
    tmp.append(improvement)

membership, improvement = la.find_partition_temporal(
                            Gs,
                            la.CPMVertexPartition,
                            interslice_weight=0.1,
                            resolution_parameter=0.001)
out={'labels':membership}
savemat(name_out, out)

foo=1
