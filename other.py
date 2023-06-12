from notears.linear import notears_linear
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from notears import utils
import networkx as nx
import numpy as  np
def sEM(n,sem_type,network):
    model = get_example_model(network)
    nodes = model.nodes()
    B_true = nx.to_pandas_adjacency(model.to_directed(),nodelist=nodes,weight=None,nonedge=0)
    B_true = B_true.values#转换ndarray
    W_true = utils.simulate_parameter(B_true)
    #np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)#n节点数
    #np.savetxt('X.csv', X, delimiter=',')

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    assert utils.is_dag(W_est)
    #np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est!= 0)
    print(acc)
    result = []
    result.append(network)
    for i in acc:
        result.append(i)
    #print(acc)
    return result

sEM(500,'gauss','asia')