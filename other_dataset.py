import numpy as np
import pandas as pd

from notears.linear import notears_linear
from pgmpy.utils import get_example_model

from notears import utils
import networkx as nx
from curriculum import anticurr, curr

utils.set_random_seed(1)
def CL_Notears(n,sem_type,network):
    #n,sem_type = 1500,'gauss'
    model = get_example_model(network)
    nodes = model.nodes()
    B_true = nx.to_pandas_adjacency(model.to_directed(),nodelist=nodes,weight=None,nonedge=0)
    cu = []
    truea = []
    curriculum = curr(network)
    for i in range(0,len(curriculum)):
        a = B_true.loc[curriculum[i],curriculum[i]]
        a = a.values
        truea.append(a)
    B1_true = B_true.values#转换ndarray
    W_true = utils.simulate_parameter(B1_true)
    #np.savetxt('W_true.csv', W_true, delimiter=',')
    X = utils.simulate_linear_sem(W_true, n, sem_type)#n节点数
    sample_X = pd.DataFrame(X)
    sample_X.columns = nodes
    est = []
    aloss = []
    aacc = []
    for i in range(0,len(curriculum)):
        sample = sample_X.loc[:,curriculum[i]]
        sample = sample.values
    # np.savetxt('X.csv', X, delimiter=',')
        W_est, loss = notears_linear(sample, lambda1=0.1, loss_type='l2')
        estw = pd.DataFrame(W_est)
        estw.columns = curriculum[i]
        estw.index = curriculum[i]
        est.append(estw)
        aloss.append(loss)
        print(loss)
        acc = utils.count_accuracy(truea[i] , W_est != 0)
        aacc.append(acc)
        if acc == 'B_est should be a DAG':
            break
        print(acc)
    if 'B_est should be a DAG' in aacc:
        return None
    edge_est = {}
    zhi = {}
    import numpy as np
    l = []
    for i in range(int(len(aloss)/2),len(aloss)):
        if int(aloss[i]) == int(np.min(aloss[int(len(aloss)/2):])):
            l.append(i)
    if int(aloss[len(aloss)-1]) <= int(aloss[len(aloss)-2]):
        l.append(i)
    best = l[-1]
    #找到最好的中间阶段
    for i in curriculum[best]:
        for j in curriculum[best]:
            if est[best].at[i, j] != 0:
                if (i, j) in edge_est.keys():
                    edge_est[(i, j)] += 0.2
                else:
                    edge_est[(i, j)] = 0.2
                zhi[(i, j)] = est[best].at[i, j]
    if best != len(est)-1:
        for a in range(best + 1, len(est)): #各个阶段修改的
            for i in curriculum[a]:
                for j in curriculum[a]:
                    if est[a].at[i, j] != 0 and ((i not in curriculum[a-1]) or (j not in curriculum[a-1])):
                        if (i, j) in edge_est.keys():
                            edge_est[(i, j)] += 0.1
                        else:
                            edge_est[(i, j)] = 0.1
                        zhi[(i, j)] = est[a].at[i, j]
    for i in curriculum[len(est)-1]:
        for j in curriculum[len(est)-1]:
            if est[len(est)-1].at[i, j] != 0:
                if (i, j) in edge_est.keys():
                    edge_est[(i, j)] += 0.1
                else:
                    edge_est[(i, j)] = 0.1
                zhi[(i, j)] = est[len(est)-1].at[i, j]
    edges = {}
    dele = []
    """
    for i in curriculum[len(est) - 1]:
        for j in curriculum[len(est) - 1]:
            if est[len(est) - 1].at[i, j] == 0.1:
                dele.append((i,j))"""
    """ sample = sample_X.loc[:, cordata]
    sample = sample.values
    # np.savetxt('X.csv', X, delimiter=',')
    W_est, loss = notears_linear(sample, lambda1=0.1, loss_type='l2')
    estc = pd.DataFrame(W_est)
    estc.columns = cordata
    estc.index = cordata
    cc = B_true.loc[cordata, cordata]
    cc = cc.values
    acc = utils.count_accuracy(cc, W_est != 0)
    for i in cordata:
        for j in cordata:
            if estc.at[i, j] != 0:
                if (i, j) in edge_est.keys():
                    edge_est[(i, j)] += 0.1
                else:
                    edge_est[(i, j)] = 0.1
                    zhi[(i, j)] = estc.at[i, j]"""
    for k in edge_est.keys() :#duplicate
        if (k[1],k[0]) in edge_est.keys():
            if edge_est[k] >= edge_est[(k[1],k[0])]:
                if (k[1],k[0]) not in dele:
                    dele.append((k[1],k[0]))
            else:
                if k not in dele:
                    dele.append(k)
        if edge_est[k] <=0.1*int(len(est)-1-best)/2:
            if k not in dele:
                dele.append(k)
    for i in dele:
        edge_est.pop(i)
    for k, v in edge_est.items():
        edges[k] = zhi[k]
    est_w = pd.DataFrame(0.00,columns=nodes,index=nodes)
    for k,v in edges.items():
        est_w.loc[k[0],k[1]] = edges[k]
    est_w= est_w.values
    acc = utils.count_accuracy(B1_true, est_w != 0)
    print(acc)
    result = []
    result.append(network)
    for i in acc:
        result.append(i)
    for j in aacc[-1]:
        result.append(j)
    # print(acc)
    return result
def anCL_Notears(n,sem_type,network):
    #n,sem_type = 1500,'gauss'
    model = get_example_model(network)
    nodes = model.nodes()
    B_true = nx.to_pandas_adjacency(model.to_directed(),nodelist=nodes,weight=None,nonedge=0)
    cu = []
    truea = []
    curriculum = anticurr(network)
    for i in range(0,len(curriculum)):
        a = B_true.loc[curriculum[i],curriculum[i]]
        a = a.values
        truea.append(a)
    B_true = B_true.values#转换ndarray
    W_true = utils.simulate_parameter(B_true)
    #np.savetxt('W_true.csv', W_true, delimiter=',')
    X = utils.simulate_linear_sem(W_true, n, sem_type)#n节点数
    sample_X = pd.DataFrame(X)
    sample_X.columns = nodes
    est = []
    aloss = []
    aacc = []
    for i in range(0,len(curriculum)):
        sample = sample_X.loc[:,curriculum[i]]
        sample = sample.values
    # np.savetxt('X.csv', X, delimiter=',')
        W_est, loss = notears_linear(sample, lambda1=0.1, loss_type='l2')
        estw = pd.DataFrame(W_est)
        estw.columns = curriculum[i]
        estw.index = curriculum[i]
        est.append(estw)
        aloss.append(loss)
        print(loss)
        acc = utils.count_accuracy(truea[i] , W_est != 0)
        aacc.append(acc)
        if acc == 'B_est should be a DAG':
            break
        print(acc)
    if 'B_est should be a DAG' in aacc:
        return None
    edge_est = {}
    zhi = {}
    import numpy as np
    l = []
    for i in range(int(len(aloss)/2+1),len(aloss)):
        if int(aloss[i]) == int(np.min(aloss[int(len(aloss)/2+1):])):
            l.append(i)
        """if int(aloss[i]) <= int(aloss[i-1]):
            l.append(i)"""
    k = l[-1]
    #找到最好的中间阶段
    for i in curriculum[k]:
        for j in curriculum[k]:
            if est[k].at[i, j] != 0:
                if (i, j) in edge_est.keys():
                    edge_est[(i, j)] += 0.1
                else:
                    edge_est[(i, j)] = 0.1
                zhi[(i, j)] = est[k].at[i, j]
    """if k != len(est)-1:
        for a in range(k + 1, len(est)-1): #各个阶段修改的
            for i in curriculum[a]:
                for j in curriculum[a]:
                    if est[a].at[i, j] != 0 and ((i not in curriculum[a-1]) or (j not in curriculum[a-1])):
                        if (i, j) in edge_est.keys():
                            edge_est[(i, j)] += 0.1
                        else:
                            edge_est[(i, j)] = 0.1
                        zhi[(i, j)] = est[a].at[i, j]"""
    #最后一次
    for i in curriculum[-1]:
        for j in curriculum[-1]:
            if est[len(est)-1].at[i, j] != 0:
                if (i, j) in edge_est.keys():
                    edge_est[(i, j)] += 0.1
                else:
                    edge_est[(i, j)] = 0.1
                zhi[(i, j)] = est[k].at[i, j]
    cordata=[]
    if i not in curriculum[k] :
        if i  in curriculum[-1]:
            cordata.append(i)
    print(cordata)
    sample = sample_X.loc[:,cordata]
    sample = sample.values
    # np.savetxt('X.csv', X, delimiter=',')
    W_est, loss = notears_linear(sample, lambda1=0.1, loss_type='l2')
    estc = pd.DataFrame(W_est)
    estc.columns = cordata
    estc.index = cordata
    cc = B_true.loc[cordata, cordata]
    cc = cc.values
    acc = utils.count_accuracy(cc , W_est != 0)
    print('!',acc)

    edges = {}
    dele = []
    for k in edge_est.keys() :#duplicate
        if (k[1],k[0]) in edge_est.keys():
            if edge_est[k] >= edge_est[(k[1],k[0])]:
                if (k[1],k[0]) not in dele:
                    dele.append((k[1],k[0]))
            else:
                if k not in dele:
                    dele.append(k)
    for i in dele:
        edge_est.pop(i)
    for k, v in edge_est.items():
        edges[k] = zhi[k]
    est_w = pd.DataFrame(0.00,columns=nodes,index=nodes)
    for k,v in edges.items():
        est_w.loc[k[0],k[1]] = edges[k]
    est_w= est_w.values
    acc = utils.count_accuracy(B_true, est_w != 0)
    print(acc)
    result = []
    result.append(network)
    for i in acc:
        result.append(i)
    for j in aacc[-1]:
        result.append(j)
    # print(acc)
    return result
"""for i in range(0,5):
    CL_Notears(1000,'gauss','sachs')"""
# 每次的新增节点
"""xiuc = []

for i in curriculum[len(est)-1] :
    if i not in curriculum[k]:
        xiuc.append(i)
sample = sample_X.loc[:, xiuc]
sample = sample.values
# np.savetxt('X.csv', X, delimiter=',')
W_est, loss = notears_linear(sample, lambda1=0.1, loss_type='l2')
estw = pd.DataFrame(W_est)
estw.columns = xiuc
estw.index = xiuc
print(loss)
B_true = nx.to_pandas_adjacency(model.to_directed(),nodelist=nodes,weight=None,nonedge=0)
a = B_true.loc[xiuc,xiuc]
a = a.values
acc = utils.count_accuracy(a, W_est != 0)
print(acc)

for i in xiuc:
    for j in xiuc:
        if i != j:
            if estw.at[i,j] != 0:
                print("xiao",(i,j))
                if (i, j) in edge_est.keys():
                    edge_est[(i, j)] += 0.1
                else:
                    edge_est[(i, j)] = 0.1
                zhi[(i, j)] = estw.at[i, j]
# 最后一次
for i in nodes:
    for j in  nodes:
        if i != j:
           if est[-1].at[i,j] != 0 :
               if (i, j) in edge_est.keys():
                   edge_est[(i, j)] += 0.1
               else:
                   edge_est[(i, j)] = 0.1
               zhi[(i, j)] = est[-1].at[i, j]


for i in curriculum[len(curriculum)]: #其他
    
    c = curriculum[len(curriculum)] -curriculum[k]
    sample = sample_X.loc[:,curriculum[i]]
    sample = sample.values
# np.savetxt('X.csv', X, delimiter=',')
    W_est, loss = notears_linear(sample, lambda1=0.1, loss_type='l2')
    estw = pd.DataFrame(W_est)
    estw.columns = curriculum[i]
    estw.index = curriculum[i]
    est.append(estw)
    aloss.append(loss)
    print(loss)
    acc = utils.count_accuracy(truea[i] , W_est != 0)
    print(acc)
for k in range(0,len(est)):
    if aloss[k] < aloss[k - 1] and k != 0:
        for i in curriculum[k]:
            for j in curriculum[k]:
                if i != j:
                    if est[k].at[i,j]==0 and (i in curriculum[k-1])and(j in curriculum[k-1]) and est[k-1].at[i,j]!=0:
                        edge_est[(i, j)] = 0
for k in range(k+1,len(est)):
    if aloss[k+1]< aloss[k] :
        weight = 0.2
    else:
        weight = 0.1
    for i in curriculum[k]:
        for j in curriculum[k]:
            if i != j:
                if est[k].at[i,j] !=0:
                    if (i,j) in edge_est.keys():
                        edge_est[(i,j)] +=weight
                    else:
                        edge_est[(i, j)] =weight
                    zhi[(i,j)] = est[k].at[i,j]
b = 0
for k, v in edge_est.items():
    b += v
if len(edge_est) == 0:
        b = 0
else:
    b = b / len(edge_est)
edges = {}
dele = []
for k, v in edge_est.items():
    if edge_est[k] < b * 0.5:
        dele.append(k)
"""

"""for i in curriculum[len(curriculum)-1]:
    for j in curriculum[len(curriculum)-1]:
        if i !=j:
            cu.append([i,j])
            a = B_true.loc[[i,j],[i,j]]
            a = a.values
            truea.append(a)

asamp = {}
aloss = {}
samples = []
for iter in range(1):
    for i in range(10):
        sample = sample_X.sample(n=50)
        sample = sample.values
    #np.savetxt('X.csv', X, delimiter=',')
        W_est,loss = notears_linear(sample, lambda1=0.1, loss_type='l2')
        print(loss)
        asamp[i] = sample
        aloss[i] = loss
    #对loss排序
    e =sorted(aloss.items(),key=lambda e:e[1])
    e1=e[0]
    samples.append(asamp[e1[0]])
esample = samples[0]
for i in range(1,len(samples)):
    esample = np.concatenate((esample,samples[i]),axis=0)
    W_est, loss = notears_linear(sample, lambda1=0.1, loss_type='l2')
    assert utils.is_dag(W_est)
    # np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)


assert utils.is_dag(W_est)
#np.savetxt('W_est.csv', W_est, delimiter=',')
acc = utils.count_accuracy(B_true, W_est!= 0)
print(acc)
"""
"""result = []
result.append(network)
for i in acc:
    result.append(i)
#print(acc)
return result



for i in range(0,len(curriculum)):
    a = B_true.loc[curriculum[i],curriculum[i]]
    a = a.values
    truea.append(a)
for i in curriculum[len(curriculum)-1]:
    for j in curriculum[len(curriculum)-1]:
        if i !=j:
            cu.append([i,j])
            a = B_true.loc[[i,j],[i,j]]
            a = a.values
            truea.append(a)
B_true = B_true.values#转换ndarray
W_true = utils.simulate_parameter(B_true)
X = utils.simulate_linear_sem(W_true, n, sem_type)
sample_X = pd.DataFrame(X)
sample_X.columns = curriculum[len(curriculum)-1] #对采样后的数据进行操作
for i in range(1,len(curriculum)):
    sample_X1 = sample_X.loc[:,curriculum[i]]
    sample_X1 = sample_X1.values
    W_est, loss = notears_linear(sample_X1, lambda1=0.1, loss_type='l2')#loss来充当监督者
    print(loss)
    assert utils.is_dag(W_est)
    acc = utils.count_accuracy(truea[i - 1], W_est != 0)
    print(acc)
"""
"""
for i in range(len(cu)):
    W_true = W_true1.loc[cu[i],cu[i]]
    W_true = W_true.values
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    W_est, loss = notears_linear(X, lambda1=0.1, loss_type='l2')#loss来充当监督者
    print(loss)
    assert utils.is_dag(W_est)
    acc = utils.count_accuracy(truea[i-1], W_est != 0)
    print(acc)

for i in range(0,len(curriculum)):
    a = B_true.loc[curriculum[i],curriculum[i]]
    a = a.values
    truea.append(a)
B_true = B_true.values#转换ndarray
W_true = utils.simulate_parameter(B_true)
W_true1 = pd.DataFrame(W_true)
W_true1.columns = curriculum[len(curriculum)-1]
W_true1.index = curriculum[len(curriculum)-1]
for i in range(1,len(curriculum)):
    W_true = W_true1.loc[curriculum[i],curriculum[i]]
    W_true = W_true.values
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    W_est, loss = notears_linear(X, lambda1=0.1, loss_type='l2')#loss来充当监督者
    print(loss)
    assert utils.is_dag(W_est)
    acc = utils.count_accuracy(truea[i-1], W_est != 0)
    print(acc)
#np.savetxt('W_true.csv', W_true, delimiter=',')
asam = {}
aloss ={}
X = utils.simulate_linear_sem(W_true, n, sem_type)
#np.savetxt('X.csv', X, delimiter=',')
W_est,loss = notears_linear(X, lambda1=0.1, loss_type='l2')
assert utils.is_dag(W_est)
acc = utils.count_accuracy(B_true, W_est != 0)
print(acc)"""
#X = pd.DataFrame(X)
"""asam[i]=X
aloss[i]=loss
print(aloss)
e =sorted(aloss.items(),key=lambda e:e[1])
e1= []
for  i in e:
    e1.append(i[0])
sampels = []
for i in e1:
    sampels.append(asam[i])"""
"""for i in sampels:
        = np.concatenate((a,b),axis=0)"""




"""assert utils.is_dag(W_est)
np.savetxt('W_est.csv', W_est, BayesianModelSamplingdelimiter=',')
acc = utils.count_accuracy(B_true, W_est!= 0)
print(acc)"""

"""utils.set_random_seed(1)


def cl_sEM(n,sem_type,network,curriculum):
    model = get_example_model(network)
    if curriculum == None:
        curriculum = model.nodes()
    B_true = nx.to_pandas_adjacency(model.to_directed(), nodelist=curriculum, weight=None, nonedge=0)
    B_true = B_true.values  # 转换ndarray
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    W_est,loss = notears_linear(X, lambda1=0.1, loss_type='l2') #loss损失函数，依靠损失函数来判断样本的强弱
    assert utils.is_dag(W_est)
    #np.savetxt('W_est.csv', W_est, BayesianModelSamplingdelimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    result = []
    result.append(network)
    for i in acc:
        result.append(i)
    print(acc)
    return result
from curriculum import  curr
# weight
for j in [1000,2000,5000]:
    for i in range(4):
        c=curr('sachs')
        cl_sEM(j,'gauss','sachs',c)

def cl(n,sem_type,network,curriculum):
    model = get_example_model(network)
    if curriculum == None:
        curriculum = model.nodes()
    B_true = nx.to_pandas_adjacency(model.to_directed(), nodelist=curriculum, weight=None, nonedge=0)
    B_true = B_true.values  # 转换ndarray
    W_true = utils.simulate_parameter(B_true)
    while samp != n:
        samp = n/4
        X = utils.simulate_linear_sem(W_true, samp, sem_type)
        W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
        assert utils.is_dag(W_est)
        np.savetxt('W_est.csv', W_est, BayesianModelSamplingdelimiter=',')
        acc = utils.count_accuracy(B_true, W_est != 0)

"""
