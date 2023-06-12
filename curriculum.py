import math
import copy
import pandas as pd
from collections import Counter
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
def getEntropy(X,str):
    data = X[str]
    counts = len(data)  # 总数据量
    counter = Counter(data)  # 每个离散变量出现的次数
    prob = [i[1] / counts for i in counter.items()]  # 计算每个随机变量发生的概率的p
    shannon = - sum([i * math.log(i) for i in prob])  # 计算信息熵
    return shannon

def getCondEntropy(data, str1, str2):#条件熵
    entr1 = data.groupby(str1).apply(lambda x:getEntropy(x,str2))
    prob2 = pd.value_counts(data[str1])/len(data[str1])
    con_ent = sum(entr1*prob2)
    return con_ent

def getEntropyGain(data,s1,s2):#熵增益
    return getEntropy(data,s2)-getCondEntropy(data,s1,s2) #s2的熵减s2条件下s1的熵

def getEntropyGainRatio(data,s1,s2):#熵增益率
    return  getEntropyGain(data,s1,s2)/getEntropy(data,s2) #熵增益除以熵

def getDiscreteCorr(data,s1,s2):#衡量离散值的相关性
    if math.sqrt(getEntropy(data,s1)*getEntropy(data,s2)) == 0:
        return  0
    else:
        return getEntropyGain(data,s1,s2)/math.sqrt(getEntropy(data,s1)*getEntropy(data,s2))
degree_2 = ['asia', 'cancer', 'earthquake', 'survey', 'child']
degree_3 = ['sachs', 'insurance', 'mildew']
degree_4 = ['alarm', 'barley', 'hailfinder']

def get_H(data):
    lable=[c for c in data]
    f_max= 10000
    max_fi = 0
    for i in lable:
        if getEntropy(data,i) <= f_max:
            #min = infor(data[i])
            max_fi = i
            f_max = getEntropy(data,i)
        #print("min_entrop:{},node:{}".format(f_min,min_fi))
    return max_fi
def max_MI(datasets, cs, ne):
    # 课程点集和候选点集
    m_ij = dict()
    node = dict()
    # print(ne)
    # 获取索引
    for i in ne:  # 候选集合
        for j in cs:  # 课程集合
            m_ij[i, j] = getDiscreteCorr(datasets, i, j)
        b = 0
        for k, v in m_ij.items():
            # print(we)
            b += v
        node[i] = b
    if i in ne:
        max_value_nodes = max(node.values())
    else:
        print("node is not in candidate")
    # 找到最大的互信息点
    for k, v in node.items():
        if v == max_value_nodes:
            #print("max MI_node：{}，value：{}".format(k, v))
            # k 为一个字符串
            return k
def min_MI(datasets, cs, ne):
    # 课程点集和候选点集
    m_ij = dict()
    node = dict()
    # print(ne)
    # 获取索引
    for i in ne:  # 候选集合
        for j in cs:  # 课程集合
            m_ij[i, j] = getDiscreteCorr(datasets, i, j)
        b = 0
        for k, v in m_ij.items():
            # print(we)
            b += v
        node[i] = b
    if i in ne:
        max_value_nodes = min(node.values())
    else:
        print("node is not in candidate")
    # 找到最大的互信息点
    for k, v in node.items():
        if v == max_value_nodes:
            #print("max MI_node：{}，value：{}".format(k, v))
            # k 为一个字符串
            return k
def upd(a,str1,str2):
    str1.append(a)
    str2.remove(a)

def anticurr(network):
    if network in degree_2:
        nc = 2
    elif network in degree_3:
        nc = 3
    elif network in degree_4:
        nc = 4
    model = get_example_model(network)
    # model = get_example_model('sachs')
    samples = BayesianModelSampling(model).forward_sample(size=int(10000))
    ncandi = [c for c in model]  # 候选集合
    c = len([c for c in model])
    #curriculms = []
    f = {}
    stage = 0
    for i in ncandi:
        f[i] = getEntropy(samples, i)
        for j in ncandi:
            if i != j:
                f[i] += getDiscreteCorr(samples, i, j)
    for key, value in f.items():
        if (value == max(f.values())):
            node = key
    fnode = node
    curriculum = []
    upd(fnode, curriculum, ncandi)
    curriculm_set = {0:[fnode]}
    while len(ncandi) != 0:
        for i in range(0,nc):
            if len(ncandi) != 0:
                result = str(min_MI(samples,curriculum,ncandi))
                upd(result, curriculum, ncandi)

        curriculm_set[stage] = copy.deepcopy(curriculum)
        stage += 1
    #print(curriculm_set)
    return  curriculm_set

def curr(network):
    if network in degree_2:
        nc = 2
    elif network in degree_3:
        nc = 3
    elif network in degree_4:
        nc = 4
    model = get_example_model(network)
    # model = get_example_model('sachs')
    samples = BayesianModelSampling(model).forward_sample(size=int(10000))
    ncandi = [c for c in model]  # 候选集合
    c = len([c for c in model])
    # curriculms = []
    f = {}
    stage = 0
    for i in ncandi:
        f[i] = getEntropy(samples, i)
        for j in ncandi:
            if i != j:
                f[i] += getDiscreteCorr(samples, i, j)
    for key, value in f.items():
        if (value == min(f.values())):
            node = key
    fnode = node
    curriculum = []
    upd(fnode, curriculum, ncandi)
    curriculm_set = {0: [fnode]}
    while len(ncandi) != 0:
        for i in range(0, nc):
            if len(ncandi) != 0:
                result = str(max_MI(samples, curriculum, ncandi))
                upd(result, curriculum, ncandi)
        if len(ncandi) == 1:
            upd(ncandi[-1], curriculum, ncandi)
        curriculm_set[stage] = copy.deepcopy(curriculum)
        stage += 1
    # print(curriculm_set)
    return curriculm_set
    """ for j in ncandi:
        mi[j] = getEntropy(samples,j)
        for i in curriculum:
            #mi[j] += getDiscreteCorr(samples, i, j)+getCondEntropy(samples,j,i)#['smoke', 'lung', 'either', 'xray', 'tub', 'dysp', 'bronc', 'asia']
            #mi[j] += getDiscreteCorr(samples, i, j) +getCondEntropy(samples,j,i) * 0.1*(c-curriculum.index(i))#larm
            mi[j] += getDiscreteCorr(samples, j, i)
    for key, value in mi.items():
        if (value == max(mi.values())):
            node = key
    upd(node, curriculum, ncandi)
    mi[node] = 0
print(curriculum)"""

