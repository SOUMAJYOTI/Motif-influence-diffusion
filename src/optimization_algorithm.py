import numpy as np
import pickle
import pandas as pd
from sklearn import linear_model
import random
import datetime
import time
from scipy.optimize import minimize


randomTime = datetime.datetime.strptime('2011-08-01', '%Y-%m-%d')
randomTime = time.mktime(randomTime.timetuple())

k = 3000 # no. of cascades to consider for traning network
degDict = pd.read_pickle('../../../data/deg_centralities_diff_T07_08-v1.pcikle')
inputDf = pd.read_pickle('../../../data/df_optimize_input_v1.0+.pickle')
midList = list(set(inputDf['mid']))[:k]

Nfeval = 1
inputDf = inputDf[inputDf['mid'].isin(midList)]

outDeg, inDeg = degDict
count_pairs_alpha = 0
nodePairs_alpha = {}

count_pairs_eta = 0
nodePairs_eta = {}
# Keep the list of unique source-target pairs for parameter retrieval
for idx, row in inputDf.iterrows():
    currNode = row['node']
    parent = row['parents']
    nonParents = row['nonParents']
    exposedNodes = row['exposedNodes']

    # For alpha parameter
    if (currNode, parent) not in nodePairs_alpha:
        nodePairs_alpha[(currNode, parent)] = count_pairs_alpha
        count_pairs_alpha += 1

    for k in nonParents:
        if (currNode, k) not in nodePairs_alpha:
            nodePairs_alpha[(currNode, k)] = count_pairs_alpha
            count_pairs_alpha += 1

    # For eta paramter
    for k in exposedNodes:
        if (currNode, k) not in nodePairs_eta:
            nodePairs_eta[(currNode, k)] = count_pairs_eta
            count_pairs_eta += 1


def lasso_solve(X, y):
    '''
    Lasso solver for least squares minimization
    :param X:
    :param y:
    :return:
    '''

    alphas = [0.01, 0.02, 0.03, 0.04]
    regr = linear_model.Lasso()
    scores = [regr.set_params(alpha=alpha).fit(X, y).score(X, y) for alpha in
              alphas]
    best_alpha = alphas[scores.index(max(scores))]
    regr.alpha = best_alpha
    regr.fit(X, y)


def G_1_alpha(alpha):
    '''

    :param inputDf: Dataframe containing the node information for each node
    :param: alpha: transmission rate
    :return:
    '''

    nodeList = inputDf['node'].tolist()
    timeList = inputDf['time'].tolist()

    nodeRtTime = dict(zip(nodeList, timeList))
    func_LL = 0. # function log likelihood

    # Compute the log likelihood for each cascade
    last = randomTime
    survival = 0.
    hazard = 0.

    for idx, row in inputDf.iterrows():
        currNode = row['node']
        parent = row['parents']
        nonParents = row['nonParents']

        t_i = row['time']

        # 1. Survival term
        for k in nonParents:
            # parent should not be in nonParents
            if k == row['parents']:
                continue
            if k not in nodeRtTime:
                t_k = last - 1
            else:
                t_k = nodeRtTime[k]
            timeDiff = int(int(t_i - t_k) / (60*60))
            survival += (-alpha[nodePairs_alpha[(currNode, k)]] * (np.power(timeDiff, 2) * 0.5))

        # 2. Hazard term
        j = parent
        if j not in nodeRtTime:
            t_j = last - 1
        else:
            t_j = nodeRtTime[j]
        timeDiff = int(int(t_i - t_j) / (60*60))
        if alpha[nodePairs_alpha[(currNode, j)]] <= 0:
            alpha[nodePairs_alpha[(currNode, j)]] = 1.
        hazard = np.log(alpha[nodePairs_alpha[(currNode, j)]] * (abs(timeDiff)+2))
        last = row['time']

        # Add the log terms
        # print(survival, hazard)
        func_LL += (survival + hazard)
        # print(func_LL)

    return -func_LL # return the negative log likelihood


def G_1_eta(eta):
    '''

    :param inputDf: Dataframe containing the node information for each node
    :param: eta: exposure rate
    :return:
    '''

    nodeList = inputDf['node'].tolist()
    timeList = inputDf['time'].tolist()

    nodeRtTime = dict(zip(nodeList, timeList))
    func_LL = 0. # function log likelihood

    # Compute the log likelihood for each cascade
    last = randomTime
    exposure = 0.
    curr_cascade_funcValue = 0.
    for idx, row in inputDf.iterrows():
        currNode = row['node']
        exposureNodes = row['exposedNodes']

        t_i = row['time']

        # 3. Exposure term - from motif nodes
        for m in exposureNodes:
            if m not in nodeRtTime:
                t_m = last - 1
            else:
                t_m = nodeRtTime[m]
            timeDiff = int(int(t_i - t_m) / (60*60))
            if eta[nodePairs_eta[(currNode, m)]] <= 0:
                eta[nodePairs_eta[(currNode, m)]] = 1.
            exposure += np.log(eta[nodePairs_eta[(currNode, m)]] * (1/(abs(timeDiff)+2)))

        last = row['time']

        curr_cascade_funcValue += exposure
        func_LL += curr_cascade_funcValue

    return -func_LL # return the negative log likelihood


def lasso_G_2():
    '''

    :param inputDf: Dataframe containing the node information for each node
    :param: degDict: degree dictionary of nodes
    :return:
    '''

    X = []
    for idx, row in inputDf.iterrows():
        currNode = row['node']
        nonParents = row['nonParents']

        try:
            x_i = degDict[str(currNode)]
        except:
            x_i = int(np.random.uniform(1, 3, 1)) # sample from uniform distribution

        for k in nonParents:
            # parent should not be in nonParents
            if k == row['parents']:
                continue

            try:
                x_k = degDict[str(k)]
            except:
                x_k = int(np.random.uniform(1, 3, 1))  # sample from uniform distribution

            X.append([x_k * x_i])

    return np.array(X)


def lasso_G_3():
    '''

    :param inputDf: Dataframe containing the node information for each node
    :param: degDict: degree dictionary of nodes
    :return:
    '''

    X = []
    for idx, row in inputDf.iterrows():
        exposureNodes = row['exposedNodes']
        for m in exposureNodes:
            try:
                x_m = degDict[str(m)]
            except:
                x_m = int(np.random.uniform(1, 3, 1) ) # sample from uniform distribution

            X.append([x_m])

    return np.array(X)


def callbackF(Xi):
    global Nfeval
    # print(jacobian(Xi), np.linalg.norm(jacobian(Xi)))
    # print('{0:4d}   {1: 3.6f}   {2: 3.6f}  {3: 3.6f} {4:3.6f} '.format(Nfeval, Xi[0], Xi[1], f(Xi), np.linalg.norm(jacobian(Xi))))
    print(Nfeval, G_1_eta(Xi))

    Nfeval += 1


def optimize_coordinate_descent():
    maxIter = 500

    constantInit = np.random.uniform(0.1, 1, 1)[0]
    alpha = [constantInit for _ in range(count_pairs_alpha)]
    eta = [constantInit for _ in range(count_pairs_eta)]

    beta = constantInit
    gamma = constantInit

    iterCount = 0
    print(len(alpha))
    while iterCount < maxIter:
        print("Iteration: ", iterCount)
        LL = G_1_alpha(alpha)
        results = minimize(G_1_alpha, np.array(alpha), method='nelder-mead', options={'maxiter':100})#, callback=callbackF)
        alpha = results.x

        LL = G_1_eta(eta)
        results = minimize(G_1_eta, np.array(eta), method='nelder-mead', options={'maxiter':100})#, callback=callbackF)
        eta = results.x

        clf_G_2 = linear_model.Lasso(alpha=0.001)
        X_G_2 = lasso_G_2()
        clf_G_2.fit(X_G_2, alpha)
        beta = clf_G_2.coef_
        # print(beta)

        clf_G_3 = linear_model.Lasso(alpha=0.001)
        X_G_3 = lasso_G_3()
        clf_G_3.fit(X_G_3, eta)
        gamma = clf_G_3.coef_

        iterCount += 1

    return alpha, eta, beta, gamma

def main():
    print('Starting optimization....')
    alpha, eta, beta, gamma = optimize_coordinate_descent()
    pickle.dump((alpha, eta, beta, gamma), open('parameters.pickle', 'wb'))

if __name__ == '__main__':
    main()