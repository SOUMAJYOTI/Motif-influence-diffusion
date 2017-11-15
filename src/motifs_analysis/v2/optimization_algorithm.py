import numpy as np
import pickle
import pandas as pd
from sklearn import linear_model
from scipy.optimize import minimize
import random

degDict = pd.read_pickle('../../../data/deg_centralities_diff_T07_08-v1.pcikle')
inputDf = pd.read_pickle('../../../data/df_optimize_input_sample.pickle')
midList = list(set(inputDf['mid']))[:10]

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


def binarySearch(alist, item):
    first = 0
    last = len(alist)-1
    found = False

    while first<=last and not found:
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1

    return found


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


def G_1_alpha(inputDf, alpha):
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
    last = 0.
    survival = 0.
    hazard = 0.
    curr_cascade_funcValue = 0.

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
            survival += (-alpha[nodePairs_alpha[(currNode, k)]] * (np.power(t_i - t_k, 2) * 0.5))

        # 2. Hazard term
        j = parent
        if j not in nodeRtTime:
            t_j = last - 1
        else:
            t_j = nodeRtTime[j]
        hazard = np.log(alpha[nodePairs_alpha[(currNode, j)]] * abs(t_i - t_j))

        last = row['time']

        # Add the log terms
        curr_cascade_funcValue += survival + hazard
        func_LL += curr_cascade_funcValue

    return -func_LL # return the negative log likelihood


def G_1_eta(inputDf, eta):
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
    last = 0.
    exposure = 0.
    curr_cascade_funcValue = 0.
    for idx, row in inputDf.iterrows():
        cuurNode = row['node']
        exposureNodes = row['exposedNodes']

        t_i = row['time']

        # 3. Exposure term - from motif nodes
        for m in exposureNodes:
            if m not in nodeRtTime:
                t_m = last - 1
            else:
                t_m = nodeRtTime[m]
            exposure += np.log(eta[nodePairs_eta[(currNode, m)]] * abs(1/(t_i - t_m)))

        last = row['time']

        curr_cascade_funcValue += exposure
        func_LL += curr_cascade_funcValue

    return -func_LL # return the negative log likelihood


def lasso_G_2(inputDf, degDict):
    '''

    :param inputDf: Dataframe containing the node information for each node
    :param: degDict: degree dictionary of nodes
    :return:
    '''

    alpha = []
    mid = []
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

            alpha.append([x_k * x_i])

    return np.array(alpha)


def lasso_G_3(inputDf, degDict):
    '''

    :param inputDf: Dataframe containing the node information for each node
    :param: degDict: degree dictionary of nodes
    :return:
    '''

    eta = []
    for idx, row in inputDf.iterrows():
        exposureNodes = row['exposedNodes']
        for m in exposureNodes:
            try:
                x_m = degDict[str(m)]
            except:
                x_m = int(np.random.uniform(1, 3, 1) ) # sample from uniform distribution

            eta.append([x_m])

    return np.array(eta)


def optimize_coordinate_descent(inputDf, degDict):
    maxIter = 100


    # alpha_init = np.array(random.sample(range(0.001, 0.1), len_input))
    # # beta_init = np.array(random.sample(range(0.001, 0.1), len_input))
    # clf_G_2 = linear_model.Lasso(alpha=0.001)
    # X_G_2 = lasso_G_2(inputDf, degDict)
    # beta_init = np.array([random.uniform(0, 5) for _ in range(X_G_2.shape[0])])
    #
    # # print(X, beta_init)
    # clf_G_2.fit(X_G_2, beta_init)
    # beta = clf_G_2.coef_
    # print(beta)

    # clf_G_3 = linear_model.Lasso(alpha=0.001)
    # X_G_3 = lasso_G_3(inputDf, degDict)
    # gamma_init = np.array([random.uniform(1, 5) for _ in range(X_G_3.shape[0])])
    #
    # # print(X, beta_init)
    # clf_G_3.fit(X_G_3, gamma_init)
    # gamma = clf_G_3.coef_






        # for iter in range(maxIter):
    #
    #     result = minimize(funcX, [inputDf])
    #     if result.success:
    #         fitted_params = result.x
    #         print(fitted_params)
    #     else:
    #         raise ValueError(result.message)


def main():
    print('Starting optimization....')
    optimize_coordinate_descent(df_opt, outDeg)
    # print(df_opt[:10])
    # funcX(df_opt, 0.5, 0.5)

if __name__ == '__main__':
    main()