import numpy as np
import pickle
import pandas as pd
from sklearn import linear_model
from scipy.optimize import minimize
import random

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


def G_1(inputDf, alpha, eta):
    '''

    :param inputDf: Dataframe containing the node information for each node
    :param: alpha: transmission rate
    :param: eta: exposure rate
    :param: degDict: degree dictionary of nodes
    :return:
    '''

    nodeList = inputDf['node'].tolist()
    timeList = inputDf['time'].tolist()

    nodeRtTime = dict(zip(nodeList, timeList))
    func_LL = 0. # function log likelihood

    # Compute the log likelihood for each cascade
    last = 0.
    survival = 0.
    exposure = 0.
    hazard = 0.
    curr_cascade_funcValue = 0.
    for idx, row in inputDf.iterrows():
        nonParents = row['nonParents']
        exposureNodes = row['exposedNodes']

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

            survival += (-alpha[idx] * (np.power(t_i - t_k, 2) * 0.5))


        # 2. Hazard term
        j = row['parents']
        if j not in nodeRtTime:
            t_j = last - 1
        else:
            t_j = nodeRtTime[j]


        alpha_ji = alpha[idx]
        hazard = np.log(alpha_ji * (t_i - t_j))

        # 3. Exposure term - from motif nodes
        for m in exposureNodes:
            if m not in nodeRtTime:
                t_m = last - 1
            else:
                t_m = nodeRtTime[m]

            exposure += np.log(eta[idx] * (1/(t_i - t_m)))
        last = row['time']

        curr_cascade_funcValue += survival + hazard + exposure

        func_LL += curr_cascade_funcValue

    return -func_LL # return the negative log likelihood


def lasso_G_2(inputDf, degDict):
    '''

    :param inputDf: Dataframe containing the node information for each node
    :param: degDict: degree dictionary of nodes
    :return:
    '''

    alpha = []
    for idx, row in inputDf.iterrows():
        currNode = row['node']
        nonParents = row['nonParents']

        try:
            x_i = degDict[str(currNode)]
        except:
            x_i = np.random.uniform(0,.1,1) # sample from uniform distribution

        for k in nonParents:
            # parent should not be in nonParents
            if k == row['parents']:
                continue

            try:
                x_k = degDict[str(k)]
            except:
                x_k = np.random.uniform(0, .1, 1)  # sample from uniform distribution

            alpha.append([x_k * x_i])

    return np.array(alpha)


def lasso_G_3(inputDf, gamma, degDict):
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
                x_m = np.random.uniform(0, .1, 1)  # sample from uniform distribution

            eta.append([gamma[idx] * x_m])

    return np.array(eta)




def optimize_coordinate_descent(inputDf, degDict):
    maxIter = 100
    len_input = len(inputDf)

    alpha_init = np.array(random.sample(range(0.001, 0.1), len_input))
    beta_init = np.array(random.sample(range(0.001, 0.1), len_input))
    clf = linear_model.Lasso(alpha=0.001)
    X = lasso_G_2(inputDf, degDict)
    print(X)
    # clf.fit(alpha_init, X)
    # beta = clf.coef_

    # for iter in range(maxIter):
    #
    #     result = minimize(funcX, [inputDf])
    #     if result.success:
    #         fitted_params = result.x
    #         print(fitted_params)
    #     else:
    #         raise ValueError(result.message)


def main():
    degDict = pd.read_pickle('../../../data/deg_centralities_diff_T07_08-v1.pcikle')
    df_opt = pd.read_pickle('../../../data/df_optimize_input_sample.pickle')
    optimize_coordinate_descent(df_opt, degDict)
    # print(df_opt[:10])
    # funcX(df_opt, 0.5, 0.5)

if __name__ == '__main__':
    main()