import numpy as np
import pickle
import pandas as pd
from sklearn import linear_model
import random
import datetime
import time
from scipy.optimize import minimize


randomTime = datetime.datetime.strptime('2011-08-01', '%Y-%m-%d')
randomTime = time.mktime(randomTime.timetuple())\


def main():
    print('Starting evaluation ....')
    alphaVal = 0.5
    etaVal = 0.1

    degDict = pd.read_pickle('../../../data/deg_centralities_diff_T07_08-v1.pcikle')
    inputDf = pd.read_pickle('../../../data/df_optimize_input_sample.pickle')
    midList = list(set(inputDf['mid']))[:3]

    inputDf = inputDf[inputDf['mid'].isin(midList)]

    outDeg, inDeg = degDict

    numTrainNodes = 100
    numTestNodes = 30 # k

    midList = list(inputDf['mid'])[:100]

    for mid in midList:
        trainNodes = []
        testNodes = []

        allPairs = []
        actual_edges = []
        output_edges = []
        currDf = inputDf[inputDf['mid'] == mid]
        timeList = {}
        parentList = {}

        for idx, row in currDf.iterrows():
            currNode = row['node']
            parent = row['parents']
            timeList[currNode] = row['time']
            timeList[parent] = row['time']

            parentList[currNode] = parent
            if len(trainNodes) < 50:
                if currNode not in trainNodes:
                    trainNodes.append(currNode)
                if parent not in trainNodes:
                    trainNodes.append(parent)

            else:
                if len(testNodes) < numTestNodes:
                    allPairs.append((parent, currNode))
                    if currNode not in testNodes and currNode not in trainNodes:
                        testNodes.append(currNode)

                    # if parent not in testNodes:
                    #     testNodes.append(parent)
                else:
                    break

        for tsNode in testNodes:
            randomNode = random.choice(trainNodes)
            # try:
            #     x_i = degDict[str(tsNode)]
            # except:
            #     x_i = int(np.random.uniform(1, 3, 1))  # sample from uniform distribution
            #
            # tsTime = timeList[tsNode]
            #
            # maxVal = -100000
            # estParent = ''
            # for trNode in trainNodes:
            #     trTime = timeList[trNode]
            #     try:
            #         x_j = degDict[str(trNode)]
            #     except:
            #         x_j = int(np.random.uniform(1, 3, 1))  # sample from uniform distribution
            #
            #     timeDiff = int(int(tsTime - trTime) / (60))
            #     alpha_ji = alphaVal * x_i * x_j
            #
            #     LL_hazard = np.log(alpha_ji * (abs(timeDiff)+2))
            #
            #     LL_survival = -alpha_ji * (np.power(abs(timeDiff)+2, 2) * 0.5)
            #
            #     LL_total = LL_survival + LL_hazard
            #
            #     if LL_total >= maxVal:
            #         maxVal = LL_total
            #         estParent = trNode
            #     # if timeDiff == 0:
            #     #     print(trNode, tsNode, trTime, tsTime)

            # print(maxVal)
            output_edges.append((randomNode, tsNode))

        accuracy = len(list(set(output_edges).intersection(set(allPairs)))) / len(allPairs)
        print(accuracy)


if __name__ == '__main__':
    main()