import numpy as np
import pickle
import pandas as pd
from sklearn import linear_model
import random
import datetime
import time
from optimization_algorithm import *

randomTime = datetime.datetime.strptime('2011-08-01', '%Y-%m-%d')
randomTime = time.mktime(randomTime.timetuple())\


def partitionTrainTest(currDf):
    numTrainNodes = 40
    timeListCurr = {}
    timeListParents = {}
    parentList = {}

    time_list_global = {}
    # Partitioning the training and testing nodes into sets
    train_set = []
    test_set = []
    ParentSet = []
    timeSet = []
    ParentTimeSet = []
    trainNodes = []
    global_nodes = []

    count_set = 0
    for idx, row in currDf.iterrows():
        if len(trainNodes) > numTrainNodes: # set complete
            test_set.append(trainNodes)
            trainNodes = []
            train_set.append(global_nodes)
            global_nodes = []
            ParentSet.append(parentList)
            parentList = {}
            timeSet.append(timeListCurr)
            timeListCurr = {}
            ParentTimeSet.append(timeListParents)
            timeListParents = {}

            count_set += 1

        currNode = row['node']
        parent = row['parents']

        timeListCurr[currNode] = row['time']
        time_list_global[currNode] = row['time']

        if parent not in time_list_global:
            timeListParents[parent] = randomTime
        else:
            timeListParents[parent] = time_list_global[parent]

        parentList[currNode] = parent

        if parent not in global_nodes:
            global_nodes.append(parent)
        if currNode not in global_nodes:
            global_nodes.append(currNode)

        # if count_set > 0:
        #     if currNode in train_set[count_set-1]:
        #         trainNodes.append(currNode)

        # if count_set == 0:
        if currNode not in trainNodes:
            trainNodes.append(currNode)

    return train_set, test_set,  ParentSet, timeSet, ParentTimeSet



def motif_inference(train_set, test_set, trainTimes, testTimes, thresh, beta, gamma):
    print("__Start__")

    edges_infer = []
    for n in test_set:
        max = -100
        final_parent = 0
        for (src, tgt) in train_set:
            if testTimes[n] - trainTimes[tgt] > thresh:
                continue

            exposure_ll = G_1_eta
            trans_LL = G_1_alpha

            if exposure_ll * trans_LL > max:
                max = exposure_ll * trans_LL
                final_parent = tgt

        edges_infer.append((tgt, n))

    return edges_infer

def main():
    print('Starting evaluation ....')

    # Load the parameters from the previous optimization algorithm
    # alphaVal = 0.5
    # etaVal = 0.1
    betaVal = 1.
    gammaVal = 1.

    # degDict = pd.read_pickle('../../../data/deg_centralities_diff_T07_08-v1.pcikle')
    inputDf = pd.read_pickle('../data/df_optimize_input_sample.pickle')

    # print(inputDf[:10])

    # outDeg, inDeg = degDict

    # midList = list(inputDf['mid'])[:100]

    midList = list(set(inputDf['mid']))[:1]
    inputDf = inputDf[inputDf['mid'].isin(midList)]

    for mid in midList:
        allPairs = []
        actual_edges = []
        output_edges = []

        currDf = inputDf[inputDf['mid'] == mid]

        train_set, test_set, ParentSet, timeSet, ParentTimeSet = partitionTrainTest(currDf)
        for idx in range(1, len(train_set)):
            trainNodes = train_set[idx-1]
            testNodes = test_set[idx]
            testParents = ParentSet[idx]

            print(len(trainNodes), len(testParents))
            print(len(list(set(trainNodes).intersection(set(list(testParents.values()))))))



if __name__ == '__main__':
    main()