import numpy as np
import pickle
import pandas as pd
from sklearn import linear_model
import random
import datetime
import time
from collections import *

randomTime = datetime.datetime.strptime('2011-08-01', '%Y-%m-%d')
randomTime = time.mktime(randomTime.timetuple())

graphDf = pd.read_pickle('../data/motif_preprocess/v1/df_graph_s3.pickle')

k = 1799 # no. of cascades to consider for traning network
# degDict = pd.read_pickle('../data/deg_centralities_diff_T07_08-v1.pcikle')
inputDf = pd.read_pickle('../data/df_optimize_input_v1.0+.pickle')

count = 0
midList = list(set(inputDf['mid']))


def split_training_test(midList):
    train = int (0.8 * ( len(midList)))
    test = 0.2

    midTrain = midList[:train]
    midTest = midList[train:]
    trainDf = inputDf[inputDf['mid'].isin(midTrain)]
    testDf = inputDf[inputDf['mid'].isin(midTest)]

    return trainDf, testDf


def main():
    # Keep the list of unique source-target pairs for parameter retrieval

    trainDf, testDf = split_training_test(midList)

    actions_u = defaultdict(int)
    actions_u2v = {}

    midTrain = list(set(trainDf['mid']))
    midTest = list(set(testDf['mid']))

    # Training process
    print("training......")
    for mid in midTrain:
        cascade = inputDf[inputDf['mid'] == mid]
        nodes_seen = []

        for idx, row in cascade.iterrows():
            currNode = row['node']
            parent = row['parents']

            if currNode not in nodes_seen:
                nodes_seen.append(currNode)
                actions_u[currNode] += 1

            if (parent, currNode) not in actions_u2v:
                actions_u2v[(parent, currNode)] = 1
            else:
                actions_u2v[(parent, currNode)] += 1

    # Testing process
    print("testing....")
    recall_sum = defaultdict(int)

    for mid in midTest:
        cnt_intervals = 0
        prev_nodes = set([])
        parents = {}
        cascade = inputDf[inputDf['mid'] == mid]

        for idx, row in cascade.iterrows():
            node = row['node']

            act_parent = row['parents']
            prev_nodes.add(act_parent)
            for np in row['nonParents']:
                prev_nodes.add(np)
            for exp in row['exposedNodes']:
                prev_nodes.add(exp)

            parents[node] = act_parent

            corr_pred = 0
            if len(parents) > 40:
                A_s2t = {}
                for n in parents:
                    for p in prev_nodes:
                        if (p, n) in actions_u2v:  # and p in actions_u:
                            # if actions_u[p] > 0:
                            A_s2t[(p, n)] = actions_u2v[(p, n)]

                    maxVal = -10
                    pred_parent = ''
                    for p, c in A_s2t:
                       # pred_parent[p] = A_s2t[(p, c)]
                        if A_s2t[(p, c)] > maxVal:
                            maxVal = A_s2t[(p, c)]
                            pred_parent  = p

                    if parents[n] == pred_parent: #in list(pred_parent.keys()):
                        corr_pred += 1

                prev_nodes = set([])
                recall_sum[cnt_intervals] += (corr_pred/len(parents))
                parents = {}

                cnt_intervals += 1

    for interval in recall_sum:
        print("Intervals: ", interval, recall_sum[interval] / len(midTest))


if __name__ == "__main__":
    main()









