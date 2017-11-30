from graph_tool.all import *
import graph_tool as gt
import graph_tool.stats as gts
import graph_tool.util as gtu
import graph_tool.draw as gtd
from pylab import *
from math import *
from numpy.random import *
import pickle
import pandas as pd
import os
import glob
import csv
import statistics as st
import random
import graph_tool.topology as gtt
import operator
import graph_tool.centrality as gtc
import time
import itertools
import csv
import resource
import time
import multiprocessing
import threading
import random

# Read the cascade intreval-wised edges dataframe
directory = '../../data/motif_preprocess/v1'
# dataDf = pickle.load(open(directory + '/df_graph_s3.pickle', 'rb'))
steep_inhib_times = pickle.load(open('../../data/steep_inhib_times.pickle', 'rb'))
dataDf = pickle.load(open('../../data/motif_preprocess/v1/exposureNodes_DF.pickle', 'rb'))


def init(args):
    global count_motifs
    count_motifs = args


def checkIsomorphism(graph_list, g):
    for gr in graph_list:
        if gtt.isomorphism(gr, g):
            return gr
    return False


def motif_operation(mid):
    cascade_set = dataDf[dataDf['mid'] == mid]

    print("Cascade of mid: ", mid, " and reshare size: ", len(cascade_set))

    numIntervals = np.max(np.array(list(cascade_set['interval_2:'])))

    midList = []
    nodeList = []
    timeList = []
    parentList = []
    exposedNodesList = []
    exposedTimeList = []
    nonParentsNodesList = []
    nonParentsTimeList = []
    notParticipatedList = []

    all_nodes = list(set(cascade_set['source']).union(cascade_set['target']))
    cum_all = []
    for int_idx in range(1, numIntervals):
        cascade_intervalDf_prev= cascade_set[cascade_set['interval_1'] == int_idx-1]
        cascade_intervalDf_curr = cascade_set[cascade_set['interval_1'] == int_idx]
        # cascade_intervalDf = pd.concat([cascade_intervalDf_prev, cascade_intervalDf_curr])

        cascade_Df_curr = cascade_intervalDf_curr[cascade_intervalDf_curr['edge_type']=='cascade'] # for current interval only


        # historical_Df = cascade_intervalDf[cascade_intervalDf['edge_type']=='historical'] # for both intervals
        # print(cascade_Df_curr)
        # cascade_Df_curr = cascade_Df_curr.sort['retweet_time']

        # # Store the parents who exposed this to node - historical edges
        # exposed_parents = {}
        #
        # for i, r in historical_Df.iterrows():
        #     src = r['source']
        #     tgt = r['target']
        #     if tgt not in exposed_parents:
        #         exposed_parents[tgt] = []

            # exposed_parents[tgt].append((src, r['retweet_time']))

        # Seen  nodes of previous interval - initial for this interval
        cum_nodes = list(set(cascade_intervalDf_prev['source']).union(set(cascade_intervalDf_prev['target'])))

        # Consider only nodes in current interval and only with cascade edges
        for i, r in cascade_Df_curr.iterrows():
            if r['target'] in nodeList:
                continue
            midList.append(mid)
            nodeList.append(r['target'])
            timeList.append(r['retweet_time'])
            # print(r['retweet_time'])
            parentList.append(r['source'])
            if len(r['exposureNodes']) == 0:
                exposedNodesList.append(r['source'])
            else:
                exposedNodesList.append(r['exposureNodes'])

            # find all the non parents - remove the source if present in cum_nodes
            nonParentsNodesList.append(list(set(cum_nodes) - set([r['source']])))
            cum_nodes.append(r['source'])
            cum_nodes.append(r['target'])

            cum_all.extend(list(set(cum_nodes)))
            notParticipatedList.append(list(set(all_nodes) - set(cum_all)))

    df_store = pd.DataFrame()
    df_store['mid'] = midList
    df_store['node'] = nodeList
    df_store['time'] = timeList
    df_store['parents'] = parentList
    df_store['exposedNodes'] = exposedNodesList
    df_store['nonParents'] = nonParentsNodesList
    df_store['notParticipated'] = notParticipatedList


    return df_store


def main():
    global count_motifs
    count_motifs = multiprocessing.Value('i', 0)

    numProcessors = 5
    pool = multiprocessing.Pool(numProcessors, initializer=init, initargs=(count_motifs,))

    print("Loading cascade data...")
    cnt_mids = 0

    count_motifs = 0
    tasks = []
    for mid in steep_inhib_times:
        tasks.append((mid))
        cnt_mids += 1
        if cnt_mids > 100:
            break

    results = pool.map_async(motif_operation, tasks)
    pool.close()
    pool.join()

    motif_data = results.get()

    count_invalid = 0
    frames = []

    for idx in range(len(motif_data)):
        try:
            df_cascade = motif_data[idx]
            frames.append(df_cascade)

        except:
            count_invalid += 1

    print('Invalid: ', count_invalid)
    df_all = pd.concat(frames)

    directory = '../../data/motif_preprocess/v1'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # print(df_all)
    pickle.dump(df_all, open(directory + '/df_optimize_input_sample_v1.0+.pickle', 'wb'))


if __name__ == '__main__':
    main()