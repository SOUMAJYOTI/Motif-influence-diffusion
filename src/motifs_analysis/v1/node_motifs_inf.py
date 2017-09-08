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


# The stored data to be loaded
motif_patterns_dict = pickle.load(open('../../data/motif_preprocess/motif_patterns_dict.pickle', 'rb')) # motif patterns dictionary
df_graphs_nw = pickle.load(open('../../data/motif_preprocess/df_graph.pickle', 'rb')) # cascade network information in pandas df
motif_maps = pickle.load(open('../../data/motif_preprocess/motifs_maps.pickle', 'rb')) # motif patterns and instances for each cascade
steep_inhib_times = pickle.load(open('../../data/steep_inhib_times.pickle', 'rb')) # steep and inhib times for each cascade
vertex_rtTimes = pickle.load(open('../../data/vertex_rtTimes.pickle', 'wb')) # retweet time of each vertex for each cascade
edges_cascade, edges_historical = pickle.load(open('../../data/motif_preprocess/edges_partitioned.pickle', 'rb')) # partitioned edges precomputed for faster processing


def init(args):
    global count_motifs
    count_motifs = args


def checkIsomorphism(graph_list, g):
    for gr in graph_list:
        if gtt.isomorphism(gr, g):
            return gr
    return False


def get_inf_nodes(mid):
    """ INPUT:
        mid: Cascade message ID
        df_graph: Dataframe containing the edge information for the cascade graph
        mVertex_maps: Motif vertex maps for the cascade

        OUTPUT: Return the influential nodes for each node retweet in the cascade apart from its
                parent node
    """

    df_graph = df_graphs_nw[df_graphs_nw['mid'] == mid]
    [motifs_graph, motifs_count, vertex_maps] = motif_maps[mid]
    edges_cas_curr = edges_cascade[mid]
    edges_hist_curr = edges_historical[mid]

    inf_nodes = {} # stores the influential nodes apart from parent for each node retweet
    inf_edges = {} # stores the influential edges between nodes apart from parent for each node retweet
    for i, r in df_graph.iterrows():
        src = r['source']
        tgt = r['target']
        src_rtTime = vertex_rtTimes[mid][src]
        tgt_rtTime = vertex_rtTimes[mid][tgt]
        rt_time = r['retweet_time']
        edge_type = r['edge_type']
        if edge_type == 'historical': # only consider the cascade retweets for the influence function compute
            continue

        if tgt not in inf_nodes:
            inf_nodes[tgt] = []
            inf_edges[tgt] = []

        # find the motifs of particular patterns attached to that pair of src and tgt
        # Patterns - [M2, M4]

        # Pattern M2
        graph_pat_act = motif_patterns_dict['M2'] # actual pattern
        for idx_map in range(len(motifs_graph)):
            graph_pat_curr = motifs_graph[idx_map]
            # check for the correct pattern among all the patterns
            if not gtt.isomorphism(graph_pat_act, graph_pat_curr):
                continue

            # check the motif instances containing source and target
            vMaps = vertex_maps[idx_map]
            # cnt_maps = 0
            for vertices in vMaps:
                # Cond. 1: the source and target should be in the motif instance
                vertex_list = vertices.a
                if src not in vertex_list or tgt not in vertex_list:
                    continue
                for v in vertex_list:
                    if v != src and v!= tgt:
                        third_vertex = v
                        break

                # Cond. 2: the target vertex should have retweeted last among all the motif vertices
                third_rtTime = vertex_rtTimes[third_vertex]
                max_time = max([tgt_rtTime, src_rtTime, third_rtTime])
                if max_time != tgt_rtTime:
                    continue

                inf_nodes[tgt].append(third_vertex)
                if (third_vertex, src) in edges_cas_curr:
                    inf_edges[tgt].append((third_vertex, src, 'cascade'))
                else:
                    inf_edges[tgt].append((third_vertex, src, 'historical'))

    return (mid, inf_nodes, inf_edges)


if __name__ == '__main__':

    global count_motifs
    motifs_inf = multiprocessing.Value('i', 0)

    numProcessors = 2
    pool = multiprocessing.Pool(numProcessors, initializer=init, initargs=(motifs_inf,))
    cnt_mids = 0

    # count_motifs = 0
    tasks = []
    for mid in steep_inhib_times:
        tasks.append((mid))
        cnt_mids += 1
        if cnt_mids > 1:
            break

    results = pool.map_async(get_inf_nodes, tasks)
    pool.close()
    pool.join()

    motif_data = results.get()

    count_invalid = 0

    for idx in range(len(motif_data)):
        try:
            mid, inf_nodes, inf_edges = motif_data[idx]
            

    #         motifs_graph, motifs_count = motif_data[idx]
    #         for idx_m in range(len(motifs_graph)):
    #             motif_shape = motifs_graph[idx_m]
    #             if not checkIsomorphism(motif_patterns_list, motif_shape):
    #                 motif_patterns_list.append(motif_shape)
    #                 dict_patterns['M' + str(patterns_count)] = motif_shape
    #                 patterns_count += 1
    #
    #     except:
    #         count_invalid += 1
    #
    # print('Invalid: ', count_invalid)
    #
    # pickle.dump(dict_patterns, open('motif_patterns_dict.pickle', 'wb'))
    #
    # for g in dict_patterns:
    #     gr = dict_patterns[g]
    #     pos = gtd.arf_layout(gr)
    #     gtd.graph_draw(gr, pos=pos, output="../../plots/motif_patterns/ " + str(g) + ".pdf")
    #     # gtd.graph_draw(gr, edge_text=cascade_edge_prop, edge_font_size=30, edge_text_distance=20, edge_marker_size=40,
    #     #            output="output.png")
    #
    #
