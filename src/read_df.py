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
dataDf = pickle.load(open(directory + '/df_graph_s3.pickle', 'rb'))
steep_inhib_times = pickle.load(open('../../data/steep_inhib_times.pickle', 'rb'))
# motif_patterns_dict = pickle.load()

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

    # print(cascade_set)

    print("Cascade of mid: ", mid, " and reshare size: ", len(cascade_set))

    numIntervals = np.max(np.array(list(cascade_set['interval_2:'])))
    last_time = 0
    inf_nodes = {}  # stores the influential nodes apart from parent for each node retweet
    inf_edges = {}  # stores the influential edges between nodes apart from parent for each node retweet
    motif_graph_patterns = [[] for _ in range(500)]
    for int_idx in range(1, numIntervals):
        cascade_intervalDf_prev= cascade_set[cascade_set['interval_1'] == int_idx-1]
        cascade_intervalDf_curr = cascade_set[cascade_set['interval_1'] == int_idx]
        cascade_intervalDf = pd.concat([cascade_intervalDf_prev, cascade_intervalDf_curr])

        cascade_Df = cascade_intervalDf[cascade_intervalDf['edge_type']=='cascade']
        historical_Df = cascade_intervalDf[cascade_intervalDf['edge_type']=='historical']

        # Create the vertex time dictionary
        vertex_rtTime_dict = {}
        for i, r in cascade_intervalDf.iterrows():
            if r['edge_type'] == 'historical':
                continue

            src = r['source']
            tgt = r['target']
            vertex_rtTime_dict[tgt] = r['retweet_time']
            if src not in vertex_rtTime_dict:
                vertex_rtTime_dict[src] = last_time

            last_time = r['retweet_time']

        # Store the cascade edges
        edges_cascade = []
        edges_historical = []
        for i, r in cascade_intervalDf.iterrows():
            src = r['source']
            tgt = r['target']

            if r['edge_type'] == 'cascade':
                edges_cascade.append((src, tgt))
            else:
                edges_historical.append((src, tgt))

        # create the graph from the edge list
        cascade_graph = gt.Graph(directed=True)
        node_cascades_diff_map = {}
        cnt_nodes = 0
        cascade_vertices = cascade_graph.new_vertex_property("string")
        cascade_edge_prop = cascade_graph.new_edge_property("int")
        cascade_map_write_file = {}

        # Add the cascade edges
        # 0 - Cascade edges
        # 1 - Diffusion edges

        for i, r in cascade_Df.iterrows():
            src = r['source']
            tgt = r['target']
            if src not in node_cascades_diff_map:
                node_cascades_diff_map[src] = cnt_nodes # map from user ID to graphnode ID
                v1 = cascade_graph.add_vertex()
                cascade_vertices[v1] = src # map from graphnode ID to user ID
                cascade_map_write_file[cnt_nodes] = src
                cnt_nodes += 1
            else:
                v1 = cascade_graph.vertex(node_cascades_diff_map[src])

            if tgt not in node_cascades_diff_map:
                node_cascades_diff_map[tgt] = cnt_nodes # map from user ID to graphnode ID
                v2 = cascade_graph.add_vertex()
                cascade_vertices[v2] = tgt # map from graphnode ID to user ID
                cascade_map_write_file[cnt_nodes] = tgt
                cnt_nodes += 1
            else:
                v2 = cascade_graph.vertex(node_cascades_diff_map[tgt])

            if cascade_graph.edge(v1, v2):
                continue
            else:
                e = cascade_graph.add_edge(v1, v2)
                cascade_edge_prop[e] = 0

        # Add the diffusion edges (even if there already exists a cascade edge, but only once)
        edges_seen = []
        for i, r in historical_Df.iterrows():
            src = r['source']
            tgt = r['target']
            v1 = node_cascades_diff_map[src]
            v2 = node_cascades_diff_map[tgt]

            if (v1, v2) in edges_seen:
                continue

            edges_seen.append((v1, v2))
            e = cascade_graph.add_edge(v1, v2)
            cascade_edge_prop[e] = 1

        gts.remove_self_loops(cascade_graph)
        # gts.remove_parallel_edges(cascade_graph)

        # FINDING THE MOTIFS IN THE CASCADE GRAPH + DIFFUSION NETWORK
        motifs_graph, motifs_count, vertex_maps = \
            gt.clustering.motifs(cascade_graph, 3, return_maps=True)

        # Store the motif patterns interval wise for retrieval later
        for idx_pat in range(len(motifs_graph)):
            motif_graph_patterns[int_idx-1].append(motifs_graph[idx_pat])

        # Find the influential nodes for each node in the retweet cascade for the current interval only - not the previous interval
        for i, r in cascade_intervalDf_curr.iterrows():
            src = r['source']
            tgt = r['target']
            if tgt in inf_nodes:
                continue

            src_vert = node_cascades_diff_map[src]
            tgt_vert = node_cascades_diff_map[tgt]
            src_rtTime = vertex_rtTime_dict[mid][src]
            tgt_rtTime = vertex_rtTime_dict[mid][tgt]
            # rt_time = r['retweet_time']
            edge_type = r['edge_type']
            if edge_type == 'historical':  # only consider the cascade retweets for the influence function compute
                continue

            if tgt not in inf_nodes:
                inf_nodes[tgt] = []
                inf_edges[tgt] = []

            # find the motifs of particular patterns attached to that pair of src and tgt
            # Patterns - [M2, M3]

            # Pattern M2
            # This is just for pattern M2, the likelihood function changes for each pattern.
            # Everything else remains the same. This is just for the training model.
            graph_pat_act = motif_patterns_dict['M1']  # actual pattern
            print(len(motifs_graph), len(vertex_maps))
            for idx_map in range(len(motifs_graph)):
                graph_pat_curr = motifs_graph[idx_map]
                # check for the correct pattern among all the patterns
                # if not gtt.isomorphism(graph_pat_act, graph_pat_curr):
                #     continue

                for M in motif_patterns_dict:
                    if gtt.isomorphism(motif_patterns_dict[M], graph_pat_curr):
                        print(M, motifs_count[idx_map])

                # return

                # check the motif instances containing source and target
                vMaps = vertex_maps[idx_map]
                # print(len(vMaps))
                cnt_maps = 0
                for vertices in vMaps:
                    # Cond. 1: the source and target should be in the motif instance
                    vertex_list = list(vertices.a)
                    if src_vert not in vertex_list or tgt_vert not in vertex_list:
                        continue
                    for v in vertex_list:
                        if v != src_vert and v != tgt_vert:
                            third_vertex = cascade_vertices[v]
                            break

                    # Cond. 2: the target vertex should have retweeted last among all the motif vertices
                    third_rtTime = vertex_rtTime_dict[third_vertex]
                    max_time = max([tgt_rtTime, src_rtTime, third_rtTime])
                    if max_time != tgt_rtTime:
                        continue

                    # For different motif patterns, need to check different types of motif edges - this is difficult !!

                    # Assertion 1: There must be an edge from source to target vertex

                    inf_nodes[tgt].append(third_vertex)
                    if (third_vertex, tgt) in edges_cas_curr:
                        inf_edges[tgt].append((third_vertex, tgt, 'cascade'))
                    elif (third_vertex, tgt) in edges_hist_curr:
                        inf_edges[tgt].append((third_vertex, tgt, 'historical'))

    return motif_graph_patterns[:numIntervals-1] #(mid, df_graph)


def main():
    global count_motifs
    count_motifs = multiprocessing.Value('i', 0)

    number_intervals = 500

    motif_patterns_list = []
    dict_patterns = {}
    patterns_count = 1

    numProcessors = 4
    pool = multiprocessing.Pool(numProcessors, initializer=init, initargs=(count_motifs,))

    num_cascades = len(steep_inhib_times.keys())

    print("Loading cascade data...")

    cnt_mids = 0

    count_motifs = 0
    tasks = []
    for mid in steep_inhib_times:
        tasks.append((mid))
        cnt_mids += 1
        if cnt_mids > 50:
            break

    results = pool.map_async(motif_operation, tasks)
    pool.close()
    pool.join()

    motif_data = results.get()

    count_invalid = 0
    frames = []
    motifs_dict = {}
    nodes_map = {}
    count = 0
    for idx in range(len(motif_data)):
        print('Count cacsade: ', count)
        try:
            count += 1
            motif_graph_data = motif_data[idx]

            for interval in range(len(motif_graph_data)):
                for m in motif_graph_data[interval]:
                    pat = checkIsomorphism(motif_patterns_list, m)
                    if pat == False:
                        motif_patterns_list.append(m)
                        dict_patterns['M' + str(count_motifs)] = m
                        count_motifs += 1

            # (mid, df_graph) = motif_data[
            #     idx]
            # frames.append(df_graph)

        except:
            count_invalid += 1

    print('Invalid: ', count_invalid)
    print(len(dict_patterns))
    pickle.dump(dict_patterns, open('motifs_pat.pickle', 'wb'))


if __name__ == '__main__':
    main()