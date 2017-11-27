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

# Load the global data files to be used for each thread in the function motif_operation()
directory = '../../data/motif_preprocess/v1'
dataDf = pickle.load(open(directory + '/df_graph_s3.pickle', 'rb'))
steep_inhib_times = pickle.load(open('../../data/steep_inhib_times.pickle', 'rb'))
motif_patterns_dict = pickle.load(open(directory + '/motifs_pat.pickle', 'rb'))

def init(args):
    global count_motifs
    count_motifs = args


def checkIsomorphism(graph_list, g):
    for gr in graph_list:
        if gtt.isomorphism(gr, g):
            return gr
    return False


def motif_operation(mid, cnt_mid):
    #
    ''' The steps of computing the exposure nodes are as follows:
        1. For each pair of intervals of a cascde [I-1, I] for I \in [1, I_C], form the cascade + historical network N_I.
        2. Compute the network motifs of size 3 exhibited in N_I.
        3. For each node $u$ activated in interval I, see whether $u$ was a part of any of
           the instances belonging to the 6 motif patterns mentioned in the paper such that $u$ had participated in the
           those instances after the other two nodes, one of which must be the source
        4. Add the third node in those selected instances barring the parent and $u$ (which together form the 3-motif instance)
           in the exposure set of $u$ for the cascade C.
        5. This step imposes the AND gating constraint to remove nodes that violate the boolean TRUE threshold constraint.
    '''
    # 1.
    cascade_set = dataDf[dataDf['mid'] == mid]

    print("Cascade : ", cnt_mid, " and reshare size: ", len(cascade_set))

    numIntervals = np.max(np.array(list(cascade_set['interval_2:'])))
    last_time = 0
    inf_nodes = {}  # stores the influential nodes apart from parent for each node retweet
    inf_edges = {}  # stores the influential edges between nodes apart from parent for each node retweet
    motif_graph_patterns = [[] for _ in range(500)]
    for int_idx in range(1, numIntervals):
        # print("Interval: ", int_idx)
        ''' Operation 1. '''
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

        ''' Operation 2. '''
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

        gts.remove_parallel_edges(cascade_graph)

        # Add the historical diffusion edges (even if there already exists a cascade edge, but only once)
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


        '''' Operation 3. '''
        # FINDING THE MOTIFS IN THE CASCADE GRAPH + DIFFUSION NETWORK - SIZE 3
        motifs_graph, motifs_count, vertex_maps = \
            gt.clustering.motifs(cascade_graph, 3, return_maps=True)

        # Store the motif patterns interval wise for retrieval later
        for idx_pat in range(len(motifs_graph)):
            motif_graph_patterns[int_idx-1].append(motifs_graph[idx_pat])


        '''' Operation 4. '''
        # Find the influential nodes for each node in the retweet cascade for the current interval only - NOT the previous interval
        for i, r in cascade_intervalDf_curr.iterrows():
            src = r['source']
            tgt = r['target']
            if tgt in inf_nodes:
                continue

            # extract the graph node IDs
            src_vert = node_cascades_diff_map[src]
            tgt_vert = node_cascades_diff_map[tgt]
            # extract the vertex timestamps
            src_rtTime = vertex_rtTime_dict[src]
            tgt_rtTime = vertex_rtTime_dict[tgt]

            # only consider the cascade retweets for (src, tgt) pair
            edge_type = r['edge_type']
            if edge_type == 'historical':
                continue

            # find the motifs of particular patterns attached to that pair of src and tgt
            # Patterns - [M4, M7, M16, M17, M23, M25]

            # Patterns handcoded - can be automated into lists or arrays for more efficiency
            graph_pat_act_M4 = motif_patterns_dict['M4']
            graph_pat_act_M7 = motif_patterns_dict['M7']
            graph_pat_act_M16 = motif_patterns_dict['M16']
            graph_pat_act_M23 = motif_patterns_dict['M23']
            graph_pat_act_M25 = motif_patterns_dict['M25']
            graph_pat_act_M31 = motif_patterns_dict['M31']


            # Extract the motif instances belonging to this pattern
            for idx_map in range(len(motifs_graph)):
                graph_pat_curr = motifs_graph[idx_map]
                # check if the instance belongs to any of these patterns
                if (not gtt.isomorphism(graph_pat_act_M4, graph_pat_curr) ) and (not gtt.isomorphism(graph_pat_act_M7, graph_pat_curr) ) \
                    and (not gtt.isomorphism(graph_pat_act_M16, graph_pat_curr) ) and (not gtt.isomorphism(graph_pat_act_M23, graph_pat_curr) ) \
                        and (not gtt.isomorphism(graph_pat_act_M25, graph_pat_curr)) and (not gtt.isomorphism(graph_pat_act_M31, graph_pat_curr) ):
                    continue

                # for M in motif_patterns_dict:
                #     if gtt.isomorphism(motif_patterns_dict[M], graph_pat_curr):
                #         print(M, motifs_count[idx_map])

                # return

                # 1st constraint: Traverse through all the motif instances of this pattern that only contain the (src, tgt) cascade edge
                vMaps = vertex_maps[idx_map]
                # print(len(vMaps))
                cnt_maps = 0
                for vertices in vMaps:
                    # print('hello....')
                    # Cond. 1: the source and target should be in the motif instance
                    vertex_list = list(vertices.a)
                    if src_vert not in vertex_list or tgt_vert not in vertex_list:
                        continue
                    # print('hello1')

                    # Find the non-source and non-target vertex
                    for v in vertex_list:
                        if v != src_vert and v != tgt_vert:
                            third_vertex = cascade_vertices[v] # this is the potential non-parent exposure node to target node
                            break

                    # print('hello2')
                    # Cond. 2: the target vertex should have retweeted last among all the motif vertices
                    third_rtTime = vertex_rtTime_dict[third_vertex]
                    max_time = max([tgt_rtTime, src_rtTime, third_rtTime])
                    if max_time != tgt_rtTime:
                        continue

                    # For different motif patterns, need to check different types of motif edges - this is difficult !!

                    # print(tgt, third_vertex)
                    if tgt not in inf_nodes:
                        inf_nodes[tgt] = []
                        inf_edges[tgt] = []

                    inf_nodes[tgt].append(third_vertex)
                    inf_nodes[tgt] = list(set(inf_nodes[tgt]))
                    # if (third_vertex, tgt) in edges_cas_curr:
                    #     inf_edges[tgt].append((third_vertex, tgt, 'cascade'))
                    # elif (third_vertex, tgt) in edges_hist_curr:
                    #     inf_edges[tgt].append((third_vertex, tgt, 'historical'))

        ''' Operation 5. '''

    #  Create a dataframe with the rows as original
    cascade_Df = cascade_set[cascade_set['edge_type'] == 'cascade']
    inf_nodes_df = []
    count_row = 0

    for idx, row in cascade_Df.iterrows():
        tgt = row['target']

        if tgt not in inf_nodes:
            inf_nodes_df.append([])
        else:
            inf_nodes_df.append(inf_nodes[tgt])

        count_row += 1

    cascade_Df['exposureNodes'] = inf_nodes_df

    return cascade_Df


def main():
    global count_motifs
    count_motifs = multiprocessing.Value('i', 0)

    numProcessors = 5
    pool = multiprocessing.Pool(numProcessors, initializer=init, initargs=(count_motifs,))

    num_cascades = len(steep_inhib_times.keys())

    print("Loading cascade data...")

    cnt_mids = 0

    count_motifs = 0
    tasks = []
    for mid in steep_inhib_times:
        tasks.append((mid, cnt_mids))
        cnt_mids += 1
        if cnt_mids > 500:
            break

    results = pool.starmap_async(motif_operation, tasks)
    pool.close()
    pool.join()

    motif_data = results.get()

    count_invalid = 0
    frames = []

    count = 0
    for idx in range(len(motif_data)):
        print('Count cascade: ', count)
        try:
            count += 1
            cascade_df = motif_data[idx]

            frames.append(cascade_df)

        except:
            count_invalid += 1

    df_all = pd.concat(frames)
    pickle.dump(df_all, open('../../data/motif_preprocess/v1/exposureNodes_DF.pickle', 'wb'))
    # print('Invalid: ', count_invalid)
    # print(len(dict_patterns))
    # pickle.dump(dict_patterns, open('motifs_pat.pickle', 'wb'))


if __name__ == '__main__':
    main()