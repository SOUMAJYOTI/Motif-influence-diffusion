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
cascade_edges = pickle.load(open(directory + '/df_graph_s3.pickle', 'rb'))
steep_inhib_times = pickle.load(open('../../data/steep_inhib_times.pickle', 'rb'))


def init(args):
    global count_motifs
    count_motifs = args


def checkIsomorphism(graph_list, g):
    for gr in graph_list:
        if gtt.isomorphism(gr, g):
            return gr
    return False


def motif_operation(mid):
    cascade_set = cascade_edges[(cascade_edges['mid'] == str(mid))]

    print("Cascade of mid: ", mid, " and reshare size: ", len(cascade_set))

    numIntervals = np.max(np.array(list(cascade_set['interval_2'])))
    for int_idx in range(1, numIntervals):
        cascade_intervalDf = cascade_set[cascade_set['interval_1'] == int_idx-1]
        cascade_intervalDf = cascade_intervalDf[cascade_intervalDf['interval_2'] == int_idx]
        cascade_edges = cascade_intervalDf[cascade_intervalDf['edge_type']==]

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
        for (src, tgt) in weighted_edges:
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

        # Add the diffusion edges (even if there already exists a cascade edge)
        for (src, tgt) in diff_edges:
            v1 = node_cascades_diff_map[src]
            v2 = node_cascades_diff_map[tgt]

            e = cascade_graph.add_edge(v1, v2)
            cascade_edge_prop[e] = 1

        gts.remove_self_loops(cascade_graph)
        # gts.remove_parallel_edges(cascade_graph)

        # FINDING THE MOTIFS IN THE CASCADE GRAPH + DIFFUSION NETWORK
        motifs_graph, motifs_count, vertex_maps = \
            gt.clustering.motifs(cascade_graph, 3, return_maps=True)

        for g in motifs_graph_filtered:
            for e in g.edges():
                print(cascade_edge_prop[e])

    # DS for storing graph_data
    df_mid = []
    df_source = []
    df_target = []
    df_flag_edge = []
    df_rt_time = []
    df_interval_1 = []
    df_interval_2 = []

    new_nodes = []
    cur_interval = 1
    df_graph_list = []

    for i, r in cascade_set.iterrows():
        src = r['source']
        tgt = r['target']

        # Set the temporal information of the nodes
        rt_time = str(r['rt_time'])
        rt_date = rt_time[:10]
        rt_t = rt_time[11:19]
        record_time = rt_date + ' ' + rt_t
        time_x = datetime.datetime.strptime(record_time, '%Y-%m-%d %H:%M:%S')
        cur_time = time.mktime(time_x.timetuple())

        df_mid.append(mid)
        df_source.append(src)
        df_target.append(tgt)
        df_rt_time.append(cur_time)
        df_flag_edge.append('cascade')
        df_interval_1.append(cur_interval-1)
        df_interval_2.append(cur_interval)

        new_nodes.append(r['source'])
        new_nodes.append(r['target'])

        new_nodes = list(set(new_nodes))

        # each interval should have at most 40 new nodes
        if len(new_nodes) > 40:
            # these are the historical edges
            diff_edges = []
            for v in new_nodes:
                try:
                    for uid, tid in diff_dict[v]:
                        if str(uid) in new_nodes:
                            if (v, uid) not in diff_edges:
                                diff_edges.append((v, uid))

                                df_mid.append(mid)
                                df_source.append(v)
                                df_target.append(uid)
                                df_rt_time.append('')
                                df_flag_edge.append('historical')
                                df_interval_1.append(cur_interval-1) # since this variable cur_interval is already updated before
                                df_interval_2.append(cur_interval)

                except:
                    continue

            new_nodes = []
            cur_interval += 1
            graph_data = {'mid': df_mid, 'source': df_source, 'target': df_target, 'retweet_time': df_rt_time,
                          'interval_1': df_interval_1, 'interval_2:': df_interval_2, 'edge_type': df_flag_edge}
            df_graph_list.append(pd.DataFrame(data=graph_data))
            df_mid = []
            df_source = []
            df_target = []
            df_flag_edge = []
            df_rt_time = []
            df_interval_1 = []
            df_interval_2 = []

    df_graph = pd.concat(df_graph_list)
    return (mid, df_graph)


if __name__ == '__main__':
    global count_motifs
    count_motifs = multiprocessing.Value('i', 0)

    number_intervals = 500

    motif_patterns_global_list = {}
    motif_patterns_global_list_inhib = {}
    motif_count_global_list = {}
    motif_count_global_list_inhib = {}

    motif_patterns_list = []
    dict_patterns = {}
    patterns_count = 1

    numProcessors = 2
    pool = multiprocessing.Pool(numProcessors, initializer=init, initargs=(count_motifs,))

    num_cascades = len(steep_inhib_times.keys())

    print("Loading cascade data...")

    cnt_mids = 0

    # count_motifs = 0
    tasks = []
    for mid in steep_inhib_times:
        tasks.append( (mid) )
        cnt_mids += 1
        # if cnt_mids > 10:
        #     break

    results = pool.map_async(motif_operation, tasks)
    pool.close()
    pool.join()

    motif_data = results.get()

    count_invalid = 0
    frames = []
    motifs_dict = {}
    nodes_map = {}
    for idx in range(len(motif_data)):
        try:
            (mid,  df_graph) = motif_data[
                idx]
            frames.append(df_graph)

        except:
            count_invalid += 1

    print('Invalid: ', count_invalid)
