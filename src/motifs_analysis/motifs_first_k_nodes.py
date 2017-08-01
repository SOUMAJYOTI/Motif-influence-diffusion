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

# form the motifs from the first k nodes where $k$ is the parameter
diff_dict = pickle.load(open('../../data/diffusion_dict_v1_t07.pickle', 'rb'))

print('Loading diffusion file...')
diff_file = '../../data/rt_df.csv'
df = pd.read_csv(diff_file, names=['index', 'target', 'source', 'rt_time', 'mid', 'post_time', 'isOn'])
df['mid'] = df['mid'].astype('str')
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
    cascade_set = df[(df['mid'] == str(mid))]

    nodes_src = [[] for interval in range(500)]
    new_nodes = []
    new_nodes_count = 0

    weighted_edges = []
    cnt_intervals = 0
    time_post_individual = {}

    motif_patterns_cascade_list = [{} for i in range(500)]
    print("Cascade of mid: ", mid, " and reshare size: ", len(cascade_set))

    start_time_cascade = 0
    last_time = 0
    count_duplicate = 0
    for i, r in cascade_set.iterrows():
        # cnt_intervals += 1
        # new_nodes = []
        # new_nodes_count = 0

        src = r['source']
        tgt = r['target']

        # Set the temporal information of the nodes
        rt_time = str(r['rt_time'])
        rt_date = rt_time[:10]
        rt_t = rt_time[11:19]
        record_time = rt_date + ' ' + rt_t
        time_x = datetime.datetime.strptime(record_time, '%Y-%m-%d %H:%M:%S')
        cur_time = time.mktime(time_x.timetuple())

        if i == 0:
            start_time_cascade = cur_time # this is the start time of the cascade
            time_post_individual[src] = cur_time - 100 # set the start time of src node as some time before this post

        if (src, tgt) in weighted_edges:
            continue

        if src not in time_post_individual:
            time_post_individual[src] = last_time

        # this check is not needed - but just for noisy data !!!
        if tgt not in time_post_individual:
            time_post_individual[tgt] = cur_time - start_time_cascade

        last_time = cur_time  # store this time for next retweet if the source node is new

        # Set the edge information of the users
        weighted_edges.append((src, tgt))

        new_nodes.append(r['source'])
        new_nodes.append(r['target'])

        new_nodes = list(set(new_nodes))
        new_nodes_count = len(new_nodes)
        nodes_src[cnt_intervals] = list(set(nodes_src[cnt_intervals]))

        # If the number of observed nodes (used for the learning algorithm) crosses a threshold = 50
        if new_nodes_count > 50:
            # these are the historical edges
            diff_edges = []
            for v in new_nodes:
                try:
                    for uid, tid in diff_dict[v]:
                        if str(uid) in new_nodes:
                            if (v, uid) not in diff_edges:
                                diff_edges.append((v, uid))
                                # if len(list(set(diff_edges_cur_interval))) > 2*len_tree_edges:
                                #     break
                except:
                    continue

            # create the graph from the edge list
            cascade_graph = gt.Graph(directed=True)
            node_cascades_diff_map = {}
            cnt_nodes = 0
            cascade_vertices = cascade_graph.new_vertex_property("string")
            cascade_edge_prop = cascade_graph.new_edge_property("int")

            # Add the cascade edges
            # 0 - Cascade edges
            # 1 - Diffusion edges
            for (src, tgt) in weighted_edges:
                if src not in node_cascades_diff_map:
                    node_cascades_diff_map[src] = cnt_nodes # map from user ID to graphnode ID
                    v1 = cascade_graph.add_vertex()
                    cascade_vertices[v1] = src # map from graphnode ID to user ID
                    cnt_nodes += 1
                else:
                    v1 = cascade_graph.vertex(node_cascades_diff_map[src])

                if tgt not in node_cascades_diff_map:
                    node_cascades_diff_map[tgt] = cnt_nodes # map from user ID to graphnode ID
                    v2 = cascade_graph.add_vertex()
                    cascade_vertices[v2] = tgt # map from graphnode ID to user ID
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
            motifs_graph_filtered, motifs_count_filtered, vertex_maps_filtered = \
                gt.clustering.motifs(cascade_graph, 3, return_maps=True)

            # for g in motifs_graph_filtered:
            #     for e in g.edges():
            #         print(cascade_edge_prop[e])

            return motifs_graph_filtered, motifs_count_filtered


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
        if cnt_mids > 500:
            break

    results = pool.map_async(motif_operation, tasks)
    pool.close()
    pool.join()

    motif_data = results.get()

    count_invalid = 0
    for idx in range(len(motif_data)):
        try:
            motifs_graph, motifs_count = motif_data[idx]
            for idx_m in range(len(motifs_graph)):
                motif_shape = motifs_graph[idx_m]
                if not checkIsomorphism(motif_patterns_list, motif_shape):
                    motif_patterns_list.append(motif_shape)
                    dict_patterns['M' + str(patterns_count)] = motif_shape
                    patterns_count += 1

        except:
            count_invalid += 1

    print('Invalid: ', count_invalid)

    pickle.dump(dict_patterns, open('motif_patterns_dict.pickle', 'wb'))

    for g in dict_patterns:
        gr = dict_patterns[g]
        pos = gtd.arf_layout(gr)
        gtd.graph_draw(gr, pos=pos, output="../../plots/motif_patterns/ " + str(g) + ".pdf")
        # gtd.graph_draw(gr, edge_text=cascade_edge_prop, edge_font_size=30, edge_text_distance=20, edge_marker_size=40,
        #            output="output.png")


