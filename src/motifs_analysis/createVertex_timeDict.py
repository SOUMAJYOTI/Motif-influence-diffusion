import pandas as pd
import pickle

df_graphs_nw = pickle.load(open('../../data/motif_preprocess/df_graph.pickle', 'rb'))

vertex_rtTime_dict = {}
for i, r in df_graphs_nw.iterrows():
    if r['edge_type'] == 'historical':
        continue
    mid = r['mid']
    if mid not in vertex_rtTime_dict:
        vertex_rtTime_dict[mid] = {}

    src = r['source']
    tgt = r['target']
    vertex_rtTime_dict[mid][tgt] = r['retweet_time']

pickle.dump(vertex_rtTime_dict, open('../../data/motif_preprocess/vertex_rtTimes.pickle', 'wb'))
