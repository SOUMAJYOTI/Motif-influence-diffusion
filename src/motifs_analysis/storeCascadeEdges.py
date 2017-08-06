import pandas as pd
import pickle

df_graphs_nw = pickle.load(open('../../data/motif_preprocess/df_graph.pickle', 'rb'))

edges_cascade = {}
edges_historical = {}
for i, r in df_graphs_nw.iterrows():
    mid = r['mid']
    if mid not in edges_cascade:
        edges_cascade[mid] = []
        edges_historical[mid] = []

    src = r['source']
    tgt = r['target']
    if r['edge_type'] == 'cascade':
        edges_cascade[mid].append((src, tgt))
    else:
        edges_historical[mid].append((src, tgt))

pickle.dump((edges_cascade, edges_historical), open('../../data/motif_preprocess/edges_partitioned.pickle', 'wb'))
