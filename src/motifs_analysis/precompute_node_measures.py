import networkx as nx
import pickle
import  matplotlib.pyplot as plt


def binarySearch(alist, item):
    first = 0
    last = len(alist)-1
    found = False

    while first<=last and not found:
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1

    return found

if __name__ == "__main__":
    diff_dict_files = [pickle.load(open('../../data/diffusion_dict_v1_t07.pickle', 'rb')),
                       pickle.load(open('../../data/diffusion_dict_v1_t08.pickle', 'rb'))]

    edge_list = []
    nodes_list =[]
    cnt_file = 1
    for ddf in diff_dict_files:
        print("File: ", cnt_file)
        for node in ddf:
            if not binarySearch(nodes_list, node):
                nodes_list.append(node)
            for nbr, time in ddf[node]:
                if not binarySearch(edge_list, (node, nbr)):
                    if node == nbr:
                        continue
                    edge_list.append((node, nbr))

        cnt_file += 1
        # if len(edge_list) > 100:
        #     break

    # print(len(nodes_list), len(edge_list))
    cascade_graph = nx.DiGraph()
    cascade_graph.add_edges_from(edge_list)

    pr_node = nx.pagerank(cascade_graph)
    bw_node = nx.betweenness_centrality(cascade_graph)

    out_deg = {}
    in_deg = {}
    for node in pr_node:
        out_deg[node] = cascade_graph.out_degree(node)
        in_deg[node] = cascade_graph.in_degree(node)

    centralities = (pr_node, bw_node, out_deg, in_deg)

    pickle.dump(centralities, open('../../data/centralities_diff_T07_08-v1.pcikle', 'wb'))






