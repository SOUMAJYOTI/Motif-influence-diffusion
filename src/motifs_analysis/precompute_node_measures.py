import networkx as nx
import pickle


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
    for ddf in diff_dict_files:
        for node in ddf:
            if not binarySearch(nodes_list, node):
                nodes_list.append(node)
            for nbr, time in ddf[node]:
                if not binarySearch(edge_list, (node, nbr)):
                    if node == nbr:
                        continue
                    edge_list.append((node, nbr))

            if len(edge_list) > 100:
                break

    cascade_graph = nx.DiGraph()
    cascade_graph.add_edges_from(edge_list)

    measure_node = nx.betweenness_centrality(cascade_graph)






