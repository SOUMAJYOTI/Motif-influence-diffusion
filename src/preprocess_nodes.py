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
    # diff_dict_files = [pickle.load(open('diffusion_dict_v1_t07.pickle', 'rb')),
    #                    pickle.load(open('diffusion_dict_v1_t08.pickle', 'rb'))]

    diff_dict_files = [ pickle.load(open('../data/diffusion_dict_v1_t08.pickle', 'rb'))]

    edge_list = []
    nodes_list =[]
    cnt_file = 1
    for ddf in diff_dict_files:
        print("File: ", cnt_file)
        for node in ddf:
            for nbr, time in ddf[node]:
                if node == nbr:
                    continue
                edge_list.append((node, nbr))
            nodes_list.append(node)
            edge_list = list(set(edge_list))

            # print(len(edge_list))
            if len(edge_list) > 5000000:
                break

        cnt_file += 1

    # print(len(nodes_list), len(edge_list))
    edge_list = list(set(edge_list))
    nodes_list = list(set(nodes_list))

    print("Number of nodes: ", len(nodes_list))
    print("Number of edges: ", len(edge_list))
    cascade_graph = nx.DiGraph()
    cascade_graph.add_edges_from(edge_list)

    print("Computing pagerank...")
    pr_node = nx.pagerank(cascade_graph)
    bw_node = nx.betweenness_centrality(cascade_graph)

    # out_deg = {}
    # in_deg = {}
    # for node in nodes_list:
    #     out_deg[node] = cascade_graph.out_degree(node)
    #     in_deg[node] = cascade_graph.in_degree(node)

    # pagerank = {}
    # for node in pr_:
    #     # print(node, pr_node[node])
    #     pagerank[node] = pr_node[node]
    # # centralities = ( out_deg, in_deg)

    pickle.dump(pr_node, open('pr_diff_T08-v1.pickle', 'wb'))
    pickle.dump(bw_node, open('bw_diff_T08-v1.pickle', 'wb'))






