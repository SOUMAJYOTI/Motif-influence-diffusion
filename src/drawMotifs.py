import graph_tool as gt
import graph_tool.stats as gts
import graph_tool.util as gtu
import graph_tool.draw as gtd
import pickle

directory = '../../data/motif_preprocess/v1/'
dict_patterns = pickle.load(open(directory + 'motifs_pat.pickle', 'rb')) # motif patterns dictionary
directory_plots = '../../plots/motif_patterns/3'

# print(dict_patterns)
for g in dict_patterns:
    gr = dict_patterns[g]
    pos = gtd.arf_layout(gr)
    gtd.graph_draw(gr, pos=pos, output=directory_plots + "/" + str(g) + ".pdf")