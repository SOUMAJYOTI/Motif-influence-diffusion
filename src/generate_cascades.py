import apgl
import numpy
from apgl.graph import *
from apgl.generator.KroneckerGenerator import KroneckerGenerator
import numpy as np
import operator
import pandas as pd
import pickle

# Inverse transform sampling to generate random values from Rayleigh pdf
def sample_rayleigh_pdf(alphaList = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]):
    alpha = np.random.choice(alphaList)
    U = np.random.uniform(0, 1)

    return 100*np.power(-(2/alpha)*np.log(U), 0.5) # 10 is the scaling factor for minutes as time stamp


def generate_graph():
    numVertcies = 6
    initialGraph = SparseGraph(VertexList(numVertcies, 1))
    for idx in range(initialGraph.getNumVertices()-1):
        initialGraph.addEdge(idx, idx+1)
    # initialGraph.addEdge(1, 2)
    # initialGraph.addEdge(2, 3)

    for i in range(numVertcies):
        initialGraph.addEdge(i, i)

    k = 5
    generator = KroneckerGenerator(initialGraph, k)
    graph = generator.generate()

    # directedEdges = graph.getAllDirEdges()
    return graph


def generate_cascades(graph):
    degList = graph.degreeSequence()
    numVertices = graph.getNumVertices()

    vertDegMap = {}

    for idx_vert in range(numVertices):
        vertDegMap[idx_vert] = degList[idx_vert]

    # Get the top k vertices by degree
    numTopVertices = 20
    sortedVertices = sorted(vertDegMap.items(), key=operator.itemgetter(1), reverse=True)[:numTopVertices]
    topVertices = []
    for (v, deg) in sortedVertices:
        topVertices.append(v)

    numCascades = 5000
    T_C = 20000 # Time limit of cascade
    cascadeIds = []
    sourceList = []
    targetList = []
    timeStamps = []

    chosenTopNodes = []
    maxNumLayers = 10

    for idx_cas in range(numCascades):
        print("Cascade of Id: ", idx_cas)
        vertTimeDict = {}

        # Choose a source node not already in other cascades sources
        while True:
            sourceVert = np.random.choice(range(numVertices))
            if sourceVert in chosenTopNodes:
                continue
            else:
                break
        chosenTopNodes.append(sourceVert)

        vertTimeDict[sourceVert] = 0
        vertexTraversed = []
        vertexTraversed.append(sourceVert)

        pairsNodes = list(set([(sourceVert, nbr) for nbr in graph.neighbours(sourceVert)]))
        vertexCurrLayer = []
        vertexCurrLayer.extend(pairsNodes)

        flagEnd = 0
        layer = 0
        while True:
            vertexNextLayer = []

            # print(len(vertexCurrLayer))
            if layer > maxNumLayers:
                break
            for src, nbr in vertexCurrLayer:
                if nbr in vertexTraversed:
                    continue
                timeDelta = sample_rayleigh_pdf()
                vertTimeDict[nbr] = vertTimeDict[src] + timeDelta
                vertexTraversed.append(nbr)

                # Fill the lists for storing in dataframe
                cascadeIds.append(idx_cas)
                sourceList.append(src)
                targetList.append(nbr)
                timeStamps.append(vertTimeDict[nbr])

                # print(vertTimeDict[nbr])

                # print(vertTimeDict[nbr])
                if vertTimeDict[nbr] > T_C:
                    flagEnd = 1
                    break

                # Append the next layer nodes
                pairsNodes = [(nbr, nbrNew) for nbrNew in graph.neighbours(nbr)]
                vertexNextLayer.extend(pairsNodes)

            # Break when the time limit of cascade exceeds or there are  no new nodes to infect
            if flagEnd == 1:
                break
            vertexCurrLayer[:] = list(set(vertexNextLayer)) # copy this object

            # print(len(vertexCurrLayer))
            if layer > maxNumLayers:
                break
            randomElem = np.random.choice(range(len(vertexCurrLayer)), min(np.power(2, maxNumLayers-layer), len(vertexCurrLayer)))
            vertexLayerFilter = []
            for idx_rand in range(len(randomElem)):
                vertexLayerFilter.append(vertexCurrLayer[randomElem[idx_rand]])

            # print(len(vertexCurrLayer))

            vertexCurrLayer[:] = vertexLayerFilter
            if len(vertexCurrLayer) == 0:
                break

            if len(sourceList) > 2000:
                break
            layer += 1

    df_synthetic = pd.DataFrame()
    df_synthetic['mid'] = cascadeIds
    df_synthetic['source'] = sourceList
    df_synthetic['target'] = targetList
    df_synthetic['rtTime'] = timeStamps

    # print(df_synthetic)
    return df_synthetic


def main():
    graph = generate_graph()
    # print(graph.getNumVertices())
    df_simulated = generate_cascades(graph)

    pickle.dump(df_simulated, open('../../data/df_simulated_diffusion.pickle', 'wb'))

if __name__ == "__main__":
    main()