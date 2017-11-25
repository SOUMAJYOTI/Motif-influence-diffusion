import apgl
import numpy
from apgl.graph import *
from apgl.generator.KroneckerGenerator import KroneckerGenerator
import numpy as np
import operator


# Inverse transform sampling to generate random values from Rayleigh pdf
def sample_rayleigh_pdf(alphaList = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]):
    alpha = np.random.choice(alphaList)
    U = np.random.uniform(0, 1)

    return 10*np.power(-(2/alpha)*np.log(U), 0.5) # 10 is the scaling factor for minutes as time stamp


def generate_graph():
    initialGraph = SparseGraph(VertexList(5, 1))
    initialGraph.addEdge(1, 2)
    initialGraph.addEdge(2, 3)

    for i in range(5):
        initialGraph.addEdge(i, i)

    k = 2
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
    numTopVertices = 2000
    sortedVertices = sorted(vertDegMap.items(), key=operator.itemgetter(1), reverse=True)[:numTopVertices]
    topVertices = []
    for (v, deg) in sortedVertices:
        topVertices.append(v)

    numCascades = 1000
    T_C = 40000
    cascadeIds = []
    sourceList = []
    targetList = []
    timeStamps = []
    for idx_cas in range(numCascades):
        vertTimeDict = {}
        sourceVert = np.random.choice(topVertices)
        vertTimeDict[sourceVert] = 0
        vertexTraversed = []
        vertexTraversed.append(sourceVert)

        pairsNodes = [(sourceVert, nbr) for nbr in graph.neighbors(sourceVert)]
        vertexCurrLayer = []
        vertexCurrLayer.extend(pairsNodes)

        while True:
            vertexNextLayer = []
            for src, nbr in vertexCurrLayer:
                timeDelta = sample_rayleigh_pdf()
                vertTimeDict[nbr] = vertTimeDict[src] + timeDelta

                # Fill the lists for storing in dataframe
                cascadeIds.append(idx_cas)
                sourceList.append(src)
                targetList.append(nbr)
                timeStamps.append(vertTimeDict[nbr])

                # Append the next layer nodes
                pairsNodes = [(nbr, nbrNew) for nbrNew in graph.neighbors(nbr)]
                vertexNextLayer.extend(pairsNodes)

            vertexCurrLayer[:] = vertexNextLayer  # copy this object


def main():
    generate_graph()

if __name__ == "__main__":
    main()