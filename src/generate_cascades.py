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

    k = 3
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

    numCascades = 10
    T_C = 20000 # Time limit of cascade
    cascadeIds = []
    sourceList = []
    targetList = []
    timeStamps = []

    chosenTopNodes = []

    for idx_cas in range(numCascades):
        print("Cascade of Id: ", idx_cas)
        vertTimeDict = {}

        # Choose a source node not already in other cascades sources
        while True:
            sourceVert = np.random.choice(topVertices)
            if sourceVert in chosenTopNodes:
                continue
            else:
                break
        chosenTopNodes.append(sourceVert)

        vertTimeDict[sourceVert] = 0
        vertexTraversed = []
        vertexTraversed.append(sourceVert)

        pairsNodes = [(sourceVert, nbr) for nbr in graph.neighbours(sourceVert)]
        vertexCurrLayer = []
        vertexCurrLayer.extend(pairsNodes)

        flagEnd = 0
        while True:
            vertexNextLayer = []

            print(len(vertexCurrLayer))
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
                if vertTimeDict[nbr] > T_C:
                    flagEnd = 1
                    break

                # Append the next layer nodes
                pairsNodes = [(nbr, nbrNew) for nbrNew in graph.neighbours(nbr)]
                vertexNextLayer.extend(pairsNodes)

            # Break when the time limit of cascade exceeds or there are  no new nodes to infect
            if flagEnd == 1:
                break
            vertexCurrLayer[:] = vertexNextLayer  # copy this object
            # print(len(vertexCurrLayer))
            if len(vertexCurrLayer) == 0:
                break



def main():
    graph = generate_graph()
    generate_cascades(graph)

if __name__ == "__main__":
    main()