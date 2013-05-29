

import sys
import networkx as nx 
import numpy as np
from groupMetrics import *

from collections import defaultdict 

class dataObject():
    """ helper class for parseGraph() data storage."""
    numNodes = 0
    numEdges = 0
    averageShortestPath = 0
    averageShortestPathWeighted = 0
    maxShortestPath = 0
    maxShortestPathWeighted = 0
    groupBetweenness = defaultdict(list)
    groupCloseness = defaultdict(list)
    def printData(self):
        print "================================================"
        print "This graph has: "
        print self.numNodes, "nodes"
        print self.numEdges, "links"
        print self.averageShortestPathWeighted, "is", \
                "the average shortest path lenght (Weighted)"
        print self.maxShortestPathWeighted, "is the",\
                "Max shortest path lenght (Weighted)"
        print self.averageShortestPath, "is the average",\
                "shortest path lenght"
        print self.maxShortestPath, "is the Max shortest",\
                "path lenght"
        for gSize in self.groupBetweenness:
            print "best Betweenness for group size", gSize, "is",\
                self.groupBetweenness[gSize][0], "for groups:"
            print "  ",
            for g in self.groupBetweenness[gSize][1]:
                print g,
            print ""
        for gSize in self.groupCloseness:
            print "best average closeness for group size", gSize, "is",\
                self.groupCloseness[gSize][0], "for groups:"
            print "   ",
            for g in self.groupCloseness[gSize][1]:
                print g,
            print ""
        print "================================================"

            


def extractData(G):
    """ extract wanted data from the graph. """
    results = dataObject()
    results.numEdges = len(G.edges())
    results.numNodes = len(G.nodes())
    pathLenghts = nx.all_pairs_dijkstra_path_length(G, 
            weight="weight").values()
    results.averageShortestPathWeighted = np.average(
            [ x.values()[0] for x in pathLenghts])
    results.maxShortestPathWeighted = np.max(
            [ x.values()[0] for x in pathLenghts])
    pathLenghts = nx.all_pairs_shortest_path_length(G).values()
    results.averageShortestPath = np.average(
            [ x.values()[0] for x in pathLenghts])
    results.maxShortestPath = np.max(
            [ x.values()[0] for x in pathLenghts])
    cache = None
    for i in range(1,4):
        res = computeGroupMetrics(G, groupSize=i, weighted=True, 
            cutoff = 2, shortestPathsCache=cache)
        cache = res[-1]
        results.groupBetweenness[i] = [res[0], res[1]]
        results.groupCloseness[i] = [res[2], res[3]]
    results.printData()


