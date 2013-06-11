

import sys
import code 
import scipy

import networkx as nx 
import numpy as np

from groupMetrics import *
from miscLibs import *

from collections import defaultdict 
from pylab import *
import matplotlib.pyplot  as pyplot

class dataObject():
    """ helper class for parseGraph() data storage."""
    numNodes = 0
    numEdges = 0
    averageShortestPath = 0
    averageShortestPathWeighted = 0
    maxShortestPath = 0
    maxShortestPathWeighted = 0
    groupMetrics = {}
    etxStats = {}
    mpr = {}
    HNA = {}
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
        for i in self.mpr:
            print i, self.mpr[i]
        print "================================================"

    def printTable(self):
        print "############ "
        print "############ begin tabular data"
        space = 12
        for l in self.etxStats:
            print "etx".ljust(space),
            for k in self.etxStats[l]:
                print k.ljust(space),
            print ""
            break

        for etx in sorted(self.etxStats):
            try:
                label = float(etx)
            except:
                continue
            print  str(label).ljust(space),
            for key in self.etxStats[l]:
                print str(self.etxStats[label][key])[0:5].ljust(space),
            print ""

        if self.groupMetrics != {}:
            print ""
            print ""
            print "groupSize".ljust(space),
            for k in self.groupMetrics:
                print str(k).ljust(space),
            print ""
            for k,v in self.groupMetrics['betweenness'].items():
                print str(k).ljust(space), str(v[0]).ljust(space), \
                        str(self.groupMetrics['closeness'][k][0]).ljust(space)
                print "# most in between groups", v[1]
                print "# closest groups", self.groupMetrics['closeness'][k][1]

        if self.HNA != {}:
            print ""
            print ""
            print "groupSize".ljust(space), "HNAbetweenness".ljust(space)
            for k,v in self.HNA['betweenness'].items():
                print str(k).ljust(space),  str(v['betweenness']).ljust(space)
                print "# most in between group, ", v['group']
            print ""
            print ""
            print "pathLength".ljust(space), "CCDF".ljust(space)
            CCDF = self.HNA['weighted']['ccdf']
            for k in sorted(CCDF):
                print str(k).ljust(space), str(CCDF[k]).ljust(space)
        print "############ end tabular data"
        print "############ "


def purgeGraph(graph, maxWeight):
    toBeRemoved = []
    purgedGraph = graph.copy()
    for s,d,data in purgedGraph.edges(data=True):
        if data['weight'] > maxWeight:
            toBeRemoved.append([s,d])
    for e in toBeRemoved:
        purgedGraph.remove_edge(e[0], e[1])
    return nx.connected_component_subgraphs(purgedGraph)[0]

def compareShortestPaths(graph, purgedGraph, results):
    """ compare shortest paths with and without weights."""

    matched = 0
    totRoutes = 0
    weigthStats = []
    for source in graph.nodes():
            for target in graph.nodes():
                if target != source:
                    
                    
                    shortestPathWeigth = nx.shortest_path_length(graph,
                        source, target, weight="weight")
                    # we are not interestes in 1-hop paths
                    if shortestPathWeigth < 2:
                        continue
                    
                    shortestPath = nx.shortest_path(graph,
                        source, target, weight="weight")

                    totRoutes += 1

                    shortestPathsGen = nx.all_shortest_paths(purgedGraph,
                        source, target)
                    nonWeigthedPathList =  \
                            [p for p in shortestPathsGen]

                    if shortestPath in nonWeigthedPathList:
                        matched += 1
                    weigthList = []
                    for p in nonWeigthedPathList:
                        weigth = 0
                        for i in range(len(p)-1):
                            w = graph[p[i]][p[i+1]]['weight']
                            weigth += w
                        weigthList.append(weigth-shortestPathWeigth)
                    weigthStats.append([min(weigthList), 
                            np.average(weigthList),
                            max(weigthList), shortestPathWeigth,
                            len(nonWeigthedPathList)])
    wS = zip(*weigthStats)
    relativeDiff = np.average([p[1]/p[3] for p in weigthStats])
    results['routesMatched'] =  float(matched)/totRoutes
    results['avgDiff'] = np.average(wS[1])
    results['maxDiff'] = max(wS[2])
    results['relAvgDiff'] = relativeDiff
    results['avgWeight'] =  np.average(wS[3])
    results['avgNumPaths'] = np.average(wS[4])
    results['maxNumPaths'] = max(wS[4])
    return results


def extractData(G):
    """ extract wanted data from the graph. """
    data = dataObject()
    print "Evaluating ETX metrics"
    print "Now evaluating Group metrics"
    #getGroupMetrics(G, data)
    #data.etxStats = etxAnalysis(G)
    HNAAnalysis(G, data)
    data.printTable()

def HNAAnalysis(graph, data):
     
    data.HNA, paths = computeGroupHNAMetrics(graph, 1, weighted=True)
    for i in range(2,4):
        r,x = computeGroupHNAMetrics(graph, groupSize=i, weighted=True, 
                shortestPathsCache=paths)
        data.HNA['betweenness'][i] = r['betweenness'][i]

def degreeAnalysis(G):
    degree = G.degree()
    ccdf(sequence)

def etxAnalysis(G):
    """ analyze the etx metric properties. """
    # estimate what is the max etx to have a working link 
    maxEtx = hysteresisComputation()
    if maxEtx == 0:
        print >> sys.stderr, "something went badly wrong in",\
            "hysteresisComputation"
        sys.exit(1)
    etxSequence = sorted(range(10, 2*int(maxEtx*10)+1), reverse=True)
    connectedEtx = 0
    purgedGraph = None
    for i in etxSequence:
        etx = float(i)/10
        newGraph = purgeGraph(G, etx)
        if (len(newGraph) != len(G)):
            break
        connectedEtx = etx
        purgedGraph = newGraph

    etxSequence = [10, 11 , 12, 13, 14, 19, 22, 40]
    results = defaultdict()
    results['connectedEtx'] = connectedEtx
    for i in etxSequence:
        etx = float(i)/10
        purgedGraph = purgeGraph(G, etx)
        nodeSet = set(purgedGraph.nodes())
        allNodes = set(G.nodes())
        newGraph = G.copy()
        for n in allNodes-nodeSet:
            newGraph.remove_node(n)

        leafnodes = 0
        for node in newGraph.nodes():
            if len(nx.neighbors(newGraph, node)) == 1:
                leafnodes += 1

        m = evaluateMPRChoice(newGraph)

        runRes = {}
        compareShortestPaths(newGraph, purgedGraph, runRes)
        runRes['edges'] = len(purgedGraph.edges())
        runRes['nonLeaves'] = len(newGraph.nodes())-leafnodes
        runRes['nodes'] = len(newGraph.nodes())
        runRes['mprs'] = m['mprs']
        results[etx] = runRes
    return results

def getGroupMetrics(G, results):
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
    runResB = {}
    runResC = {}
    for i in range(4,6):
        res = computeGroupMetrics(G, groupSize=i, weighted=True, 
            cutoff = 2, shortestPathsCache=cache)
        cache = res[-1]
        runResB[i] = [res[0], res[1]]
        runResC[i] = [res[2], res[3]]
    results.groupMetrics['betweenness'] = runResB 
    results.groupMetrics['closeness'] = runResC 


# this stuff is not public yet, comment this
#from mpr import *
#def evaluateMPRChoice(G):
#    sol = solveMPRProblem(G)
#    if not checkSolution(G, sol):
#        print >> sys.stderr, "Error in the computing of the MPR set!"
#        sys.exit(1)
#    mprs = set()
#    for s in sol.values():
#        if len(s) != 0:
#            # this takes an element from a set without removing it
#            mprs |= iter(s).next()
#    res = {'mprs': len(mprs)}
#    return res


