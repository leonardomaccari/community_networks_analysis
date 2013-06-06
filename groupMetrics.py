import sys 
import time
import copy
import math

from guppy import hpy
import numpy as np
import itertools as itt
import networkx as nx
from collections import defaultdict
from multiprocessing import Process, Queue
from miscLibs import *

def computeGroupMetrics(graph, groupSize=1, weighted=False, cutoff=1,
        shortestPathsCache = None):
    """ find the set of nodes of with highest group betweenness.  
    
    Use weighted/nonweighted graphs and consider paths
    longer than cutoff. It uses multi-process to speed-up analysis

    Parameters
    --------------
    graph : graph
    groupSize: int
    weighted: bool
    cutoff : int
        minimum lenght of the paths considered
    shortestPathsCache: defaultdict(defaultdict())
        in case this function has already been run, we keep a cache
        for the shortestPath dictionary, for optimization

    Return : best betweenness, array of groups of nodes, best closeness 
             array of groups of nodes, shortestPaths
    """


    # compute all the shortest paths. Depending on the graph dimension
    # this could be really large. Depending on the number of times you 
    # will need this data, it is worth to compute it once and keep it
    # for further analysis.
    if shortestPathsCache == None:
        if cutoff <= 0:
            print >> sys.stderr, "Error: did you set a cutoff value",\
                "lower than 1?"
            sys.exit(1)

        shortestPaths = defaultdict(defaultdict)
        for source in graph.nodes():
                for target in graph.nodes():
                    if not weighted:
                        shortestPathsGen = nx.all_shortest_paths(graph,
                            source, target)
                        pathList =  [p for p in shortestPathsGen]
                        weight = len(pathList[0])-1
                    else :
                        shortestPathsGen = nx.all_shortest_paths(graph,
                            source, target, weight="weight")
                        weight = nx.shortest_path_length(graph,
                            source, target, weight="weight")
                        pathList =  [p for p in shortestPathsGen]
                    weightedPaths = []
                    for p in pathList:
                        weightedPaths.append({'path':p, 'weight':weight})
                    
                    shortestPaths[source][target] = weightedPaths
                    tooShortPaths = []
                    if cutoff > 1:
                        for route in shortestPaths[source][target]:
                            routeL = len(route['path'])
                            # recall that len(route) counts the end nodes
                            # so len(path(0,1)) = len([0,1]) = 2
                            if routeL <= cutoff:
                                tooShortPaths.append(target)
                                break
                    
                    for dest in tooShortPaths:
                        del shortestPaths[source][dest] 

    else:
        shortestPaths = shortestPathsCache

    solutionsBetweenness = defaultdict(list)
    solutionsCloseness = defaultdict(list)
    # remove leaf nodes, they have no centrality
    purgedGraph = []
    for node in graph.nodes():
        if graph.degree(node) > 1:
            purgedGraph.append(node)

    # combinations() returns an iterable
    groupArray = [g for g in itt.combinations(purgedGraph, groupSize)]

    numComb = len(groupArray)
    procNumber = len(groupArray)

    counter = 0
    procList = {}
    parallelism = 8
    # unused, if you want to kill processes after a max time, 
    # set to smaller values
    maxLifeTime = 100000
    killed = 0
    queueStack = []
    print "launching subprocesses"
    for i in range(parallelism):
        queueStack.append(Queue())
    while True:
        aliveProc = 0
        #time.sleep(sleepTime)
        toBePurged = []
        for p,v in procList.items():
            if p.is_alive():
                tt = time.time()
                if tt-v[0] < maxLifeTime: 
                    aliveProc += 1
                else: 
                    print "killed a process after ",\
                        int(tt-v[0]), "seconds"
                    p.terminate()
                    killed += 1
                    for proc in procList:
                        if proc.is_alive():
                            proc.terminate()
                    toBePurged.append(p)
            else:
                sol = v[1].get()
                queueStack.append(v[1])
                toBePurged.append(p)
                solutionsBetweenness[sol["betweenness"]].append(set(v[2]))
                solutionsCloseness[sol["closeness"]].append(set(v[2]))

        for proc in toBePurged:
            del procList[proc]

        if aliveProc == 0 and procNumber == 0: 
            break

        if aliveProc < parallelism and procNumber > 0:
            group = copy.copy(groupArray[counter])
            q = queueStack.pop()
            p = Process(target=groupBetweenness,
                    args=(graph, group, shortestPaths, q))
            procList[p] = [time.time(), q, group]
            p.start()
            procNumber -= 1
            counter += 1
            if numComb > 10 and counter % (numComb/10) == 0:
                print "Analyzed", counter*100 / numComb, "% of groups"
    print "Launched and ended", numComb, "processes"
    bestBet = sorted(solutionsBetweenness, reverse=True)[0]
    bestClos = sorted(solutionsCloseness)[0]
    # to have less computation it is better to compute betweenness 
    # without 1-hop routes. But to compute closeness we need all the routes.
    # If cutoff = 2 we can get the 1-hop neighbors from the graph, if cutoff>2 
    # the closeness value is meaningless
    if cutoff > 2:
        print >> sys.stderr, "Cutoff value larger than 2, closeness \
                centrality has no meaning"
        solutionsCloseness[bestClos] = []
    return bestBet, solutionsBetweenness[bestBet], \
            bestClos, solutionsCloseness[bestClos], shortestPaths


def groupBetweenness(graph, group, shortestPaths, q):
    """ compute the group betweeness and closeness centrality."""

    numPathsMatched = 0.0
    numPaths = 0
    totalPathLenght = 0
    allPaths = 0
    MAX_PATH_WEIGHT = 10000 
    for source in shortestPaths:
        if source in group:
            continue
        firstMatchLength = MAX_PATH_WEIGHT
        neighSet = set(nx.neighbors(graph,source)) & set(group)
        firstMatchLength = MAX_PATH_WEIGHT
        # 1) a node in group is neighbor of source
        if neighSet != set():
            # if cutoff > 1 we miss 1-hop routes, so get the neighbors
            # from the graph
            for n in neighSet:
                if n in graph[source]:
                    if 'weight' in graph[source][n]:
                        length = graph[source][n]['weight']
                    else:
                        length = 1
                    if length < firstMatchLength:
                        firstMatchLength = length
        for dest in shortestPaths[source]:
            # 2) dest is in group (even if case 1) is true, we may have
            # shortest paths when weight != 1)
            #   - update the closeness centrality
            #   - skip the betweenness (we omit the case "dest is in group")
            if dest in group:
                if shortestPaths[source][dest] != []:
                    length = shortestPaths[source][dest][0]['weight']
                    if firstMatchLength > length:
                        firstMatchLength = length
                continue
            #print "xx", source, dest, shortestPaths[source][dest]
            for route in shortestPaths[source][dest]:
                numPaths += 1
                for node in route['path']:
                    if node in group:
                        numPathsMatched += 1
                        break 
        totalPathLenght += firstMatchLength
        allPaths += 1
        #print group, source, totalPathLenght,firstMatchLength ,  allPaths, \
        #        totalPathLenght/float(allPaths)
    betweennessValue = float(numPathsMatched)/numPaths
    closenessValue = float(totalPathLenght)/allPaths
    solution = {"betweenness":betweennessValue, 
            "closeness":closenessValue}
    q.put(solution)


    """ find the set of nodes with the highest group betw. to gateways."""



def computeBetweennessToHNA(nodeList, routes, groupSize):

    print "Evaluating groupSize", groupSize
    groupArray = [g for g in itt.combinations(nodeList, groupSize)]
    bestBetw = 0
    bestGroup = []
    for group in groupArray:
        matched = 0
        numRoutes = 0
        for path in zip(*routes)[0]:
            if path[0] in group:
                continue
            for node in path:
                if node in group:
                    matched += 1
                    break
            numRoutes += 1
        if len(groupArray) > 10 and groupArray.index(group) % (len(groupArray)/10) == 0:
            print "Analyzed", groupArray.index(group)*100 / len(groupArray), "% of groups"
                
        if float(matched)/numRoutes > bestBetw:
            bestBetw = float(matched)/numRoutes
            bestGroup = group
    return bestBetw, bestGroup
    
            

def compressGraph(graph):
    """ compact the gateway nodes to one single node."""

    """ Each node in the graph with ID < 0 is a 0.0.0.0/0 HNA route (a gateway 
    to the Internet). We compact them in one single node for easier analysis."""

    newGraph = graph.copy()
    if 0 in newGraph.nodes():
        print "We use node id 0 for representing an abstract gateway node",\
        "please do not use ID 0 in the graph definition"
        sys.exit(1)
    negNodes = []
    negNeighs = []
    for node in newGraph.nodes():
        if node < 0:
            negNodes.append(node)
            for neighs in nx.neighbors(newGraph, node):
                negNeighs.append(neighs)
    if len(negNodes) == 0:
        print "no gateway found for the network. Gateways are expected to ",\
                "have an ID lower than zero"
    newGraph.add_node(0)
    for neigh in negNeighs:
        newGraph.add_edge(0, neigh, weight=1)
    for node in negNodes:
        newGraph.remove_node(node)
    return newGraph

def computeInternetAccessStats(G, weighted=True):
    """ compute stats for paths to internet gateways. """
    # 0-node is the compressed node representing all gateways
    # remove the neighbors of the 0-node, we do not care of 1-hop neighbors

    nodeList = set(G.nodes())
    nodeList = nodeList - set(nx.neighbors(G, 0)) - set([0])
    paths = []
    if weighted:
        for source in nodeList:
            p = nx.shortest_path(G, source, 0, weight='weight')
            w = nx.shortest_path_length(G, source, 0 , weight="weight") - 1
            paths.append([p,w])
    else:
        allPathGen = nx.shortest_path(G, 0)
        for t,l in allPathGen.items():
            if t in nodeList:
                paths.append([l, len(l) - 1 - 1]) 
                # one for the fake hop, one couse 
                # path begins on source node
    weights = zip(*paths)[1]
    bins = []
    for i in range(1, int(math.ceil(max(weights))) + 1 ):
        bins.append(i)
        bins.append(i+0.5)
    bins.pop()

    CCDF = ccdf(weights, bins=bins)
    retValue = {}
    retValue['ccdf'] = CCDF
    retValue['avg'] = np.average(weights)
    retValue['min'] = min(weights)
    retValue['max'] = max(weights)
    return  retValue, paths
