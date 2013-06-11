import sys 
import copy
import math

from guppy import hpy
import numpy as np
import itertools as itt
import networkx as nx
from collections import defaultdict
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

    diameter = 0
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
                        if weight > diameter:
                            diameter = weight
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

    # remove leaf nodes, they have no centrality
    purgedGraph = []
    for node in graph.nodes():
        if graph.degree(node) > 1:
            purgedGraph.append(node)

    
    # use launchParallelProcesses to parallelize the search

    parallelism = 4
    dataObjects = []
    for p in range(parallelism):
        dataObj = {}
        dataObj['input'] = {}
        dataObj['output'] = {}
        # free some memory
        dataObj['input']['paths'] = shortestPaths
        dataObj['input']['id'] = p
        dataObj['input']['groupSize'] = groupSize
        dataObj['input']['nodeList'] = purgedGraph
        dataObj['input']['graph'] = graph
        dataObj['input']['numProcessess'] = parallelism
        dataObj['input']['diameter'] = diameter
        dataObjects.append(dataObj)
        # when each process deletes memory for its own group, 
        # if we still have a reference to the dataObj object 
        # it is not deallocated. So deallocate it (it is worth if
        # you pass a lot of data to each process)
        del dataObj
    launchParallelProcesses(dataObjects, 
            targetFunction=singleProcessGroupMetrics) 

    bestBetw = 0
    bestCloseness = diameter
    bestGroupB = []
    bestGroupC = []
    for o in dataObjects:
        if o['output']['betweenness'] >= bestBetw:
            if  o['output']['betweenness'] > bestBetw:
                bestGroupB =[]
            bestBetw = o['output']['betweenness']
            bestGroupB += o['output']['groupB']
        if o['output']['closeness'] <= bestCloseness:
            if o['output']['closeness'] < bestCloseness:
                bestGroupC = []
            bestCloseness = o['output']['closeness']
            bestGroupC += o['output']['groupC']
    
    # to have less computation it is better to compute betweenness 
    # without 1-hop routes. But to compute closeness we need all the routes.
    # If cutoff = 2 we can get the 1-hop neighbors from the graph, if cutoff>2 
    # the closeness value is meaningless
    if cutoff > 2:
        print >> sys.stderr, "Cutoff value larger than 2, closeness \
                centrality has no meaning"
        bestCloseness = 0
        bestGroupC = []
    return bestBetw, bestGroupB, \
        bestCloseness, bestGroupC, shortestPaths


def singleProcessGroupMetrics(dataObject, q):
    """" compute the centrality for a batch of groups. """ 

    """ this function must be called from launchParallelProcesses(), the input
    must be formatted accordingly (see the definiton). Each launched process,
    will generate a set of groups depending on its id and evaluate each of
    then using groupMetricForOneGroup(). Outputs are writte in a data queue
    and collected in launchParallelProcesses"""

    nodeList = dataObject['input']['nodeList']
    groupSize = dataObject['input']['groupSize']
    myId = dataObject['input']['id']
    numProcessess = dataObject['input']['numProcessess']
    graph = dataObject['input']['graph']
    shortestPaths = dataObject['input']['paths']
    diameter = dataObject['input']['diameter']

    groupIt, myStart, myEnd = generateCombinations(nodeList, 
            groupSize, myId, numProcessess)
    currGroup = myStart
    bestRes = {}
    if myStart == myEnd:
        bestRes['betweenness'] = 0
        bestRes['closeness'] = diameter
        bestRes['groupB'] = []
        bestRes['groupC'] = []
        dataObject['output'] = bestRes
        print "Subprocess with empy dataset"
        q.put(bestRes)
        return
    bestB = 0
    bestC = diameter
    bestGroupB =[]
    bestGroupC = []

    while True:
        try:
            group = groupIt.next()
        except StopIteration:
            break;
        if myEnd-myStart > 3:
            if ((currGroup-myStart) % ((myEnd-myStart)/3)) == 0:
                workProgress = 100*(currGroup-myStart)/(myEnd-myStart)
                print "Process ", myId, "elaborated", \
                        workProgress, "% of the groups"
        if currGroup == myEnd:
            break
        currGroup += 1
        betw, clos = groupMetricForOneGroup(graph, group, shortestPaths)
        if betw >= bestB:
            if betw > bestB:
                bestB = betw
                bestGroupB = []
            bestGroupB.append(set(group))
        if clos <= bestC:
            if clos < bestC:
                bestC = clos
                bestGroupC = []
            bestGroupC.append(set(group))

    bestRes['groupC'] = bestGroupC
    bestRes['groupB'] = bestGroupB
    bestRes['betweenness'] = bestB
    bestRes['closeness'] = bestC
    q.put(bestRes)


def groupMetricForOneGroup(graph, group, shortestPaths):
    """ compute the group betweeness and closeness centrality."""

    """ graph:  the graph

        group: the group to be tested

        shortestPaths: all the shortest paths in the graph (so they
                       can be precomputed and cached)

        Return: bertweenness group centrality and closeness group centrality
    """

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
            for route in shortestPaths[source][dest]:
                numPaths += 1
                for node in route['path']:
                    if node in group:
                        numPathsMatched += 1
                        break 
        totalPathLenght += firstMatchLength
        allPaths += 1
    betweennessValue = float(numPathsMatched)/numPaths
    closenessValue = float(totalPathLenght)/allPaths
    return betweennessValue, closenessValue



def generateCombinations(nodeList, groupSize, myId, numProcessess):
    """ returns an iterator with all the combinations of size groupSize. """

    """ 
        nodeList: the set of nodes to combine

        groupSize: the dimension of the group

        myId: return the iterator in the correct position to be used by
              the myId-esim process over a total of numProcesses (see below)

        Return: iterator over the combinations, start and end for this process
    """
    groupIt = itt.combinations(nodeList, groupSize)
    combinations = math.factorial(len(nodeList))/\
                   (math.factorial(groupSize)*\
                   math.factorial(len(nodeList)-groupSize)) 
    
    # if you have N combinations, m processes and myId = x then 
    # this function will return an iterator to all the combinations
    # that points to the element (N/m)*x, together with the start and end value
    # for this process

    if combinations > numProcessess:
        blockSize = combinations/numProcessess
    else:
        blockSize = 1
    myStart = myId*blockSize
    if myId != numProcessess-1 and myStart < combinations:
        myEnd = (myId+1)*blockSize
    elif myStart < combinations:
        myEnd = combinations
    else:
        myEnd = 0
        myStart = 0

    for i in range(myStart):
        groupIt.next()
    return groupIt, myStart, myEnd

def computeGroupHNAMetrics(graph, groupSize=1, weighted=False,
        shortestPathsCache = None):

    # TODO: 
    #  - computa min-cut for any node to a gateway

    newGraph = compressGraph(graph)
    statRes = {}
    if shortestPathsCache == None:
        if weighted == False:
            nw, shortestPathsCache  = computeInternetAccessStats(newGraph, weighted=False) 
            statRes['unweighted'] = nw
        else:
            w, shortestPathsCache = computeInternetAccessStats(newGraph, weighted=True) 
            statRes['weighted'] = w
    nodeList = newGraph.nodes()
    # remove the leaves from the groups
    for n in newGraph.nodes():
        if len(nx.neighbors(newGraph, n)) == 1:
            nodeList.remove(n)

    # remove the gateway from the groups
    if 0 in nodeList:
        nodeList.remove(0)

    # TODO set this somewhere global
    parallelism = 4
    results = {}
    results[groupSize] = {}
    print "Evaluating groupSize", groupSize
    dataObjects = []
    for p in range(parallelism):
        dataObj = {}
        dataObj['input'] = {}
        dataObj['output'] = {}
        dataObj['input']['paths'] =shortestPathsCache 
        dataObj['input']['nodeList'] = nodeList
        dataObj['input']['id'] = p
        dataObj['input']['numProcessess'] = parallelism
        dataObj['input']['groupSize'] = groupSize
        dataObjects.append(dataObj)
        # when each process deletes memory for its own group, 
        # if we still have a reference to the dataObj object 
        # it is not deallocated. So deallocate it
        del dataObj

    launchParallelProcesses(dataObjects, 
            targetFunction=computeBetweennessToHNA) 
    
    bestBetw = 0
    bestGroup = []
    for o in dataObjects:
        if o['output']['betweenness'] >= bestBetw:
            bestBetw = o['output']['betweenness']
            bestGroup = o['output']['group']
    results[groupSize] = {'betweenness':bestBetw, 'group':bestGroup}
    statRes['betweenness'] = results
    return statRes, shortestPathsCache


def computeBetweennessToHNA(dataObject, q):
    """ find the set of nodes with the highest group betw. to gateways."""
    
    """ this function must be called from launchParallelProcesses(), the input
    must be formatted accordingly (see the definiton). Each launched process,
    will generate a set of groups depending on its id and evaluate each of
    then using groupMetricForOneGroup(). Outputs are written in a data queue
    and collected in launchParallelProcesses"""

    # HNA entries are compressed in a single node with ID 0

    bestBetw = 0
    bestGroup = []
    routes =  dataObject['input']['paths']
    nodeList = dataObject['input']['nodeList']
    myId = dataObject['input']['id']
    numProcessess = dataObject['input']['numProcessess']
    groupSize = dataObject['input']['groupSize']

    groupIt, myStart, myEnd = generateCombinations(nodeList, groupSize, myId,
            numProcessess)
    currGroup = myStart
    bestRes = {}

    if myStart == myEnd:
        bestRes['betweenness'] = 0
        bestRes['group'] = []
        dataObject['output'] = bestRes
        print "Subprocess with empy dataset"
        q.put(bestRes)
        return
    
    while True:
        try:
            group = groupIt.next()
        except StopIteration:
            break;
        if myStart - myEnd > 3:
            if ((currGroup-myStart) % ((myEnd-myStart)/3)) == 0:
                workProgress = 100*(currGroup-myStart)/(myEnd-myStart)
                print "Process ", myId, "elaborated", \
                        workProgress, "% of the groups"
        if currGroup == myEnd:
            break
        currGroup += 1
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
        if float(matched)/numRoutes > bestBetw:
            bestBetw = float(matched)/numRoutes
            bestGroup = group
    #free some memory
    bestRes['betweenness'] = bestBetw
    bestRes['group'] = bestGroup
    dataObject['output'] = bestRes
    q.put(bestRes)
            

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
    """ compute stats for path length to internet gateways. """
    """ 
        G: graph

        weighted: use/don't use weights

        returns: dict with some stats and the shortest paths computed
    """
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
        for source in nodeList:
            pt = nx.all_shortest_paths(G, source, 0)
            for p in pt:
                paths.append([p, len(p) -2]) 
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
