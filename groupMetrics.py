import sys 
import math
import time

import numpy as np
import itertools as itt
import networkx as nx
from collections import defaultdict
from miscLibs import *
from scipy import stats

def computeGroupMetrics(graph, groupSize=1, weighted=False, cutoff=1,
        shortestPathsCache = None, mode="exhaustive"):
    """ find the set of nodes of with highest group betweenness.  
    
    Use weighted/nonweighted graphs and consider paths
    >= than cutoff. It uses multi-process to speed-up analysis

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
    mode : string
        this can be "exhaustive", "greedy" and determines the kind of
        search that must be done
    Return : best betweenness, array of groups of nodes, best closeness 
             array of groups of nodes, shortestPaths
    """


    # compute all the shortest paths. Depending on the graph dimension
    # this could be really large. Depending on the number of times you 
    # will need this data, it is worth to compute it once and keep it
    # for further analysis.

    diameter = 10000 # just a very long path weight
    if shortestPathsCache == None:
        shortestPaths = defaultdict(defaultdict)
        for source in graph.nodes():
                for target in graph.nodes():
                    if target in shortestPaths[source]:
                        continue
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
                    
                    # we consider only symmetric links (non directed 
                    # graphs)
                    shortestPaths[source][target] = weightedPaths
                    shortestPaths[target][source] = weightedPaths

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

    if mode == "exhaustive":
        launchParallelProcesses(dataObjects, 
                targetFunction=singleProcessGroupMetrics) 
    if mode == "greedy":
        launchParallelProcesses(dataObjects, 
                targetFunction=greedyGroupBetweenness) 

    # to have less computation it is better to compute betweenness 
    # without 1-hop routes. But to compute closeness we need all the routes.
    # If cutoff = 2 we can get the 1-hop neighbors from the graph, if cutoff>2 
    # the closeness value is meaningless
    if cutoff > 2:
        print >> sys.stderr, "Cutoff value larger than 2, closeness \
                centrality has no meaning"
    # add shortestPaths to the orginal tuple as return value
    return parseGroupMetricResults(dataObjects, mode) + (shortestPaths,)

def parseGroupMetricResults(dataObjects, mode):
    """ parse the results of a parallel computation and returns 
        the values.

        Parameters
        ----------
        dataObjects : retudn value from the parallel execution
        mode : exhaustive/greedy, it is needed since the two modes return
        different values
        
        Returns : best Betweenness, best corresponding group , 
                    best Closeness, best corresponding group
                    in case of greedy behaviour it returns an array of each
                    result with the results from 1 to groupSize
    """

    bestBetw = {}
    bestGroupB = {}
    bestGroupC = {}
    bestCloseness = {}
    # dataObjects is as follows:
    #   an array of dataObj one for process, each one is a dict with
    #   ['input'] / ['output'] labels
    #     label 'input' contains a dict with input parameter
    #     label 'output' contains a dict of solutions, one for each group 
    #     size  each element in the list contains a piece of the solution 
    #
    # so dataObject[0]['output'][5]['betweenness'] is the betweenness 
    # computed by process 0 on the group size 5. 

    # Note that each process computes a solution with a different input or 
    # with a different random seed, and only the greedy algorith returns a
    # dict of sizes, the exhaustive returns only the solution for the size 
    # you requested

    # note, we return only one solution among the possible ones. This has no real
    # impact in weighted graphs, in which the solution is most likely just 
    # one, but it may change in unweighted graphs
    for j in dataObjects[0]['output']: # loop on the sol size (1..groupSize)
        bestGroupB[j] = []
        bestGroupC[j] = []
        bestCloseness[j] = dataObjects[0]['input']['diameter']
        bestBetw[j] = 0
        for o in dataObjects: #loop on the solution produced by each process
            groupSize = o['input']['groupSize']
            if j in o['output']:
                if o['output'][j]['betweenness'] > bestBetw[j]:
                    bestGroupB[j] = o['output'][j]['groupB']
                    bestBetw[j] = o['output'][j]['betweenness']
                if o['output'][j]['closeness'] < bestCloseness[j]:
                    bestGroupC[j] = o['output'][j]['groupC']
                    bestCloseness[j] = o['output'][j]['closeness']

    if mode == "greedy":
        return bestBetw, bestGroupB, \
                bestCloseness, bestGroupC
    else:
        return bestBetw[groupSize], bestGroupB[groupSize], \
                bestCloseness[groupSize], bestGroupC[groupSize]

def greedyGroupBetweenness(dataObject, q):
    """ compute the shortest path betweenness (weighted) using the greedy 
    heuristic from a BU techrep: "A Framework for the Evaluation and 
    Management of Network Centrality", modified to be a grasp procedure:
    http://en.wikipedia.org/wiki/Greedy_randomized_adaptive_search_procedure

    This function must be called from launchParallelProcesses(), the input
    must be formatted accordingly (see the definiton). 
    
    At each step you have a current group of nodes g, a betweenness B(g)
    and a set of candidates made of [G.nodes()] - [g]. For each candidate x
    compute B(g+x) and build a map Inc[x] = B(g+x) - B(g). Then the process
    with id == 0 always chooses the x with maximum Inc[x]. The others choose 
    a random x with probability proportional to Inc[x].  At the end, the best
    one is chosen.
    Betweenness and Closeness vary a little in the computation but the principle 
    is the same.

    Parameters
    ---------

    dataObject : input and output values
    q : output queue
    """

    nodeList = dataObject['input']['nodeList']
    groupSize = dataObject['input']['groupSize']
    myId = dataObject['input']['id']
    graph = dataObject['input']['graph']
    shortestPaths = dataObject['input']['paths']
    nodeSet = set(nodeList)
    currentBGroup = set()
    currentCGroup = set()
    bestRes = {}
    # initialize with out of scale values
    bestB = 0 
    bestC = 100
    # these two dict are needed since 
    # the random generator uses floats 
    # as labels, not strings
    nodeToFloat = {}
    floatToNode = {}
    counter = 0
    for n in graph.nodes():
        nodeToFloat[n] = counter
        floatToNode[counter] = n
        counter += 1

    for i in range(1,groupSize+1):
        bestRes[i] = {}
        candidatesB = nodeSet - currentBGroup
        candidatesC = nodeSet - currentCGroup
        Bdict = {}
        Cdict = {}
        # I try to use one loop for both metrics
        for n in candidatesB|candidatesC:
            newG = currentBGroup|set([n])
            betw, cl = groupMetricForOneGroup(graph, newG, shortestPaths)
            # save for each candidate group the increment Vs the 
            # current solution
            Bdict[nodeToFloat[n]] = betw - bestB

            # closeness can be not monotinc with len(newG). See comments in
            # groupMetricForOneGroup(). Shoul be unneeded but I keep it 
            # for reference
            if bestC > cl: 
                Cdict[nodeToFloat[n]] = bestC - cl

        # to have only one loop I may add nodes already in the current solution
        for n in currentBGroup:
            try: 
                del Bdict[nodeToFloat[n]]
            except KeyError:
                pass
        for n in currentCGroup:
            try: 
                del Cdict[nodeToFloat[n]]
            except KeyError:
                pass

        if myId == 0:
            # one process deterministically chooses the best solution.
            # Recall we want the highest betweenness and the lowest closeness
            if len(Bdict) != 0:
                f = sorted(Bdict.items(), key=lambda x: x[1],
                        reverse=True)[myId]
                currentBGroup.add(floatToNode[f[0]]) 
                bestB = bestB + f[1]
            if len(Cdict) != 0:
                f = sorted(Cdict.items(), key=lambda x: x[1],
                        reverse=True)[myId]
                currentCGroup.add(floatToNode[f[0]])
                bestC = bestC - f[1]
        else:
            # every element has a probability of being chosen that 
            # is proportional to the normalized gain of the target
            # function for that choice
            # this is a grasp optimization method, the more processes
            # are run in parallel the higher the chance of finding 
            # a better solution
            np.random.seed(myId*int(time.time()))
            if len(Bdict) != 0:
                totIncrement = sum(Bdict.values())
                normalizedIncrement = [k/totIncrement for k in Bdict.values()]
                # stats only handles integer labels, that's the reason for 
                # dummy floatToNode[] and nodeToFloat[]
                custDist = stats.rv_discrete(values=(Bdict.keys(), 
                    normalizedIncrement), name="custDist")
                r = custDist.rvs()
                f = floatToNode[r]
                currentBGroup.add(f)
                bestB = bestB + Bdict[r]

            if len(Cdict) != 0:
                totIncrement = sum(Cdict.values())
                normalizedIncrement = [k/totIncrement for k in Cdict.values()]
                custDist = stats.rv_discrete(values=(Cdict.keys(), 
                    normalizedIncrement), name="custDist")

                r = custDist.rvs()
                f = floatToNode[r]
                currentCGroup.add(f)
                bestC = bestC - Cdict[r]


        bestRes[i]['groupC'] = [n for n in currentCGroup]
        bestRes[i]['groupB'] = [n for n in currentBGroup]
        bestRes[i]['betweenness'] = bestB
        bestRes[i]['closeness'] = bestC
    q.put(bestRes)

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

    # with the echaustive algorithm, bestRes is a simple dictionary, it
    # should not need to be an array of dictionaries. I use an array 
    # for compatibility with the greedy approach

    bestRes[groupSize] = {}
    bestRes[groupSize]['groupC'] = bestGroupC
    bestRes[groupSize]['groupB'] = bestGroupB
    bestRes[groupSize]['betweenness'] = bestB
    bestRes[groupSize]['closeness'] = bestC
    q.put(bestRes)


def groupMetricForOneGroup(graph, group, shortestPaths):
    """ compute the group betweeness and closeness centrality."""

    """ graph:  the graph

        group: the group to be tested

        shortestPaths: all the shortest paths in the graph (so they
                       can be precomputed and cached)

        Return: bertweenness group centrality and closeness group centrality
    """

    # paths that have been matched by any node in the group (betweenness)
    numPathsMatched = 0.0
    # total number of paths, excluding the ones stardint from one of the nodes
    # in the group (Note, there can be more than one shortest path 
    # between every (source,dest))
    numPaths = 0
    # sum variable of the minimum distance to any of the nodes in the group
    totalPathLenght = 0
    # upper bound to closeness
    MAX_PATH_WEIGHT = 10000 
    for source in shortestPaths:
        # if source is in the group, his minimum distance from 
        # the group is 0. This keeps the closeness monotonic with
        # the number of nodes in the group.
        # Otherwise, closeness is not monotonic with the number of elements in
        # the group. Consider the line:
        # 
        # 1 --  2 -- 3 -- 4
        #
        # g = [2] has closeness 1+1+2/3 = 4/3;  g = [2,1] has closeness 1+2/2 =
        # 3/2 > 4/3
        m = 0
        tot = 0

        if source in group:
            firstMatchLength = 0
        else:
            firstMatchLength = MAX_PATH_WEIGHT

        for dest in shortestPaths[source]: 
            if dest == source:
                continue
            # compute closeness centrality (for nodes that are 
            # not in the group that have distance = 0)
            if dest in group and firstMatchLength != 0: 
                    if shortestPaths[source][dest] != []:
                     length = shortestPaths[source][dest][0]['weight']
                     if firstMatchLength > length:
                            firstMatchLength = length
            # loop on all the routes between source and dest
            # compute betweenness
            for route in shortestPaths[source][dest]: 
                tot += 1
                numPaths += 1
                # loop on the nodes of this route
                for node in route['path']: 
                    if node in group:
                        numPathsMatched += 1
                        m += 1
                        break # exit from this route
        totalPathLenght += firstMatchLength
        if firstMatchLength == MAX_PATH_WEIGHT:
            print >> sys.stderr,  "Error: node", source,\
                 " has no route to any node in ", group 
            sys.exit(1)
    if numPaths == 0 or len(shortestPaths) == 0:
        print >> sys.stderr, "ERROR: ", len(shortestPaths)
        return 0,0
    else:
        betweennessValue = float(numPathsMatched)/numPaths
        closenessValue = float(totalPathLenght)/len(shortestPaths)
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
