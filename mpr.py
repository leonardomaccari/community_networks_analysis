import code
import random as rnd
import networkx as nx 
from collections import defaultdict
import sys
    
#global to speed up search in set of neighbors
hop1set = {}

def checkSolution(g, mprChoice):
    """ check if this mpr Choice is correct."""

    # rebuild the a graph made only of links between 
    # selector-mpr using the given mpr choice. 
    # If the choice is correct the graph must be still connected 

    newGraph = purgeNonMPRLinks(g, mprChoice)
    return nx.is_connected(newGraph) 


def purgeNonMPRLinks(graph, solution):
    """ generate a new graph only with links selector-MPR."""
    newGraph = nx.Graph()
    newGraph.add_nodes_from(graph.nodes())
    for n1 in graph.nodes():
        # see the return value of solveMPRProblem to understand this:
        for m in iter(solution[n1]).next():
            newGraph.add_edge(n1,m)
    return newGraph

def solveMPRProblem(graph, mode = "RFC"):
    """ given a graph, return one or more than one MPR choice.

    Parameters:
    -----------
    graph: the graph 
    mode: "RFC" follows the greedy approach of the RFC, "lq" follows
    the link quality approach of freifunk (present in olsrd)

    Returns:
        ss: dict{'node':set(frozenset(mpr1,mpr2..))}
            to each node corresponds a set of solutions for its choice
            of MPRs. Each solution is a frozenset of nodes. Currently
            only one solution (the first found) is returned.
    """
    for node in graph.nodes():
        hop1set[node] = hop1(node, graph)

    nodes = graph.nodes()
    ss = {}
    #counter = 0
    for node in nodes:
        #percent = int(100*(len(nodes)-counter)/float(len(nodes)))
        #if percent % 10 == 0:
        #    print 100-percent,"% completed"
        #counter += 1
        if mode == "lq":
            s = lqmprset(node, graph)
        else:
            s = mprset(node, graph)
        solset = set([frozenset(x) for x in s])
        ss[node] = solset 
    return ss

def lqmprset(i, graph):
    """ find isolated nodes and apply quality-based algorithm."""
    mprv, ni2d = isolatedNodes(i,graph)
    n2hop=defaultdict(dict)
    for neigh in graph[i].items():
        try:
            w = graph[i][neigh[0]]["weight"]
        except KeyError:
            print >> sys.stderr, "ERROR: can not compute lq mpr on a graph",\
                    "without weights on the edges: w", graph[i][neigh[0]]
            sys.exit()
        for h2neigh in graph[neigh[0]].items():
            n2hop[h2neigh[0]][neigh[1]['weight']+w] = neigh[0]
    mprset = set()
    for neigh2, weightSet in n2hop.items():
        weightMin = min(weightSet)
        mprset.add(weightSet[weightMin])
    for n in mprv:
        mprset.add(n)

    return [mprset]

def mprset(i, graph):
    """ find isolated nodes and start recursion."""
    mprv, ni2d = isolatedNodes(i,graph)
    solutions = []
    # recursively find one/all solutions
    recmpr(mprv, solutions, ni2d, i, graph)
    return solutions

def isolatedNodes(i, graph):
    """ insert into the mpr set the nodes that connect isolated nodes."""
    mprv = []
    ni1, ni2, ni2d = hop2(i, graph)
    # seleziona mpr che sono i soli a raggiungere nodi a 2 hop
    for j in ni2:
        nj1 = hop1set[j] #, graph)
        mpr = ni1 & nj1
        if len(mpr) == 1:
            for m in mpr: 
                if m not in mprv:
                    mprv.append(m)
            # remove covered nodes
            for k in ni2d.keys():
                for mn in mpr:
                    ni2d[k] = ni2d[k] - hop1set[mn] #, graph)
    return mprv, ni2d

def hop1(node, graph):
    """ helper function."""
    return set(graph.neighbors(node))

def hop2(node, graph):
    """ helper function."""
    neighs = hop1set[node]
    hop2n = set()
    neighsDic = {}
    for nneigh in neighs:
        nneighsSet = hop1(nneigh, graph)
        nneighsSet.remove(node)
        neighsDic[nneigh] = nneighsSet-neighs
        hop2n |= (nneighsSet-neighs)
    return neighs, hop2n, neighsDic


def recmpr(m, solutions, n, thisNode, graph):
    """ recursive MPR solution finder."""
    l = sorted(n.items(), key=lambda x: len(x[1]), reverse=True)
    if len(l[0][1]) == 0:
        solutions.append(m)
        #print "return"
        return
    for i in l:
        # If I just want one solution, I've already generated 
        # the first one and I'm trying to optimize it, just break
        # after the first loop. This is the default greedy algorithm.
        # Else, if you want to explore the full space of solutions
        # remove this break.
        if i != l[0]:
            break
        
        if i[0] in m:
            continue;
        # add old mprs
        mprv = [e for e in m] 
        # add the new one
        mprv.append(i[0])
        ni2d = {}
        for k in n.keys():
            ni2d[k] = n[k] - hop1set[i[0]] #, graph)
        recmpr(mprv, solutions, ni2d, thisNode, graph)

def hysteresisComputation():
    
    # real default parameters from olsrd
    HYST_SCALING = 0.5
    HYST_THRESHOLD_HIGH = 0.8
    HYST_THRESHOLD_LOW = 0.3
    L_link_quality = 0.5

    # parameters for the estimation window. If the time spent in the 
    # pending state is more than this fraction, bad link (recall that 
    # pending does not necessarily mean off)
    MAX_PENDING_TIME = 0.8
    averageWindow = 100

    qList = []
    pending = True
    maxEtx = 0
    for i in range(1,50):
        etx = 1+float(i)/10
        prob = 1/etx
        # this loop implements the OLSR Hysteresis algorithm to decide
        # if a link is ok or it is pending. We compute the "pending" state
        # for 1000 seconds
        for i in range(1000):
            if rnd.random() < prob:
                L_link_quality = (1-HYST_SCALING)*L_link_quality + \
                    HYST_SCALING
            else:
                L_link_quality = (1-HYST_SCALING)*L_link_quality

            if L_link_quality > HYST_THRESHOLD_HIGH:
                pending = False
            elif L_link_quality < HYST_THRESHOLD_LOW:
                pending = True
            qList.append(pending)

        # if the link is in the pending state more than MAX_PENDING_TIME
        # then the link is off 
        if len(filter(lambda x: x == True, qList[-averageWindow:]))/\
                float(averageWindow) > MAX_PENDING_TIME:
            break
        else:
            maxEtx = etx
    return maxEtx


