
# a set of functions related to the robustness of a graph, and to the
# robustness of the diffusion of messages in a graph

from collections import defaultdict
from random import choice
import networkx as nx
import numpy as np
from scipy import stats
import sys


def computeRobustness(graph, tests=100, mode="simple", 
        purge="links", nodeList=""):
    """ Compute robustness with a percolation approach. 
    Parameters:
    ----------
    graph: nx graph
            the graph to analyze 
    tests: int
            the number of tests to perform in each graph
    mode: str
            "simple" == remove random link/nodes
            "core" == remove only links/nodes in the core graph (no leaves)
    purge: str
            "links" == remove edges (bond percolation)
            "nodes" == remove nodes (edge percolation)
    nodeList: list
            instead of simulating random failures, pass a list of nodes
            to be progressively removed
    """

    links = []
    weights = []
    # build a list of nodes/links that can be removed
    for l in graph.edges(data=True):
        if mode == "core":
            if len(graph[l[0]]) != 1 and \
                    len(graph[l[1]]) != 1:
                links.append((l[0], l[1])) 
                if "weight" in l[2]:
                    weights.append(float(l[2]["weight"]))
                else: 
                    weights.append(1.0)
        else:
            links.append((l[0], l[1])) 
            if "weight" in l[2]:
                weights.append(float(l[2]["weight"]))
            else: 
                weights.append(1.0)

    if purge == "links": 
        totWeight = sum(weights)
        # this will increase the probablity of failures for links
        # that have a high weight. it is meaningful when the weight
        # represents the badness of the link

        normalizedWeight = [l/totWeight for l in weights]
        #normalizedWeight = [1.0/len(links) for l in links]
        custDist = stats.rv_discrete(values=(range(len(links)),
            normalizedWeight), name="custDist")

    mainCSize = defaultdict(list)
    mainNonCSize = defaultdict(list)
    nlen = float(len(graph.nodes()))
    elen = float(len(graph.edges()))
    print nlen

    for i in range(tests):
        purgedGraph = graph.copy()
        purgedLinks = []
        purgedNodes = []
        nodeRank = nodeList[:]
        if purge == "links":
            purgedItems = int(elen/2)
        elif purge == "nodes":
            purgedItems = int(nlen/2)

        for k in range(1,purgedItems):
            if purge == "links":
                r = custDist.rvs()
                if len(purgedLinks) >= elen:
                    print >> sys.stderr, "Trying to purge",k,"links",\
                        "from a graph with",elen," total links" 
                    break
                # get a random item that we did not remove yet
                while (r in purgedLinks):
                    r = custDist.rvs()
                purgedLinks.append(r) 
                l = links[r]
                purgedGraph.remove_edge(l[0],l[1])
            elif purge == "nodes":
                if len(purgedNodes) >= nlen:
                    print >> sys.stderr, "Trying to purge",k,"nodes",\
                        "from a graph with", nlen," total nodes" 
                    break
                if nodeRank == "":
                    r = choice(purgedGraph.nodes())
                    purgedNodes.append(r)
                else :
                    r = nodeRank.pop()
                    purgedNodes.append(r)
                purgedGraph.remove_node(r)

            compList = nx.connected_components(purgedGraph)
            mainCSize[k].append(len(compList[0])/nlen)
            compSizes = [len(h) for h in compList[1:]]

            if len(compSizes) == 0:
                mainNonCSize[k].append(0)
            else:
                mainNonCSize[k].append(
                    np.average(compSizes)/nlen)
    mainCSizeAvg = {}
    for k, tests in mainCSize.items():
        mainCSizeAvg[k] = np.average(tests)
    return mainCSizeAvg, mainNonCSize


