#! /usr/bin/python

import sys
import random
import getopt
import time
import os
import math
import scipy.stats 

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Process, Queue

from genGraphs import *
from groupMetrics import *
from graphAnalyzer import *



def parseArgs():
    """ argument parser."""
    lfile = []
    showGraph = False
    extractData = False
    testRun = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:set")
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) 
        usage()
        sys.exit(2)
    for option,v in opts:
        if option == "-f":
            lfile.append(v)
        elif option == "-s":
            showGraph = True
        elif option == "-e":
            extractData = True
        elif option == "-t":
            testRun = True
        else:
            assert False, "unhandled option"

    if lfile == [] and testRun == False:
        usage()
        sys.exit(1)
    return  configuration(lfile, s = showGraph, e=extractData,
            t=testRun)

def usage():
    """ print the allowed arguments. """
    print >> sys.stderr, "Analyze the community-network graph"
    print >> sys.stderr, "usage:", 
    print >> sys.stderr, "./parse_ninux.py:"
    print >> sys.stderr, " -f graph\
            (adjacency list (.adj) or edge list (.edges) as used by networkX)"
    print >> sys.stderr, " [-s] show graph" 
    print >> sys.stderr, " [-t] run a test with known graphs" 


class configuration():
    """ configuration parameters storage class."""
    fileList = []
    showGraph = False
    extractData = True
    testRun = False
    libPath = "./libs/"
    def __init__(self, fileList, s=False, e=False, t=False):
        self.fileList = fileList
        self.showGraph = s
        self.extractData = e
        self.testRun = t

def testRun():
    """ run groupMetrics on a known set of graphs, check the results.""" 
    print >> sys.stderr, "Testing group betweenness centrality"
    L = genGraph("LINEAR", 3)
    betw, solB, clos, solC, s = computeGroupMetrics(L, 1, weighted = False, 
            cutoff = 2)
    ok = True

    if solB[0] != set([1]) or betw != 1:
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong betweenness on the 3-nodes line network: ", solB[0], betw
        ok = False
    if solC[0] != set([1]) or clos != 1:
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong closeness on the 3-nodes line network: ", solC[0], clos
        ok = False

    L = genGraph("LINEAR", 5)
    betw, solB, clos, solC, s  = computeGroupMetrics(L, 1, weighted = False, 
            cutoff = 2)

    if solB[0] != set([2]) or betw != 1:
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong betweenness on the 5-nodes line network: ", solB[0], betw
        ok = False
    if solC[0] != set([2]) or clos != 1.5:
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong closeness on the 5-nodes line network: ", solC[0], clos
        ok = False

    L = genGraph("LINEAR", 8)
    betw, solB, clos, solC, s  = computeGroupMetrics(L, 2, weighted = False, 
            cutoff = 2)

    if solB[0] != set([2,5]) or betw != 1:
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong betweenness on the 8-nodes line network: ", solB[0], betw
        ok = False
    if solC[0] not in [set([1, 6]), set([2, 5]), set([1, 5]), set([2,6])]\
            or str(clos)[0:3] != "1.3":
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong closeness on the 8-nodes line network: ", solC[0], \
            str(clos)[0:3]
        ok = False


    L = genGraph("MESH", 9)
    betw, solB, clos, solC, s  = computeGroupMetrics(L, 1, weighted = False, 
            cutoff = 2)
    if solB[0] != set([4]) :
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong betweenness on the 9-nodes grid network: ", solB[0], betw
        ok = False
    if solC[0] != set([4]) or str(clos)[0:3] != "1.0":
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong closeness on the 9-nodes line network: ", solC[0], \
            str(clos)[0:3]
        ok = False

    M = genGraph("MESH", 9)
    #make a mesh that makes node 1 more appealing than node 4
    for src,dest,data in M.edges(data=True):
        if src == 1 or dest == 1:
            data['weight'] = 1
        else:
            data['weight'] = 10
    betw, solB, clos, solC, s  = computeGroupMetrics(M, 1, weighted = True, 
            cutoff = 2)
    if solB[0] != set([1]) :
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong betweenness on the 9-nodes mesh weigthed network: ",\
            solB[0], betw
        ok = False
    if solC[0] != set([1]) or str(clos)[0:4] != "4.75":
        print >> sys.stderr , " ERROR: computeGroupMetrics is giving",\
            "wrong closeness on the 9-nodes mesh weigthed network: ", \
            solC[0], str(clos)[0:4]
        ok = False

    if not ok:
        sys.exit(1)
    else:
        print >> sys.stderr, "********** All tests run ok **********"



def showGraph(C):
    """ show the graph topology."""
    weights = zip(*C.edges(data=True))[2]
    colors = []
    for w in weights:
        if w != {}:
            colors.append(1/float(w['weight']))
    if len(colors) != 0:
        nx.draw(C, edge_color=colors, width=4, 
            edge_cmap=plt.cm.Blues, 
            edge_vmin=min(colors), edge_vmax=max(colors))
    else:
        nx.draw(C, width=4)
    #plt.savefig("/tmp/graph.svg")
    #nx.write_adjlist(C, "/tmp/graph.list")
    plt.show()

if __name__ == '__main__':
    graphArray = []
    conf = parseArgs()
    if conf.testRun:
        testRun()
        sys.exit()
    if conf.fileList != []:
        for fname in conf.fileList:
            # load a file using networkX adjacency matrix structure
            C = loadGraph(fname)
            if conf.showGraph == True:
                showGraph(C)
            if conf.extractData == True:
                extractData(C)


