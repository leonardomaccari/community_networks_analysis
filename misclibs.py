
import code
import scipy
import sys
import time

import networkx as nx 
import random as rnd
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

def showGraph(C, sol=None):
    """ show the graph topology."""

    """ sol is a selection of nodes to be shown with different colors. 
    It can be uses to show MPRs, or the chosen groups for betweennes"""

    weights = zip(*C.edges(data=True))[2]
    edgeColors = []

    for w in weights:
        if w != {}:
            edgeColors.append(1/float(w['weight']))
        else:
            edgeColors.append(1)

    nodeColors = []
    nodeLabels = {}
    if sol != None:
        mprs = set()
        for s in sol.values():
            if len(s) != 0:
                # this takes an element from a set without removing it
                mprs |= iter(s).next()
  
        for i in C.nodes():
            if i in mprs:
                nodeColors.append('r')
            else :
                nodeColors.append('b')
    else:
        for i in C.nodes():
            if i > 0:
                nodeColors.append('y')
                nodeLabels[i] = str(i)
            else:
                nodeColors.append('r')
                nodeLabels[i] = ""

    ### this is a more printer-friendly version
    #for i in range(len(nodeColors)):
    #    if nodeColors[i] == 'y':
    #        nodeColors[i] = 'black'
    #    if nodeColors[i] == 'r':
    #        nodeColors[i] = 'w'
    #for i in nodeLabels:
    #    nodeLabels[i] = ''
    # #nx.draw_spring(C, width=2, node_color=nodeColors, labels=nodeLabels,
    #        node_size=100)
    #nx.write_dot(C, "/tmp/graph.dot")
    #plt.show()
    #plt.savefig("/tmp/graph.png")
    ###

    if len(edgeColors) != 0:
        nx.draw(C, edge_color=edgeColors, width=4, 
            edge_cmap=plt.cm.Blues, 
            edge_vmin=min(edgeColors), edge_vmax=max(edgeColors),
            node_color=nodeColors, labels=nodeLabels)
    else:
        nx.draw(C, width=4, node_color=nodeColors, labels=nodeLabels)

    #plt.savefig("/tmp/graph.svg")
    #nx.write_adjlist(C, "/tmp/graph.list")
    plt.show()
    
def showMPRs(G, sol):
      plt.show()


def ccdf(sequence, numbins=10, bins=[]):
    """ compute the CCDF over a sequence."""

    counter = 0
    sortedSeq = sorted(sequence)
    if bins == []:
        maxS = sortedSeq[-1]
        for i in range(1,11):
            bins.append(i*float(maxS)/10)
    CCDF = {}
    for i in range(len(bins)):
        while True:
            if counter < len(sortedSeq) and sortedSeq[counter] <= bins[i]:
                counter += 1
            else:
                break
        CCDF[bins[i]] = 1-float(counter)/len(sortedSeq) 
    return CCDF



def launchParallelProcesses(inputValues, parallelism=4, maxLifeTime=-1, 
        targetFunction=None):
    """ launch parallel processing units."""

    """ parallelism : int
                      number of processes in parallel
        maxLifeTime : int
                      number of seconds after which a process is killed
                      leave to -1 to have no max lifetime
        inputValues: iterable 
                     each item is a dict with an 'input' key and an 'output'
                     key. A number of process is launched equal to the lenght of
                     this iterable, each dict is feeded to one process, 
                     the return values are in the 'output' key.

        Return: the same inputValues filled with the output
    """

    if targetFunction == None:
        print >> sys.stderr , "You did not define a function to be called",\
                "by each subprocess"
        sys.exit(1)
    counter = 0
    procList = {}
    killed = 0
    procNumber = len(inputValues)
    print "launching subprocesses"
    queueStack = []
    for i in range(parallelism):
        queueStack.append(Queue())
    while True:
        aliveProc = 0
        toBePurged = []
        for p,v in procList.items():
            if p.is_alive():
                tt = time.time()
                if tt-v[0] < maxLifeTime or maxLifeTime < 0: 
                    aliveProc += 1
                elif maxLifeTime > 0 and tt-v[0] >= maxLifeTime: 
                    print "killed a process after ",\
                        int(tt-v[0]), "seconds"
                    p.terminate()
                    killed += 1
                    for proc in procList:
                        if proc.is_alive():
                            proc.terminate()
                    toBePurged.append(p)
            else:
                # get the results from the corresponding queue
                inputValues[v[2]]['output'] = v[1].get()
                queueStack.append(v[1])
                toBePurged.append(p)

        for proc in toBePurged:
            del procList[proc]

        if aliveProc == 0 and procNumber == 0: 
            break

        if aliveProc < parallelism and procNumber > 0:
            q = queueStack.pop()
            p = Process(target=targetFunction,
                    args=(inputValues[counter],q))

            procList[p] = [time.time(), q, counter]
            p.start()
            procNumber -= 1
            counter += 1
        if aliveProc == parallelism:
            time.sleep(1)

