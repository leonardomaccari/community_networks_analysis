
import code
import scipy

import networkx as nx 
import random as rnd
import matplotlib.pyplot as plt

def showGraph(C, sol=None):
    """ show the graph topology."""

    """ sol is a selection of nodes to be shown with different colors. 
    It can be uses to show MPRs, or the chosen groups for betweennes"""

    weights = zip(*C.edges(data=True))[2]
    edgeColors = []
    for w in weights:
        if w != {}:
            edgeColors.append(1/float(w['weight']))

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
