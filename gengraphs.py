import networkx as nx
import sys
import random 
import numpy as np

linearGraph = "LINEAR"
ringGraph = "RING"
unitDisk = "UNIT"
gridGraph = "GRID"
meshGraph = "MESH" # grid with diagonals
regularGraph = "REGULAR"
plainGrid = "TEST"
powerLaw = "PLAW"
allowedGraphs = [linearGraph, unitDisk, gridGraph, regularGraph,\
        plainGrid, ringGraph, powerLaw, meshGraph]



def loadGraph(fname, remap=False, connected=True):
    """ Parameters
    --------------
    fname : string
        filname to open
    remap : bool
        remap the labels to a sequence of integers 
    connected : bool
        return only the larges component subgraph

    """
    G=nx.Graph()
    print "Loading/Generating Graph"
    # load a file using networkX adjacency matrix structure
    if fname.lower().endswith(".adj"):
        try:
            G = nx.read_adjlist(fname, nodetype=int)
        except IOError as err:
            print
            print err
            sys.exit(1)
    elif fname.lower().endswith(".edges"):
        try:
            G = nx.read_weighted_edgelist(fname, nodetype=int)
        except IOError as err:
            print
            print err
            sys.exit(1)
    else:
        print >> sys.stderr, "Error: Allowed file extensions are .adj for",\
            "adjacency matrix and .edges for edge-list"
        sys.exit(1) 
    if connected:
        C = nx.connected_component_subgraphs(G)[0]
        G = C
    print >> sys.stderr, "Graph", fname, "loaded",
    # remap node labels so we don't have "holes" in the numbering
    if remap:
        mapping=dict(zip(G.nodes(),range(G.order())))
        H = nx.relabel_nodes(G,mapping)
        return H 
    return G

def genGraph(graphKind, numNodes):
    """ Parameters
    --------------
    graphKind: string
    numNodes: integer

    """
    G=nx.Graph()
    print "Loading Graph"
    if graphKind == linearGraph:
        G.add_nodes_from(range(0,numNodes))
        for i in range(0,numNodes-1):
            G.add_edge(i,i+1)
    elif graphKind == ringGraph:
        G.add_nodes_from(range(0,numNodes))
        for i in range(0,numNodes):
            G.add_edge(i,(i+1)%numNodes)
    elif graphKind == unitDisk:
        r = 70
        # 90 nodes with 400*400 is ok, try to keep the same density
        #density = 90.0/(400*400)
        #area = numNodes/density
        #xSize = np.sqrt(area)

        xSize = 150 # use this to keep area fixed
        w = dict((i,r/2) for i in range(numNodes))
        for i in range(1000):
            random.jumpahead(i)
            p = dict((i,(random.uniform(0,xSize),random.uniform(0,xSize))) 
                    for i in range(numNodes)) 
            G = nx.geographical_threshold_graph(numNodes,theta=1,alpha=1,
                    pos=p, weight=w)
            if nx.is_connected(G):
                break
        if not nx.is_connected(G):
            print >> sys.stderr, "Could not find a connected graph \
                    with the given features in 1000 attempts!"
            sys.exit(1)
    elif graphKind == regularGraph:
        degree = 6
        G= nx.random_regular_graph(degree,numNodes)
    elif graphKind == gridGraph or graphKind == meshGraph:
        if graphKind == meshGraph:
            radius = 90.0 # with diagonals
        else:
            radius = 70.0  # without diagonals
        side = int(np.sqrt(numNodes))
        if side*side != numNodes:
            print >> sys.stderr, "Error, you want a squared \
                    grid with",numNodes,"nodes, it's not a square"
            sys.exit(1)
        distance = 60
        positions = {}
        w = {}
        for i in range(numNodes):
            positions[i] = (distance*(i%side), distance*(i/side))
            w[i] = radius/2
        G = nx.geographical_threshold_graph(numNodes, theta=1,
                alpha=1, pos=positions, weight=w)
        if not nx.is_connected(G):
            print >> sys.stderr, "Error, something is \
                    wrong with the graph definition"
            sys.exit(1)
    elif graphKind == plainGrid:
        side = int(np.sqrt(float(numNodes)))
        G = nx.grid_2d_graph(side, side)
        numNodes = side*side
    elif graphKind == powerLaw:
        gamma=2
        powerlaw_gamma = lambda x: nx.powerlaw_sequence(x, exponent=gamma)
        loop = True
        for i in range(1000):
            z = nx.utils.create_degree_sequence(numNodes, 
                    powerlaw_gamma, max_tries=5000)
            G = nx.configuration_model(z)
            G = nx.Graph(G)
            G.remove_edges_from(G.selfloop_edges())
            mainC = nx.connected_component_subgraphs(G)[0]
            if len(mainC.nodes()) >= numNodes*0.9:
                loop = False
                break
        if loop:
            print "ERROR: generating powerLow graph with \
                    impossible parameters"
            sys.exit()
    else: 
        errMsg = "Unknown graph type " + graphKind 
        print >> sys.stderr, errMsg 
        sys.exit(1)
    
    print >> sys.stderr, "ok"
    return G

