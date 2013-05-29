community_networks_analysis
===========================

some python script to inspect the network graphs of community networks


This scripts relies on networkX library, currently tested on 1.6.2.

parse_network.py: main files, command line parser, test functions
graphAnalyzer.py: the routine that calls groupMetrics.py to produce some
                  data
groupMetrics.py: extract group betweenness and closeness from a graph,
                 with a brute force approach
genGraph.py: generate some common graph topologies to make tests

publicdata/tests.edges: a fragment of the ninux.org topology

usage:
 ./parse_network.py -f publicdata/test.edges -e 
