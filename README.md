community_networks_analysis
===========================

python scripts to inspect the network graphs of community networks

The first release is able to compute the group betweenness and closeness
centrality with an exhaustive search over all the possible groups. This
scales up to group size of 4/5 nodes on a network of 110 nodes. 
Future versions may have some more optimized algorithms in order to
scale on larger sizes. 



This scripts relies on networkX library, currently tested on 1.6.2.

usage:
 ./parse_network.py -f publicdata/test.edges -e 

this will output some debug stuff and a table with three columns:
groupsize, betweenness, closeness in a gnuplot-parsable table. In the
table, as comments also the groups are reported

This release contains the first version of a code that will be used to
submit a paper to the CNBUB2013 workshop. It contains the following
files:

- parse_network.py: the main file with CL parsing and some test functions
- miscLibs.py: some helper functions to draw the graph, compute ccdn
               etc...
- groupMetrics.py: the functions needed to compute the group betweenness
                   and group closeness centrality.
- genGraphs.py: graph loader and generator
- graphAnalyzer.py: this is the place were you should ghange your code to
                    use the functions defined in other libraries. Some
                    of its current functions will be improved and moved
                    into libraries.
- smalldata/tests.edges: a fragment of the ninux.org topology
 


BugFixes and Updates:

============= Version 0.1 : January 2014

This version adds two features and refactors some code:
 - Now the centrality to the HNA gateways can be computed. This is 
   important if one wants to identify the nodes that have a high
   betweenness considering only the routes to a gateway. It gives
   a measure of how robust the network is to interception of traffic
   when it is used as an access network to reach the Internet
 - A new group centrality algorithm has been added. The previous 
   algorithm checks the centrality of every group of nodes of size
   k. The number of the groups of size k scales approximately ~ as N!
   in a network of size N, which is not manageable for large networks.
   The new algorithm is based on a greedy algorithm published in 
   http://www.cs.bu.edu/fac/best/res/papers/sdm12.pdf
   that I have modified implementing a simple GRASP variation (see 
   http://en.wikipedia.org/wiki/Greedy_randomized_adaptive_search_procedure)
   Now the solution may not be a global optimum but the code can handle 
   networks of 250 nodes with groups up to 5 nodes in less than one 
   minute on COTS hardware. More on this algorithm will be detailed
   in future publications
 - I have refactored the multi-process code to be easier to 
   use for generic purposes. 
   
This version of the code has been developed with the contribution
of Luca Baldesi and Prof. Renato Lo Cigno, from the DISI, University 
of Trento. Their support has been partially funded by EIT ICTLabs
Activity 12180 'Smart Ubiquitous Content' 

============= License 

All the code is coypright or Leonardo Maccari, released under a GPLv3
License. See the GPL.txt file for details.
