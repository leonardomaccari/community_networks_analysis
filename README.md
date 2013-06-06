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
- 


============= License 

All the code is coypright or Leonardo Maccari, released under a GPLv3
License. See the GPL.txt file for details.
