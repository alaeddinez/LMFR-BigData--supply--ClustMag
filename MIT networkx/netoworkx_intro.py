##https://github.com/getsmarter/bda/blob/master/module_4/M4_NB1_NetworkX_Introduction.ipynb

# Load relevant libraries.
import networkx as nx  
import matplotlib.pylab as plt
%matplotlib inline
import pygraphviz as pgv
import random
from IPython.display import Image, display




# Instantiate an empty network undirected graph object, and assign to variable G.
G=nx.Graph()

# Add a node (1) to G.
G.add_node(1)


# Add another node ('x') to G.
G.add_node('x')



def pydot(G):
    '''
    A function for graph visualization using the dot framework
    '''
    pdot = nx.drawing.nx_pydot.to_pydot(G)
    display(Image(pdot.create_png()))

pydot(G)


# Add an edge between two nodes, 1 and 3. 
# Note that nodes are automatically added if they do not exist.
G.add_edge(1,3)



# Add edge information, and specify the value of the weight attribute.
G.add_edge(2,'x',weight=0.9)
G.add_edge(1,'x',weight=3.142)



# Add edges from a list of tuples.
# In each tuple, the first 2 elements are nodes, and third element is value of the weight attribute. 
edgelist=[('a','b',5.0),('b','c',3.0),('a','c',1.0),('c','d',7.3)] 
G.add_weighted_edges_from(edgelist)


# Visualize the graph object.
pydot(G)




# Visualize the graph object, including weight information.
for u,v,d in G.edges(data=True):
    d['label'] = d.get('weight','')
pydot(G)


# Add a sine function, imported from the math module, as a node.
from math import sin
G.add_node(sin)


# Add file handle object as node.
fh = open("/home/alaeddinez/Desktop/projects/LMFR-BigData--supply--ClustMag/MIT networkx/CallLog.csv","r") # handle to file object.
G.add_node(fh)


# List the nodes in your graph object.
list(G.nodes())


# How many nodes are contained within your graph model?
G.number_of_nodes()


# Alternative method for getting the number nodes.
G.order()

# List the edges in the graph object.
list(G.edges())

# How many edges do you have? 
G.number_of_edges()



# Alternative method for getting number of edges.
G.size()

for (u, v, wt) in G.edges.data('weight'):
    if wt != None:
        print('(%s, %s, %.3f)' % (u, v, wt))
    if wt is None:
        print(u,v, wt)


for n1,n2,attr in G.edges(data=True):
         print(n1,n2,attr)

list(G.neighbors('x'))


for node in G.nodes():
         print(node, list(G.neighbors(node)))

# Add a set of edges from a list of tuples.
e = [(1 ,2) ,(1 ,3)]
G.add_edges_from(e)
# Remove edge (1,2).
G.remove_edge(1,2)
# Similarly, you can also remove a node, and all edges linked to that node will also fall away.
G.remove_node(3)
# Multiple edge or node removal is also possible.
G.remove_edges_from(e)
# Trying to remove a node not in the graph raises an error.
G.remove_node(3)
# Close the file handle object used above.
fh.close()

# Remove the graph object from the workspace.
del G


# Generate some of the small, famous graphs.
petersen=nx.petersen_graph()
tutte=nx.tutte_graph()
maze=nx.sedgewick_maze_graph()
tet=nx.tetrahedral_graph()


# Generate some classic graphs.
K_5=nx.complete_graph(5)
K_3_5=nx.complete_bipartite_graph(3,5)
barbell=nx.barbell_graph(10,10)
lollipop=nx.lollipop_graph(10,20)


