# Load the relevant libraries to your notebook. 
import pandas as pd            # Processing csv files and manipulating the DataFrame.
import networkx as nx          # Graph-like object representation and manipulation module.
import matplotlib.pylab as plt # Plotting and data visualization module. 
                               # This is used for basic graph visualization.
import numpy as np       

from networkx.drawing.nx_agraph import graphviz_layout

import random

from IPython.display import Image, display

# Set global parameters for plotting. 
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 8)


def pydot(G):
    pdot = nx.drawing.nx_pydot.to_pydot(G)
    display(Image(pdot.create_png()))

# Read the CallLog.csv file, print the number of records loaded as well as the first 5 rows.
calls = pd.read_csv("/home/alaeddinez/Desktop/projects/LMFR-BigData--supply--ClustMag/MIT networkx/CallLog.csv")
print('Loaded {0} rows of call log.'.format(len(calls)))
calls.head()

# Initial number of records.
calls.shape[0]
calls.info()

# Drop rows with NaN in either of the participant ID columns.
calls = calls.dropna(subset = ['participantID.A', 'participantID.B'])
print('{} rows remaining after dropping missing values from selected columns.'.format(len(calls)))
calls.head(n=5)


# Create a new object containing only the columns of interest.
interactions = calls[['participantID.A', 'participantID.B']]


# Get a list of rows with different participants.
row_with_different_participants = interactions['participantID.A'] != interactions['participantID.B']

# Update "interactions" to contain only the rows identified. 
interactions = interactions.loc[row_with_different_participants,:]
interactions.head()



# Create an unweighted undirected graph using the NetworkX's from_pandas_edgelist method.
# The column participantID.A is used as the source and participantID.B as the target.
G = nx.from_pandas_edgelist(interactions, 
                             source='participantID.A', 
                             target='participantID.B', 
                             create_using=nx.Graph())


# Print the number of nodes in our network.
print('The undirected graph object G has {0} nodes.'.format(G.number_of_nodes()))

# Print the number of edges in our network.
print('The undirected graph object G has {0} edges.'.format(G.number_of_edges()))


# Declare a variable for number of nodes to get neighbors of.
max_nodes = 5



# Variable initialization.
count = 0
ndict = {}

# Loop through G and get a node's neigbours, store in ndict. Do this for a maximum of 'max_nodes' nodes. 
for node in list(G.nodes()):
    ndict[node] = tuple(G.neighbors(node))
    count = count + 1
    if count > max_nodes:
        break

print(ndict)


# Print only the first item in the dict.
print([list(ndict)[0], ndict[list(ndict)[0]]])

