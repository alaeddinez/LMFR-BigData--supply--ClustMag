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
#%matplotlib inline
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


# Get the count of interactions between participants and display the top 5 rows.
grp_interactions = pd.DataFrame(interactions.groupby(['participantID.A', 'participantID.B']).size(), 
                                columns=['counts']).reset_index()

grp_interactions.head(5)

nx.to_pandas_edgelist?

# Create a directed graph with an edge_attribute labeled counts.
g = nx.from_pandas_edgelist(grp_interactions, 
                             source='participantID.A', 
                             target='participantID.B', 
                             edge_attr='counts', 
                             create_using=nx.DiGraph())


#undirected graph
# Create a directed graph with an edge_attribute labeled counts.
g = nx.from_pandas_edgelist(grp_interactions, 
                             source='participantID.A', 
                             target='participantID.B', 
                             edge_attr='counts', 
                             create_using=nx.DiGraph())


# Set all the weights to 0 at this stage. We will add the correct weight information in the next step.
G = nx.Graph()
G.add_edges_from(g.edges(), counts=0)


for u, v, d in g.edges(data=True):
    G[u][v]['counts'] += d['counts']

# Print a sample of the edges, with corresponding attribute data.
max_number_of_edges = 5
count = 0
for n1,n2,attr in G.edges(data=True): # unpacking
    print(n1,n2,attr)
    count = count + 1
    if count > max_number_of_edges:
        break

# Verify our attribute data is correct using a selected (u,v) pair from the data.
u = 'fa10-01-77'
v = 'fa10-01-78'
print('Number of undirected call interactions between {0} and {1} is {2}.'.format(u,
                                                                    v,
                                                            G.get_edge_data(v,u)['counts']))

# Compare our data set to the interactions data set.
is_uv_pair = ((interactions['participantID.A'] == u) & (interactions['participantID.B'] == v)) 
is_vu_pair = ((interactions['participantID.A'] == v) & (interactions['participantID.B'] == u))
print('Number of undirected call interactions between {0} and {1} is {2}'.format(u,
                                            v, 
                                            interactions[is_uv_pair | is_vu_pair].shape[0]))                                                      


pos = graphviz_layout(G, prog='dot') # you can also try using the "neato" engine
nx.draw_networkx(G, pos=pos, with_labels=False)
_ = plt.axis('off')

layout = nx.spring_layout(G)
nx.draw_networkx(G, pos=layout, node_color='green', with_labels=False)
_ = plt.axis('off')

# Extract the degree values for all the nodes of G
degrees = []
for (nd,val) in G.degree():
    degrees.append(val)

# Plot the degree distribution histogram.
out = plt.hist(degrees, bins=50)
plt.title("Degree Histogram")
plt.ylabel("Frequency Count")
plt.xlabel("Degree")


# Logarithmic plot of the degree distribution.
values = sorted(set(degrees))
hist = [list(degrees).count(x) for x in values]
out = plt.loglog(values, hist, marker='o')
plt.title("Degree Histogram")
plt.ylabel("Log(Frequency Count)")
plt.xlabel("Log(Degree)")

# Plot degree centrality.
call_degree_centrality = nx.degree_centrality(G)
colors =[call_degree_centrality[node] for node in G.nodes()]
pos = graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos, node_color=colors, node_size=300, with_labels=False)
_ = plt.axis('off')


# Arrange in descending order of centrality and return the result as a tuple, i.e. (participant_id, deg_centrality).
t_call_deg_centrality_sorted = sorted(call_degree_centrality.items(), key=lambda kv: kv[1], reverse=True)

# Convert tuple to pandas dataframe.
df_call_deg_centrality_sorted = pd.DataFrame([[x,y] for (x,y) in t_call_deg_centrality_sorted], 
                                             columns=['participantID', 'deg.centrality'])


# Top 5 participants with the highest degree centrality measure.
df_call_deg_centrality_sorted.head()


# Number of unique actors associated with each of the five participants with highest degree centrality measure.
for node in df_call_deg_centrality_sorted.head().participantID:
    print('Node: {0}, \t num_neighbors: {1}'.format(node, len(list(G.neighbors(node)))))


# Total call interactions are associated with each of these five participants with highest degree centrality measure.
for node in df_call_deg_centrality_sorted.head().participantID:
    outgoing_call_interactions = interactions['participantID.A']==node
    incoming_call_interactions = interactions['participantID.B']==node
    all_call_int = interactions[outgoing_call_interactions | incoming_call_interactions]
    print('Node: {0}, \t total number of calls: {1}'.format(node, all_call_int.shape[0]))

# Plot closeness centrality.
call_closeness_centrality = nx.closeness_centrality(G)
colors = [call_closeness_centrality[node] for node in G.nodes()]
pos = graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos=pos,node_color=colors, with_labels=False)
_ = plt.axis('off')

# Arrange participants according to closeness centrality measure, in descending order. 
# Return the result as a tuple, i.e. (participant_id, cl_centrality).
t_call_clo_centrality_sorted = sorted(call_closeness_centrality.items(), key=lambda kv: kv[1], reverse=True)

# Convert tuple to pandas dataframe.
df_call_clo_centrality_sorted = pd.DataFrame([[x,y] for (x,y) in t_call_clo_centrality_sorted], 
                                             columns=['participantID', 'clo.centrality'])


# Top 5 participants with the highest closeness centrality measure.
df_call_clo_centrality_sorted.head()


# Plot betweenness centrality.
call_betweenness_centrality = nx.betweenness_centrality(G)
colors =[call_betweenness_centrality[node] for node in G.nodes()]
pos = graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos=pos, node_color=colors, with_labels=False)
_ = plt.axis('off')



# Arrange participants according to betweenness centrality measure, in descending order. 
# Return the result as a tuple, i.e. (participant_id, btn_centrality). 
t_call_btn_centrality_sorted = sorted(call_betweenness_centrality.items(), key=lambda kv: kv[1], reverse=True)

# Convert tuple to a Pandas DataFrame.
df_call_btn_centrality_sorted = pd.DataFrame([[x,y] for (x,y) in t_call_btn_centrality_sorted], 
                                             columns=['participantID', 'btn.centrality'])


# Top 5 participants with the highest betweenness centrality measure.
df_call_btn_centrality_sorted.head()

# Plot eigenvector centrality.
call_eigenvector_centrality = nx.eigenvector_centrality(G)
colors = [call_eigenvector_centrality[node] for node in G.nodes()]
pos = graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos=pos, node_color=colors,with_labels=False)
_ = plt.axis('off')

# Arrange participants according to eigenvector centrality measure, in descending order. 
# Return the result as a tuple, i.e. (participant_id, eig_centrality).
t_call_eig_centrality_sorted = sorted(call_eigenvector_centrality.items(), key=lambda kv: kv[1], reverse=True)

# Convert tuple to pandas dataframe.
df_call_eig_centrality_sorted = pd.DataFrame([[x,y] for (x,y) in t_call_eig_centrality_sorted], 
                                             columns=['participantID', 'eig.centrality'])

# Top 5 participants with the highest eigenvector centrality measure. 
df_call_eig_centrality_sorted.head()




# Execute this cell to define a function that produces a scatter plot.
def centrality_scatter(dict1,dict2,path="",ylab="",xlab="",title="",line=False):
    '''
    The function accepts two dicts containing centrality measures and outputs a scatter plot
    showing the relationship between the two centrality measures
    '''
    # Create figure and drawing axis.
    fig = plt.figure(figsize=(7,7))
    
    # Set up figure and axis.
    fig, ax1 = plt.subplots(figsize=(8,8))
    # Create items and extract centralities.
    
    items1 = sorted(list(dict1.items()), key=lambda kv: kv[1], reverse=True)
    
    items2 = sorted(list(dict2.items()), key=lambda kv: kv[1], reverse=True)
    xdata=[b for a,b in items1]
    ydata=[b for a,b in items2]
    ax1.scatter(xdata, ydata)

    if line:
        # Use NumPy to calculate the best fit.
        slope, yint = np.polyfit(xdata,ydata,1)
        xline = plt.xticks()[0]
        
        yline = [slope*x+yint for x in xline]
        ax1.plot(xline,yline,ls='--',color='b')
        # Set new x- and y-axis limits.
        plt.xlim((0.0,max(xdata)+(.15*max(xdata))))
        plt.ylim((0.0,max(ydata)+(.15*max(ydata))))
        # Add labels.
        ax1.set_title(title)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)


# Let us compare call_betweenness_centrality, call_degree_centrality.
centrality_scatter(call_betweenness_centrality, call_degree_centrality, xlab='betweenness centrality',ylab='closeness centrality',line=True)

# Make a (deep) copy of the graph; we work on the copy.
g2 = G.copy()

# Remove the 2 nodes with the highest centrality meaures as discussed above 
g2.remove_node('fa10-01-04')
g2.remove_node('fa10-01-13')

# Recompute the centrality measures.
betweenness2 = nx.betweenness_centrality(g2)
centrality2 = nx.degree_centrality(g2)

# Scatter plot comparison of the recomputed measures.
centrality_scatter(betweenness2, centrality2, 
                   xlab='betweenness centrality',ylab='degree centrality',line=True)

m1 = pd.merge(df_call_btn_centrality_sorted, df_call_clo_centrality_sorted)
m2 = pd.merge(m1, df_call_deg_centrality_sorted)
df_merged  = pd.merge(m2, df_call_eig_centrality_sorted)
df_merged.head()


df_merged.to_csv('centrality.csv', index=False)



#CLUSTERING
# Create a graph object from a provided edges list and visualize the graph. 
e = [(1,2), (2,3), (3,4), (4,5), (4,6), (5,6), (5,7), (6,7)]
g =  nx.Graph()
g.add_edges_from(e)
pos = graphviz_layout(g, prog='dot')
nx.draw_networkx(g, pos=pos, with_labels=True, node_size=500)
_ = plt.axis('off')

# Number of actual edges there are between neigbhors of 5. 
actual_edges = len([(4,6), (6,7)])

# Total possible edges between neighbors of 5.
total_possible_edges = len([(4,6), (6,7), (4,7)])

# Clustering coeff of node. 
local_clustering_coef  = 1.0 * actual_edges / total_possible_edges

print(local_clustering_coef)

# Local clustering for node 5
print((list(nx.clustering(g, nodes=[5]).values())[0]))



# Compute average clustering coefficient for our graph g directly.
print(('(Direct) The average local clustering coefficient of the network is {:0.3f}'.format(np.mean(list(nx.clustering(g).values())))))

# Or using NetworkX.
print(('(NetworkX) The average local clustering coefficient of the network is {:0.3f}'.format(nx.average_clustering(g))))


# Define a function that trims.
def trim_degrees(g, degree=1):
    """
    Trim the graph by removing nodes with degree less than or equal to the value of the degree parameter
    Returns a copy of the graph.
    """
    g2=g.copy()
    d=nx.degree(g2)
    for n in g.nodes():
        if d[n]<=degree: g2.remove_node(n)
    return g2



# Effect of removing weakly connected nodes.
G1 = trim_degrees(G, degree=1)
G3 = trim_degrees(G, degree=3)
G5 = trim_degrees(G, degree=5)

# Compare the clustering coefficient of the resulting network objects.
print((round(nx.average_clustering(G),3), round(nx.average_clustering(G1),3), round(nx.average_clustering(G3),3),
round(nx.average_clustering(G5),3)))