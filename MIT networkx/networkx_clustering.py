"""

https://github.com/https://github.com/getsmarter/bda/blob/master/module_4/call.adjmatrixgetsmarter/bda/blob/master/module_4/M4_NB3_NetworkClustering.ipynb
Community detection is an important task in social network analysis. The idea behind it is to identify groups of people that share a common interest, based on the assumption that these people tend to link to each other more than to the rest of the network. Specifically, real-world networks exhibit clustering behavior that can be observed in the graph representation of these networks by the formation of clusters or partitions. These groups of nodes on a graph (clusters) correspond to communities that share common properties, or have a common role in the system under study.

Intuitively, it is expected that such clusters are associated with a high concentration of nodes. In the following examples, you will explore the identification of these clusters using the following approaches, as discussed in the video content:

    Hierarchical clustering (using a distance matrix)
    The Louvain Algorithm (using modularity maximization)
    Spectral graph partitioning


"""
#require to install python-louvain
import networkx as nx
import pandas as pd
import numpy as np
import pygraphviz

%matplotlib inline
import matplotlib.pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout

from collections import defaultdict, Counter
import operator

## For hierarchical clustering.
from scipy.cluster import hierarchy
from scipy.spatial import distance

## For spectral graph partitioning.
from sklearn.cluster import spectral_clustering as spc

## For Community Detection (Louvain Method).
import community
import sys
#from utils import draw_partitioned_graph
#from utils import fancy_dendrogram

def fancy_dendrogram(*args, **kwargs):
    '''
    Source: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    '''
    from scipy.cluster import hierarchy
    import matplotlib.pylab as plt
    
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = hierarchy.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def draw_partitioned_graph(G, partition_obj, layout=None, labels=None,layout_type='spring', 
               node_size=70, node_alpha=0.7, cmap=plt.get_cmap('jet'),
               node_text_size=12,
               edge_color='blue', edge_alpha=0.5, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # if a premade layout haven't been passed, create a new one
    if not layout:
        if graph_type == 'spring':
            layout=nx.spring_layout(G)
        elif graph_type == 'spectral':
            layout=nx.spectral_layout(G)
        elif graph_type == 'random':
            layout=nx.random_layout(G)
        else:
            layout=nx.shell_layout(G)

    # prepare the partition list noeds and colors

    list_nodes, node_color = partition_to_draw(partition_obj)
      
    # draw graph
    nx.draw_networkx_nodes(G,layout,list_nodes,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color, cmap = cmap)
    nx.draw_networkx_edges(G,layout,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    #nx.draw_networkx_labels(G, layout,font_size=node_text_size,
    #                        font_family=text_font)

    if labels is None:
        labels = range(len(G))

    edge_labels = dict(zip(G, labels))
    #nx.draw_networkx_edge_labels(G, layout, edge_labels=edge_labels, 
    #                            label_pos=edge_text_pos)

    # show graph

    plt.axis('off')
    plt.xlim(0,1)
    plt.ylim(0,1)


plt.rcParams['figure.figsize'] = (15, 9)
plt.rcParams['axes.titlesize'] = 'large'

call_adjmatrix = pd.read_csv('https://raw.githubusercontent.com/getsmarter/bda/master/module_4/call.adjmatrix', index_col=0)
call_graph     = nx.from_numpy_matrix(call_adjmatrix.as_matrix())


# Display call graph object.
plt.figure(figsize=(10,10))
plt.axis('off')

pos = graphviz_layout(call_graph, prog='dot')
nx.draw_networkx(call_graph, pos=pos, node_color='#11DD11', with_labels=False)
_ = plt.axis('off')

##HIERACHICAL CLUSTERING 


def create_hc(G, linkage='average'):
    """
    Creates hierarchical cluster of graph G from distance matrix
    """ 
    
    path_length=nx.all_pairs_shortest_path_length(G)
    distances=np.zeros((G.order(),G.order())) 
    
    for u,p in dict(path_length).items():
        for v,d in p.items():
            distances[list(G.nodes)[u]][list(G.nodes)[v]] = d
            distances[list(G.nodes)[v]][list(G.nodes)[u]] = d
            if u==v: 
                distances[list(G.nodes)[u]][list(G.nodes)[u]]=0
    # Create hierarchical cluster (HC).
    Y=distance.squareform(distances)
    if linkage == 'max':
        # Creates HC using farthest point linkage.
        Z=hierarchy.complete(Y)  
    if linkage == 'single':
        # Creates HC using closest point linkage.
        Z=hierarchy.single(Y)  
    if linkage == 'average':
        # Creates HC using average point linkage.
        Z=hierarchy.average(Y)
        
    return Z

def get_cluster_membership(Z, maxclust):
    '''
    Assigns cluster membership by specifying cluster size.
    '''
    hc_out=list(hierarchy.fcluster(Z,maxclust, criterion='maxclust'))
    
    # Map cluster values to a dictionary variable.
    cluster_membership = {}
    i = 0
    for i in range(len(hc_out)):
        cluster_membership[i]=hc_out[i]
    
    return cluster_membership

# Perform hierarchical clustering using 'average' linkage. 
Z = create_hc(call_graph, linkage='average')



hierarchy.dendrogram(Z)
plt.show()


plt.title('Hierarchical Clustering Dendrogram (pruned)')
plt.xlabel('sample index (or leaf size)')
plt.ylabel('distance')
hierarchy.dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=10,                   # show only the last p merged clusters
    show_leaf_counts=True,  # numbers in brackets are counts for each leaf
    leaf_rotation=90,
    leaf_font_size=12)
plt.show()


