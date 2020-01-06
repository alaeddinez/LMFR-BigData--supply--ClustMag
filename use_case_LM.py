# Author: Aric Hagberg (hagberg@lanl.gov)
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd 
G = nx.Graph()
data = pd.read_csv("stype.csv",sep=";")
data.mag1 = data.mag1.astype("str")
data.mag2 = data.mag2.astype("str")
data["coeff"] = data.commun_reappro1 / data.union_product
data_res = data.copy()
#data_res = data_res[(data_res.mag1 == "119") | (data_res.mag2 == "119") ]
for edge in range(0,data_res.shape[0]) :
    G.add_edge(data_res.mag1[edge], data_res.mag2[edge], weight= data_res.coeff[edge] )

data_res = data_res.sort_values(by='coeff', ascending=False)

elarge =  [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.60]
emedium = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] <= 0.60) and (d['weight'] > 0.5)]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

pos = nx.spring_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=200,node_color='g',nodelist=)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,
                       width=0.3,edge_color = 'r')

nx.draw_networkx_edges(G, pos, edgelist=emedium,
                       width=0.0,edge_color = 'w')


nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=0.0, alpha=0, edge_color='w', style='dashed')

# labels
nx.draw_networkx_labels(G, pos, font_size=7, font_family='sans-serif')

plt.axis('off')
plt.show()