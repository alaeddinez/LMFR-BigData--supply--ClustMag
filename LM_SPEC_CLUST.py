import pandas as pd 
import numpy as np
data = pd.read_csv("stype.csv", sep=";")
data.mag1 = data.mag1.astype("str")
data.mag2 = data.mag2.astype("str")
data["coeff"] = data.commun_reappro1 / data.union_product
data_res = data.copy()
#data_res.coeff[data_res.coeff > .66] = 1
#data_res.coeff[data_res.coeff <= .66] = 0
list_mag = np.unique([data_res.mag1,data_res.mag2])

df_comp = data_res.copy()
df_comp = df_comp[0:0]
df_comp.mag1 = list_mag
df_comp.mag2 = list_mag
df_comp.coeff = 0


data_res = data_res.append(df_comp, ignore_index=True)
A = data_res.pivot_table(columns='mag1', index='mag2', values='coeff')
A = A.fillna(0)
A = A.values
from sklearn.cluster import KMeans


# our adjacency matrix
print("Adjacency Matrix:")
print(A)

# diagonal matrix
D = np.diag(A.sum(axis=1))

# graph laplacian
L = D-A

# eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(L)

#get rid of complex part  (to test !!)
vals = vals.real
vecs = vecs.real

# sort these based on the eigenvalues
vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]







index_largest_gap = np.argsort(np.diff(vals))[::-1][:20]
nb_clusters = index_largest_gap + 1




# kmeans on first two vectors with nonzero eigenvalues
kmeans = KMeans(n_clusters= min(nb_clusters))
kmeans.fit(vecs[:,1:3])
colors = kmeans.labels_
print("Clusters:", colors)





RES  = pd.DataFrame()
RES["mag"] = list_mag
RES["cluster"] = colors

RES.to_csv("res.csv",sep=";",index=False)
##########################
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd 
G = nx.Graph()

#data_res = data_res[(data_res.mag1 == "119") | (data_res.mag2 == "119") ]
for edge in range(0,data_res.shape[0]) :
    G.add_edge(data_res.mag1[edge], data_res.mag2[edge], weight= data_res.coeff[edge] ,
    node_color = data_res.mag1[edge]
    )

data_res = data_res.sort_values(by='coeff', ascending=False)

elarge =  [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.70]
emedium = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] <= 0.70) and (d['weight'] > 0.5)]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

pos = nx.spring_layout(G)  # positions for all nodes



# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,
                       width=0.4,edge_color = 'r')

nx.draw_networkx_edges(G, pos, edgelist=emedium,
                       width=0.01,edge_color = 'b')


nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=0.0, alpha=0, edge_color='w', style='dashed')

# labels
nx.draw_networkx_labels(G, pos, font_size=7, font_family='sans-serif')

carac = RES.copy()
carac= carac.set_index('mag')

carac=carac.reindex(G.nodes())

carac['cluster']=pd.Categorical(carac['cluster'])
carac['cluster'].cat.codes
 
# nodes
nx.draw_networkx_nodes(G, pos, node_size=150,node_color=carac.cluster.cat.codes)
#nx.draw(G, with_labels=True, node_color=carac.cluser.cat.codes, node_size = 500)


plt.axis('off')
plt.show()

plt.savefig('graph.png')




