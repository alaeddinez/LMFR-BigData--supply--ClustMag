"""
An example of drawing a weighted graph using the NetworkX module
This is sample code and not indicative of how Qxf2 writes Python code

---------------
I. The problem:
---------------
I will be plotting how often these four world chess champions played each other:
a) Anatoly Karpov
b) Gary Kasparov
c) Vladimir Kramnik
d) Vishwanathan Anand

-------------------------
II. Technical references: 
-------------------------
1. https://networkx.github.io/documentation/networkx-1.9/examples/drawing/weighted_graph.html
2. https://stackoverflow.com/questions/28372127/add-edge-weights-to-plot-output-in-networkx

-----------------------------------------
III. Reference for data (as of Aug 2017):
-----------------------------------------
1. Karpov - Kasparov: 170 classical games
http://www.chessgames.com/perl/chess.pl?pid=15940&amp;pid2=20719

2. Karpov - Kramnik: 15 classical games
http://www.chessgames.com/perl/chess.pl?pid=20719&amp;pid2=12295

3. Karpov - Anand: 45 classical games
http://www.chessgames.com/perl/chess.pl?pid=20719&amp;pid2=12088

4. Kasparov - Kramnik: 49 classical games
http://www.chessgames.com/perl/chess.pl?pid=12295&amp;pid2=15940

5. Kasparov - Anand: 51 classical games
http://www.chessgames.com/perl/chess.pl?pid=12088&amp;pid2=15940

6. Kramnik - Anand: 91 classical games
http://www.chessgames.com/perl/chess.pl?pid=12295&amp;pid2=12088
"""

#1. Import pyplot and nx
import matplotlib.pyplot as plt
import networkx as nx


def plot_weighted_graph():
    "Plot a weighted graph"

    #2. Add nodes
    G = nx.Graph() #Create a graph object called G
    node_list = ['Karpov','Kasparov','Kramnik','Anand']
    for node in node_list:
        G.add_node(node)

    #Note: You can also try a spring_layout
    pos=nx.circular_layout(G) 
    nx.draw_networkx_nodes(G,pos,node_color='green',node_size=7500)

    #3. If you want, add labels to the nodes
    labels = {}
    for node_name in node_list:
        labels[str(node_name)] =str(node_name)
    nx.draw_networkx_labels(G,pos,labels,font_size=16)


    #4. Add the edges (4C2 = 6 combinations)
    #NOTE: You usually read this data in from some source
    #To keep the example self contained, I typed this out
    G.add_edge(node_list[0],node_list[1],weight=170) #Karpov vs Kasparov
    G.add_edge(node_list[0],node_list[2],weight=15) #Karpov vs Kramnik
    G.add_edge(node_list[0],node_list[3],weight=45) #Karpov vs Anand
    G.add_edge(node_list[1],node_list[2],weight=49) #Kasparov vs Kramnik
    G.add_edge(node_list[1],node_list[3],weight=51) #Kasparov vs Anand
    G.add_edge(node_list[2],node_list[3],weight=91) #Kramnik vs Anand

    all_weights = []
    #4 a. Iterate through the graph nodes to gather all the weights
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) #we'll use this when determining edge thickness

    #4 b. Get unique weights
    unique_weights = list(set(all_weights))

    #4 c. Plot the edges - one by one!
    for weight in unique_weights:
        #4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        #4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
        width = weight*len(node_list)*3.0/sum(all_weights)
        nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width)

    #Plot the graph
    plt.axis('off')
    plt.title('How often have they played each other?')
    plt.savefig("chess_legends.png") 
    plt.show() 

#----START OF SCRIPT
if __name__=='__main__':
    plot_weighted_graph()