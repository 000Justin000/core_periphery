import networkx as nx 

G = nx.read_gml('dolphins.gml')
H = nx.convert_node_labels_to_integers(G)
nx.write_edgelist(H,'dolphins.txt')
print(nx.info(H))
