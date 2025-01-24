from enhanced_knowledge_graph import EnhancedKnowledgeGraph
from graph_plt import plot_graph
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")  # Set the backend to TkAgg
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

study_file = "data/experiment_1.csv"

# Create an EnhancedKnowledgeGraph object
ekg = EnhancedKnowledgeGraph()
ekg.add_experiment_from_csv(study_file)

# options = {
#     'node_color': 'blue',
#     'node_size': 100,
#     'width': 3,
#     'arrowstyle': '-|>',
#     'arrowsize': 12,
# }
# nx.draw_networkx(ekg.experiment_graph, arrows=True, **options)
# plt.show()

# Define layout
G = ekg.experiment_graph
pos = nx.spring_layout(G)

# Draw graph
plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, arrows=True)

# Extract edge labels
edge_labels = {(u, v): d["relationship_type"] for u, v, d in G.edges(data=True)}

# Draw edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

plt.show()

