import networkx as nx
import numpy as np


class SyntheticRandomNetwork:
	def __init__(self, num_nodes, num_edges):
		self.num_nodes = num_nodes
		self.num_edges = num_edges

	def generate(self):
		G = nx.Graph()

		# Add nodes
		for i in range(self.num_nodes):
			G.add_node(i)
		
		# Add edges
		from_node = np.random.random_integers(0, self.num_nodes, self.num_edges)
		to_node = np.random.random_integers(0, self.num_nodes, self.num_edges)
		for i in range(self.num_edges):
			G.add_edge(from_node[i], to_node[i])

		return  G.to_undirected()