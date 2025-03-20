import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import ast  # For safely parsing dictionary strings
import re  # For regular expressions

# Function to generate channel capacities based on a log-normal distribution
def generate_channel_capacities(num_channels, mean_capacity, median_capacity):
    """
    Generate channel capacities based on a log-normal distribution.
    Parameters:
    - num_channels: Number of channels
    - mean_capacity: Mean channel capacity (satoshis)
    - median_capacity: Median channel capacity (satoshis)
    Returns:
    - capacities: List of channel capacities
    """
    mu = np.log(median_capacity)
    sigma = np.sqrt(2 * (np.log(mean_capacity) - mu))
    capacities = np.random.lognormal(mean=mu, sigma=sigma, size=num_channels)
    capacities = np.round(capacities).astype(int)
    return capacities

# Function to generate a scale-free network and assign channel capacities
def generate_lightning_network(num_nodes, m, mean_capacity, median_capacity):
    """
    Generate a simulated Lightning Network topology with bidirectional edges.
    Parameters:
    - num_nodes: Number of nodes
    - m: Number of connections per new node in BA model
    - mean_capacity: Mean channel capacity (satoshis)
    - median_capacity: Median channel capacity (satoshis)
    Returns:
    - G: NetworkX directed graph with nodes and edges containing capacity attributes
    """
    # Generate a scale-free network using the Barabási-Albert model
    undirected_graph = nx.barabasi_albert_graph(num_nodes, m)
    directed_graph = nx.DiGraph()

    # Copy nodes to the directed graph
    directed_graph.add_nodes_from(undirected_graph.nodes())

    # Generate capacities for all edges
    num_channels = undirected_graph.number_of_edges()
    capacities = generate_channel_capacities(num_channels, mean_capacity, median_capacity)

    # Add edges and their reverse edges with capacities
    for i, (u, v) in enumerate(undirected_graph.edges()):
        capacity = capacities[i]
        directed_graph.add_edge(u, v, capacity=capacity)  # Original direction
        directed_graph.add_edge(v, u, capacity=capacity)  # Reverse direction

    return directed_graph

if __name__ == "__main__":
    num_nodes = int(input("Set the number of nodes: "))
    m = 5  # BA model parameter
    mean_capacity = 64040688  # Mean channel capacity: 20 million satoshis
    median_capacity = 900421  # Median channel capacity: 5 million satoshis

    # Generate the bidirectional network
    G = generate_lightning_network(num_nodes, m, mean_capacity, median_capacity)

    # Save the network to a file
    nx.write_edgelist(G, "lightning_network.txt", data=True)