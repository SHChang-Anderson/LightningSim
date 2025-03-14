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
    Generate a simulated Lightning Network topology.
    Parameters:
    - num_nodes: Number of nodes
    - m: Number of connections per new node in BA model
    - mean_capacity: Mean channel capacity (satoshis)
    - median_capacity: Median channel capacity (satoshis)
    Returns:
    - G: NetworkX graph with nodes and edges containing capacity attributes
    """
    G = nx.barabasi_albert_graph(num_nodes, m)
    num_channels = G.number_of_edges()
    capacities = generate_channel_capacities(num_channels, mean_capacity, median_capacity)
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['capacity'] = capacities[i]
    return G

if __name__ == "__main__":
    num_nodes = int(input("set the num_nodes"))
    m = 4  # BA model parameter
    mean_capacity = 200000000  # Mean channel capacity: 20 million satoshis
    median_capacity = 50000000  # Median channel capacity: 5 million satoshis
    file_path = "creditcard.csv"  # CSV file path

    # Generate network
    G = generate_lightning_network(num_nodes, m, mean_capacity, median_capacity)
    nx.write_edgelist(G, "lightning_network.txt", data=True)