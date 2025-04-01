import networkx as nx
import numpy as np
import random

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

# Function to generate fees for each channel
def generate_channel_fees_and_rates(num_channels, mean_fee, mean_rate):
    """
    Generate channel fees and fee rates based on normal distributions.
    Parameters:
    - num_channels: Number of channels
    - mean_fee: Mean base fee (satoshis)
    - mean_rate: Mean fee rate (percentage)
    Returns:
    - fees: List of base fees
    - rates: List of fee rates
    """
    # Generate fees (base fee) with a small standard deviation
    fees = np.random.normal(loc=mean_fee, scale=mean_fee * 0.1, size=num_channels)
    fees = np.clip(fees, 0, None)  # Ensure no negative fees

    # Generate fee rates with a small standard deviation
    rates = np.random.normal(loc=mean_rate, scale=mean_rate * 0.1, size=num_channels)
    rates = np.clip(rates, 0, None)  # Ensure no negative rates

    return fees, rates

# Function to generate a scale-free network and assign channel capacities, fees, and rates
def generate_lightning_network(num_nodes, m, mean_capacity, median_capacity, mean_fee, mean_rate):
    """
    Generate a simulated Lightning Network topology with bidirectional edges.
    Parameters:
    - num_nodes: Number of nodes
    - m: Number of connections per new node in BA model
    - mean_capacity: Mean channel capacity (satoshis)
    - median_capacity: Median channel capacity (satoshis)
    - mean_fee: Mean base fee (satoshis)
    - mean_rate: Mean fee rate (percentage)
    Returns:
    - G: NetworkX directed graph with nodes and edges containing capacity, fee, and rate attributes
    """
    # Generate a scale-free network using the Barabási-Albert model
    undirected_graph = nx.barabasi_albert_graph(num_nodes, m)
    directed_graph = nx.DiGraph()

    # Copy nodes to the directed graph
    directed_graph.add_nodes_from(undirected_graph.nodes())

    # Generate capacities, fees, and rates for all edges
    num_channels = undirected_graph.number_of_edges()
    capacities = generate_channel_capacities(num_channels, mean_capacity, median_capacity)
    fees, rates = generate_channel_fees_and_rates(num_channels, mean_fee, mean_rate)

    # Add edges and their reverse edges with capacities, fees, and rates
    for i, (u, v) in enumerate(undirected_graph.edges()):
        capacity = capacities[i]
        fee = fees[i]
        rate = rates[i]
        directed_graph.add_edge(u, v, capacity=capacity, fee=fee, rate=rate)  # Original direction
        directed_graph.add_edge(v, u, capacity=capacity, fee=fee, rate=rate)  # Reverse direction

    return directed_graph

if __name__ == "__main__":
    num_nodes = int(input("Set the number of nodes: "))
    m = 5  # BA model parameter
    mean_capacity = 64040688  # Mean channel capacity: 20 million satoshis
    median_capacity = 900421  # Median channel capacity: 5 million satoshis
    mean_fee = 1.0  # Mean base fee (satoshis)
    mean_rate = 0.00022  # Mean fee rate (percentage)

    # Generate the bidirectional network
    G = generate_lightning_network(num_nodes, m, mean_capacity, median_capacity, mean_fee, mean_rate)

    # Save the network to a file
    nx.write_edgelist(G, "lightning_network.txt", data=True)