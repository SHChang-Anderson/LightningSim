import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Set random seed for reproducibility
np.random.seed(42)

# Function to randomly select a sender and receiver
def random_sender_receiver(G):
    nodes = list(G.nodes())  # Get a list of all nodes
    sender = random.choice(nodes)  # Randomly select sender
    receiver = random.choice(nodes)  # Randomly select receiver
    while sender == receiver:  # Ensure sender and receiver are different
        receiver = random.choice(nodes)
    return sender, receiver

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

# Function to simulate a payment
def simulate_payment(G, sender, receiver, amount):
    """
    Simulate a payment in the Lightning Network.
    Parameters:
    - G: NetworkX graph
    - sender: Sender node
    - receiver: Receiver node
    - amount: Payment amount (satoshis)
    Returns:
    - success: Whether the payment was successful
    - path: Payment path (if successful)
    """
    try:
        path = nx.shortest_path(G, sender, receiver)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G[u][v]['capacity'] < amount:
                return False, []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            G[u][v]['capacity'] -= amount
        return True, path
    except nx.NetworkXNoPath:
        return False, []

# Function to load payment amounts from creditcard.csv
def load_payment_amounts(file_path, num_payments):
    """
    Load transaction amounts from creditcard.csv and map them to payment simulation.
    Parameters:
    - file_path: CSV file path
    - num_payments: Number of payments to simulate
    Returns:
    - amounts: List of payment amounts (satoshis)
    """
    df = pd.read_csv(file_path)
    amounts = df['Amount'].head(num_payments).values * 100000  # Convert to satoshis (assuming 1 USD = 100,000 satoshis)
    amounts = np.round(amounts).astype(int)
    return amounts

# Function to visualize the network
def visualize_network(G):
    """
    Visualize the network topology.
    Parameters:
    - G: NetworkX graph
    """
    pos = nx.spring_layout(G)
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color=capacities, edge_cmap=plt.cm.Blues)
    plt.show()

# Main script
if __name__ == "__main__":
    # Set parameters
    num_nodes = 1000  # Number of nodes
    m = 2  # BA model parameter
    mean_capacity = 20000000  # Mean channel capacity: 20 million satoshis
    median_capacity = 5000000  # Median channel capacity: 5 million satoshis
    file_path = "creditcard.csv"  # CSV file path

    # Generate network
    G = generate_lightning_network(num_nodes, m, mean_capacity, median_capacity)

    # Output statistics
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of channels: {G.number_of_edges()}")
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]
    print(f"Average channel capacity: {np.mean(capacities):.2f} satoshis")
    print(f"Median channel capacity: {np.median(capacities):.2f} satoshis")

    # Load payment amounts from creditcard.csv
    num_payments = 1000  # Simulate 1000 payments
    payment_amounts = load_payment_amounts(file_path, num_payments)

    # Simulate multiple payments
    sender = 0
    receiver = 99
    for i, amount in enumerate(payment_amounts):
        sender, receiver = random_sender_receiver(G)
        success, path = simulate_payment(G, sender, receiver, amount)
        if success:
            print(f"Payment {i+1} successful! Amount: {amount} satoshis, Path: {path}")
        else:
            print(f"Payment {i+1} failed! Amount: {amount} satoshis")

    # Visualize the network (optional)
    visualize_network(G)