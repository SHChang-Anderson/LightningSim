import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import ast  # For safely parsing dictionary strings
import re  # For regular expressions

# Set random seed for reproducibility
np.random.seed(42)

import os

def get_paths_from_routing_table(filename, source, destination):
    """
    Extracts all paths from the given source to destination from the routing table file.

    Parameters:
    - filename (str): Path to the routing table file.
    - source (int): Source node.
    - destination (int): Destination node.

    Returns:
    - List of tuples (path, flow), where:
      - path (list): List of nodes in the path.
      - flow (int): Maximum flow for that path.
    """
    paths = []
    if not os.path.exists(filename):
        print(f"Routing table file {filename} not found.")
        return paths

    with open(filename, "r") as file:
        is_target_section = False  # Track whether we're in the correct section
        for line in file:
            line = line.strip()
            
            # Identify section header
            if line.startswith(f"Paths from node{source} to node{destination}:"):
                is_target_section = True
                continue  # Move to the next line

            # If a new section starts, stop processing
            if is_target_section and line.startswith("Paths from node"):
                break

            # Process paths within the relevant section
            if is_target_section and "Path:" in line:
                parts = line.split(", ")
                path_str = parts[0].split(":")[1].strip()  # Extract path
                flow_str = parts[1].split(":")[1].strip()  # Extract flow
                
                # Convert path to list of integers
                path = [int(node.replace("node", "")) for node in path_str.split()]
                flow = int(flow_str)
                paths.append((path, flow))

    return paths

# Function to randomly select a sender and receiver
def random_sender_receiver(G):
    nodes = list(G.nodes())  # Get a list of all nodes
    sender = random.choice(nodes)  # Randomly select sender
    receiver = random.choice(nodes)  # Randomly select receiver
    while sender == receiver:  # Ensure sender and receiver are different
        receiver = random.choice(nodes)
    return sender, receiver


# Function to simulate a payment
def simulate_payment(G, sender, receiver, amount):
    print(sender)
    print(receiver)
    candidate_paths = get_paths_from_routing_table("routing_table/node" + str(sender), sender, receiver)
    if not candidate_paths:
        return False, []
    candidate_paths.sort(key=lambda x: x[1], reverse=True)  # Sort by flow (descending)
    print("Path with the highest flow:", candidate_paths[0][0], candidate_paths[0][1]) # Print the path with the highest flow
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
        # path = candidate_paths[0][0]
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G[u][v]['capacity'] < amount:
                return False, []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            G[u][v]['capacity'] -= amount
            G[v][u]['capacity'] += amount 
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

    file_path = "creditcard.csv"  # CSV file path

    # Generate network
    G = nx.Graph()
    with open("lightning_network.txt", "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=2)
            if len(parts) != 3:
                continue
            u, v, attr_str = parts
            u, v = int(u), int(v)
            match = re.search(r"np\.int64\((\d+)\)", attr_str)
            if match:
                capacity = int(match.group(1))
                attrs = {'capacity': capacity}
                G.add_edge(u, v, **attrs)
            else:
                print(f"Skipping malformed line: {line.strip()}")
    
    # Output statistics
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of channels: {G.number_of_edges()}")
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]
    print(f"Average channel capacity: {np.mean(capacities):.2f} satoshis")
    print(f"Median channel capacity: {np.median(capacities):.2f} satoshis")

    # Load payment amounts from creditcard.csv
    num_payments = 1000  # Simulate 1000 payments
    payment_amounts = load_payment_amounts(file_path, num_payments)

    successful_payments = 0  # Counter for successful payments
    total_payments = len(payment_amounts)  # Total number of payments

    for i, amount in enumerate(payment_amounts):
        sender, receiver = random_sender_receiver(G)
        success, path = simulate_payment(G, sender, receiver, amount)
        if success:
            successful_payments += 1
            print(f"Payment {i+1} successful! Amount: {amount} satoshis, Path: {path}")
        else:
            print(f"Payment {i+1} failed! Amount: {amount} satoshis")

    # Calculate and print success rate
    success_rate = (successful_payments / total_payments) * 100
    print(f"Payment Success Rate: {success_rate:.2f}%")

    # Visualize the network (optional)
    visualize_network(G)