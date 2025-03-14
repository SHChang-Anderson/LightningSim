import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import ast
import re
import threading
import time
from queue import Queue
from collections import deque

np.random.seed(42)

def get_paths_from_routing_table(filename, source, destination):
    """
    Extract all paths from the routing table file for a specified source node to a destination node.

    Parameters:
    - filename (str): Path to the routing table file.
    - source (int): Source node.
    - destination (int): Destination node.

    Returns:
    - A list containing (path, flow) tuples, where:
    - path (list): A list of nodes in the path.
    - flow (int): The maximum flow capacity of the path.
    """
    paths = []
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return paths

    with open(filename, "r") as file:
        is_target_section = False  # whether we are in the correct section
        for line in file:
            line = line.strip()
            
            #  Identify section header
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
                
                #  Convert path to list of integers
                path = [int(node.replace("node", "")) for node in path_str.split()]
                flow = int(flow_str)
                paths.append((path, flow))

    return paths

# Randomly select a sender and receiver
def random_sender_receiver(G):
    nodes = list(G.nodes())  # Get a list of all nodes
    sender = random.choice(nodes)  # Randomly select sender
    receiver = random.choice(nodes)  # Randomly select receiver 
    while sender == receiver:  # Ensure sender and receiver are different
        receiver = random.choice(nodes)
    return sender, receiver

#  Payment task class
class PaymentTask:
    def __init__(self, payment_id, sender, receiver, amount, path):
        self.payment_id = payment_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.path = path
        self.success = False
        self.message = ""
        
    def __str__(self):
        status = "Success" if self.success else "Failed"
        return f"Payment {self.payment_id}: {self.amount} satoshis from {self.sender} to {self.receiver}, Status: {status}, Path: {self.path}"

# Check if the path has enough capacity
class ChannelLockManager:
    def __init__(self):
        self.locks = {}
        self.lock_dict_lock = threading.Lock()
        
    def acquire_channel_lock(self, channel):
        """Acquire the lock for the specified channel"""
        with self.lock_dict_lock:
            if channel not in self.locks:
                self.locks[channel] = threading.Lock()
        
        self.locks[channel].acquire()
        
    def release_channel_lock(self, channel):
        """Release the lock for the specified channel"""
        if channel in self.locks:
            self.locks[channel].release()
            
    def acquire_path_locks(self, path):
        """Acquire locks for all channels on the path"""
        channels = [(path[i], path[i+1]) for i in range(len(path)-1)]
        # Sort the channels to avoid deadlocks
        channels.sort()
        
        for channel in channels:
            self.acquire_channel_lock(channel)
        
        return channels
        
    def release_path_locks(self, channels):
        """Release locks for all channels on the path"""
        for channel in channels:
            self.release_channel_lock(channel)

# Payment worker
def payment_worker(G, task_queue, result_queue, lock_manager, stop_event):
    """
    Worker function for processing individual payments.
    """
    while not stop_event.is_set():
        try:
            # Get the next payment task from the queue
            payment_task = task_queue.get(block=True, timeout=1)
            
            try:
                # Acquire locks for all channels on the path
                channels = lock_manager.acquire_path_locks(payment_task.path)
                
                # Check if the path has enough capacity
                has_capacity = True
                for i in range(len(payment_task.path) - 1):
                    u, v = payment_task.path[i], payment_task.path[i + 1]
                    if G[u][v]['capacity'] < payment_task.amount:
                        has_capacity = False
                        payment_task.message = f"Channel {u}-{v} has insufficient capacity: {G[u][v]['capacity']} < {payment_task.amount}"
                        break
                
                # Update the capacity of all channels on the path
                if has_capacity:
                    # Update channel capacities
                    for i in range(len(payment_task.path) - 1):
                        u, v = payment_task.path[i], payment_task.path[i + 1]
                        G[u][v]['capacity'] -= payment_task.amount
                        G[v][u]['capacity'] += payment_task.amount
                    
                    payment_task.success = True
                    payment_task.message = "Payment successful"
                
                # Release locks for all channels on the path
                lock_manager.release_path_locks(channels)
                
            except Exception as e:
                payment_task.message = f"Error processing payment: {str(e)}"
                
            # Add the payment task to the result queue
            result_queue.put(payment_task)
            
            # Mark the task as done
            task_queue.task_done()
                
        except task_queue.Empty:
            # Continue if the queue is empty
            pass

# Simulate threaded payments
def simulate_threaded_payments(G, payment_tasks, num_threads=10):
    """
    Simulate parallel payments in the Lightning Network using multiple threads.
    """
    # Create task and result queues
    task_queue = Queue()
    result_queue = Queue()
    
    # Create a lock manager
    lock_manager = ChannelLockManager()
    
    # Create a stop event
    stop_event = threading.Event()
    
    # Create and start worker threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=payment_worker,
            args=(G, task_queue, result_queue, lock_manager, stop_event)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Add payment tasks to the task queue
    for task in payment_tasks:
        task_queue.put(task)
    
    # Wait for all tasks to be processed
    task_queue.join()
    
    # Set the stop event
    stop_event.set()
    
    # Join all threads
    for thread in threads:
        thread.join(timeout=1)
    
    # Get results from the result queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Count successful payments
    successful_payments = sum(1 for task in results if task.success)
    total_payments = len(results)
    
    # Print payment results
    print(f"\nPayment results:")
    for task in sorted(results, key=lambda x: x.payment_id):
        print(task)
    
    return successful_payments, total_payments

# Prepare payment tasks
def prepare_payment_tasks(G, payment_amounts):
    """
    Prepare payment tasks for the simulation.
    """
    tasks = []
    
    for i, amount in enumerate(payment_amounts):
        sender, receiver = random_sender_receiver(G)
        
        # Try to find a path in the routing table
        try:
            route_file = f"routing_table/node{sender}"
            candidate_paths = get_paths_from_routing_table(route_file, sender, receiver)
            
            if candidate_paths:
                # Sort paths by flow
                candidate_paths.sort(key=lambda x: x[1], reverse=True)
                path = candidate_paths[0][0]  # Select the path with the highest flow
            else:
                # If no path is found in the routing table, use the shortest path
                path = nx.shortest_path(G, sender, receiver)
                
            # Create a payment task
            task = PaymentTask(i+1, sender, receiver, amount, path)
            tasks.append(task)
                
        except nx.NetworkXNoPath:  
            print(f"Cannot find a path from {sender} to {receiver}, skipping payment {i+1}")  
        except Exception as e:  
            print(f"Error preparing payment {i+1}: {str(e)}")

    
    return tasks

# Load payment amounts
def load_payment_amounts(file_path, num_payments):
    """
    Load transaction amounts from creditcard.csv and map them to payment simulation.
    """
    df = pd.read_csv(file_path)
    amounts = df['Amount'].head(num_payments).values * 100000  # Convert to satoshis (assuming 1 USD = 100,000 satoshis)
    amounts = np.round(amounts).astype(int)
    return amounts

# Visualize the network
def visualize_network(G):
    """
    Visualize the network topology.
    """
    pos = nx.spring_layout(G)
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color=capacities, edge_cmap=plt.cm.Blues)
    plt.show()

if __name__ == "__main__":
    file_path = "creditcard.csv"  # CSV file path

    # Load the Lightning Network graph
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
                print(f"Skipping incorrectly formatted line: {line.strip()}")  

    # Output statistics  
    print(f"Number of nodes: {G.number_of_nodes()}")  
    print(f"Number of channels: {G.number_of_edges()}")  
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]  
    print(f"Average channel capacity: {np.mean(capacities):.2f} satoshis")  
    print(f"Median channel capacity: {np.median(capacities):.2f} satoshis")

    # Load payment amounts from creditcard.csv
    num_payments = 1000  # Simulate 1000 payments
    payment_amounts = load_payment_amounts(file_path, num_payments)

    # Prepare payment tasks
    payment_tasks = prepare_payment_tasks(G, payment_amounts)
    
    print(f"Prepared {len(payment_tasks)} payment tasks")
    
    # Visualize the network
    start_time = time.time()
    
    # Simulate threaded payments
    successful_payments, total_payments = simulate_threaded_payments(G, payment_tasks, num_threads=10)
    
    # Calculate and print the success rate
    end_time = time.time()
    
    # Print the results
    success_rate = (successful_payments / total_payments) * 100
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    print(f"Successful Payments: {successful_payments}/{total_payments}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
