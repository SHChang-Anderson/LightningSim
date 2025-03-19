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
import queue
import sys

np.random.seed(42)

def clear_log_table():
    log_dir = "./log_table"
    if os.path.exists(log_dir):
        # Delete all files and subdirectories in the directory
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete subdirectory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"{log_dir} does not exist.")

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
    def __init__(self, payment_id, sender, receiver, amount, path, arrival_time):
        self.payment_id = payment_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.path = path
        self.arrival_time = arrival_time  # Time when the payment arrives
        self.processing_time = 0  # Time spent processing the payment
        self.completion_time = 0  # Time when the payment completes
        self.success = False
        self.message = ""
        
    def __str__(self):
        status = "Success" if self.success else "Failed"
        return f"Payment {self.payment_id}: {self.amount} satoshis from {self.sender} to {self.receiver}, Status: {status}, Arrival: {self.arrival_time:.2f}s, Completion: {self.completion_time:.2f}s, Processing: {self.processing_time:.2f}s, Path: {self.path}"

    # For compatibility with PriorityQueue
    def __lt__(self, other):
        return self.arrival_time < other.arrival_time
    
# Probing task class
class ProbeTask:
    def __init__(self, path, arrival_time):
        self.path = path
        self.amount = sys.maxsize  # Amount to probe
        self.arrival_time = arrival_time  # For compatibility with PaymentTask
    
# For compatibility with PriorityQueue
    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

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

def record_probe_results(node_id, probe_tasks):
    """
    Record the results of probing tasks to a log file.

    Parameters:
    - node_id (str): The ID of the source node.
    - probe_tasks (list): A list of ProbeTask objects containing probing results.
    """
    log_dir = "./log_table"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    log_file = os.path.join(log_dir, f"node_{node_id}.log")

    # Group probe tasks by destination node
    results_by_destination = {}
    
    destination = probe_tasks.path[-1]  # The last node in the path is the destination
    if destination not in results_by_destination:
        results_by_destination[destination] = []
    results_by_destination[destination].append({
        "path": probe_tasks.path,
        "flow": probe_tasks.amount,  # The probed flow is the minimum capacity along the path
        "fee": len(probe_tasks.path) - 1  # Fee is based on the number of hops
    })

    # Write results to the log file
    with open(log_file, "a") as f:
        for dest, paths in results_by_destination.items():
            for path_info in paths:
                path_str = " ".join(f"node{n}" for n in path_info["path"])
                flow = path_info["flow"]
                fee = path_info["fee"]
                f.write(f"  Path: {path_str}, Flow: {flow}, Fee: {fee}\n")

def probing_worker(G, task_queue, stop_event, simulation_start_time):
    """
    Worker function for probing the network.
    """
    while not stop_event.is_set():
        try:
            # Get the next probing task from the queue
            probing_task = task_queue.get(block=True, timeout=1)

                        # Calculate the current simulation time
            current_time = time.time() - simulation_start_time
            
            # Check if it's time to process this payment
            if current_time < probing_task.arrival_time:
                # If not time yet, put it back in the queue and wait
                task_queue.put(probing_task)
                task_queue.task_done()  # Important: mark this task as done before re-adding
                # time.sleep(0.001)  # Short sleep to prevent CPU spinning
                continue

            try:

                # Update channel capacities
                for i in range(len(probing_task.path) - 1):
                    u, v = probing_task.path[i], probing_task.path[i + 1]
                    probing_task.amount = min(probing_task.amount, G[u][v]['capacity'])
                
            except Exception as e:
                probing_task.message = f"Error probing path: {str(e)}"
            
            # Record the probing results to log table
            record_probe_results(probing_task.path[0], probing_task)

            # Mark the task as done
            task_queue.task_done()
                   
        except queue.Empty:
            # Continue if the queue is empty
            pass

def payment_worker(G, task_queue, result_queue, lock_manager, stop_event, simulation_start_time):
    """
    Worker function for processing individual payments.
    """
    while not stop_event.is_set():
        try:
            # Get the next payment task from the queue
            payment_task = task_queue.get(block=True, timeout=1)
            
            # Calculate the current simulation time
            current_time = time.time() - simulation_start_time
            
            # Check if it's time to process this payment
            if current_time < payment_task.arrival_time:
                # If not time yet, put it back in the queue and wait
                task_queue.put(payment_task)
                task_queue.task_done()  # Important: mark this task as done before re-adding
                # time.sleep(0.001)  # Short sleep to prevent CPU spinning
                continue
                
            # Start processing the payment
            processing_start_time = time.time()
            
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
            
            # Calculate processing time and completion time
            payment_task.processing_time = time.time() - processing_start_time
            payment_task.completion_time = time.time() - simulation_start_time
            
            # Add the payment task to the result queue
            result_queue.put(payment_task)

            # Mark the task as done
            task_queue.task_done()
                
        except queue.Empty:
            # Continue if the queue is empty
            pass


# Simulate threaded payments with Poisson arrival
def simulate_threaded_payments_poisson(G, payment_tasks, probing_task, num_threads=10):
    """
    Simulate parallel payments in the Lightning Network using multiple threads.
    Payments arrive according to a Poisson process.
    """
    # Create task and result queues
    task_queue = queue.PriorityQueue()
    result_queue = queue.Queue()
    
    # Create a probing task queue
    probing_task_queue = queue.PriorityQueue()

    # Create a lock manager
    lock_manager = ChannelLockManager()
    
    # Create a stop event
    stop_event = threading.Event()
    
    # Record the simulation start time
    simulation_start_time = time.time()
    
    # Create and start worker threads
    threads = []
    for _ in range(5):
        thread = threading.Thread(
            target=payment_worker,
            args=(G, task_queue, result_queue, lock_manager, stop_event, simulation_start_time)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Add payment tasks to the task queue
    for task in payment_tasks:
        task_queue.put(task)
    
    # Create and start worker threads
    threads1 = []
    for _ in range(20):
        thread = threading.Thread(
            target=probing_worker,
            args=(G, probing_task_queue, stop_event, simulation_start_time)
        )
        thread.daemon = True
        thread.start()
        threads1.append(thread)
    
    # Add payment tasks to the task queue
    for task in probing_task:
        probing_task_queue.put(task)

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
    
    # Calculate statistics
    if results:
        avg_processing_time = sum(task.processing_time for task in results) / len(results)
        avg_completion_time = sum(task.completion_time for task in results) / len(results)
        print(f"Average processing time: {avg_processing_time:.2f} seconds")
        print(f"Average completion time: {avg_completion_time:.2f} seconds")
    
    # Print payment results
    print(f"\nPayment results:")
    for task in sorted(results, key=lambda x: x.payment_id):
        print(task)
    
    return successful_payments, total_payments, results

# Generate payment arrival times using Poisson process
def generate_poisson_arrival_times(num_payments, rate):
    """
    Generate arrival times for payments based on a Poisson process.
    
    Parameters:
    - num_payments (int): Number of payments to generate
    - rate (float): Average number of payments per second
    
    Returns:
    - List of arrival times
    """
    # Generate inter-arrival times (exponential distribution)
    inter_arrival_times = np.random.exponential(1/rate, num_payments)
    
    # Convert to absolute arrival times
    arrival_times = np.cumsum(inter_arrival_times)
    
    return arrival_times

# Prepare payment tasks with Poisson arrival times
def prepare_payment_tasks_poisson(G, payment_amounts, rate):
    """
    Prepare payment tasks for the simulation with Poisson arrival times.
    
    Parameters:
    - G (nx.Graph): The network graph
    - payment_amounts (list): List of payment amounts
    - rate (float): Average number of payments per second
    
    Returns:
    - List of PaymentTask objects
    """
    tasks = []
    probing_tasks = []

    # Generate arrival times
    arrival_times = generate_poisson_arrival_times(len(payment_amounts), rate)
    
    for i, (amount, arrival_time) in enumerate(zip(payment_amounts, arrival_times)):
        sender, receiver = random_sender_receiver(G)
        
        # Try to find a path in the routing table
        try:
            route_file = f"routing_table/node{sender}"
            candidate_paths = get_paths_from_routing_table(route_file, sender, receiver)
            
            # Sort paths by flow
            candidate_paths.sort(key=lambda x: x[1], reverse=True)
            path = candidate_paths[0][0]  # Select the path with the highest flow
            # path = nx.shortest_path(G, sender, receiver)
                
            # Create a payment task with arrival time
            task = PaymentTask(i+1, sender, receiver, amount, path, arrival_time)
            tasks.append(task)

            ct = 0
            for path1 in candidate_paths:
                ct += 1
                if (ct > 5):
                    break
                print(path1[0])
                probing_task = ProbeTask(path1[0], arrival_time)
                probing_tasks.append(probing_task)
    
        except nx.NetworkXNoPath:  
            print(f"Cannot find a path from {sender} to {receiver}, skipping payment {i+1}")  
        except Exception as e:  
            print(f"Error preparing payment {i+1}: {str(e)}")


    return tasks, probing_tasks

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
    nx.draw(G, pos, with_labels=True, node_size=500, edge_color=capacities)
    plt.show()

# Plot payment arrival and completion times
def plot_payment_statistics(results):
    """
    Plot statistics about payment arrival and completion times.
    """
    arrival_times = [task.arrival_time for task in results]
    completion_times = [task.completion_time for task in results]
    processing_times = [task.processing_time for task in results]
    
    success_status = [task.success for task in results]
    
    plt.figure(figsize=(12, 8))
    
    # Plot arrival and completion times
    plt.subplot(2, 1, 1)
    plt.scatter(arrival_times, range(len(arrival_times)), alpha=0.5, label='Arrival Time')
    plt.scatter(completion_times, range(len(completion_times)), alpha=0.5, label='Completion Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Payment Index')
    plt.title('Payment Arrival and Completion Times')
    plt.legend()
    
    # Plot processing times (success vs failure)
    plt.subplot(2, 1, 2)
    success_processing = [t for t, s in zip(processing_times, success_status) if s]
    failure_processing = [t for t, s in zip(processing_times, success_status) if not s]
    
    plt.hist([success_processing, failure_processing], 
             bins=20, 
             label=['Successful', 'Failed'],
             alpha=0.7)
    plt.xlabel('Processing Time (seconds)')
    plt.ylabel('Count')
    plt.title('Processing Time Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "creditcard.csv"  # CSV file path

    clear_log_table() # Clear the log table

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
    num_payments = int(input("Numbers of payments: "))  # Simulate 1000 payments
    payment_per_second = int(input("Payments per second: "))  # Average 1 payment per second
    payment_amounts = load_payment_amounts(file_path, num_payments)

    # Prepare payment tasks
    payment_tasks, probing_task = prepare_payment_tasks_poisson(G, payment_amounts, payment_per_second)
    
    print(f"Prepared {len(payment_tasks)} payment tasks and {len(probing_task)} probing tasks.")
    
    # Visualize the network
    start_time = time.time()
    
    # Simulate threaded payments
    successful_payments, total_payments, results = simulate_threaded_payments_poisson(G, payment_tasks, probing_task, num_threads=20)
    
    # Calculate and print the success rate
    end_time = time.time()
    
    # Print the results
    success_rate = (successful_payments / total_payments) * 100
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    print(f"Successful Payments: {successful_payments}/{total_payments}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    visualize_network(G)
    plot_payment_statistics(results)
