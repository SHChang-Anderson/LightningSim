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
import uuid
import shutil
# {'param1': 96.09985339519194, 'param2': 2.10183642984912, 'param3': 0.8338142960594785, 'param4': 0.19995197310808, 'param5': 9}


# 'param1': 1.0611725984543918, 'param2': 60.27367308783179, 'param3': 7.360643237372914, 'param4': 0.3626886219630006, 'param5': 2} 

'''
import sys
sys.argv = [
    "simulator_thread.py",  # Script name
    str(0),   # Probing mode
    str(1000),      # Number of payments
    str(10),  # Payments per second
    str(1.0611725984543918), # Parameter 1
    str(60.27367308783179), # Parameter 2
    str(7.360643237372914), # Parameter 3
    str(0.3626886219630006), # Parameter 4
    str(2) # Parameter 5
]
'''

np.random.seed(42)

G = nx.DiGraph()

log_table = {}  # Global variable used to store the log table
log_table_locks = {}  # Used to store the locks for each [sender][receiver]
log_table_locks_lock = threading.Lock()  # Lock used to protect log_table_locks

# Global variable to store probing results
probing_results = {}
probing_results_lock = threading.Lock()

# Create a probing task queue
probing_task_queue = queue.PriorityQueue()

# store transaction timestamps
transaction_timestamps = {}

# Trust pairs for nodes
trust_pairs = {}

def add_trust_pair(trust_pairs, node1, node2):
    """
    Add a trust relationship between two nodes.

    Parameters:
    - trust_pairs (dict): The dictionary storing trust relationships.
    - node1 (int): The first node.
    - node2 (int): The second node.
    """
    if node1 not in trust_pairs:
        trust_pairs[node1] = set()
    trust_pairs[node1].add(node2)

def remove_trust_pair(trust_pairs, node1, node2):
    """
    Remove a trust relationship between two nodes.

    Parameters:
    - trust_pairs (dict): The dictionary storing trust relationships.
    - node1 (int): The first node.
    - node2 (int): The second node.
    """
    if node1 in trust_pairs and node2 in trust_pairs[node1]:
        trust_pairs[node1].remove(node2)
        if not trust_pairs[node1]: 
            del trust_pairs[node1]

def is_trusted(trust_pairs, node1, node2):
    """
    Check if node1 trusts node2.

    Parameters:
    - trust_pairs (dict): The dictionary storing trust relationships.
    - node1 (int): The first node.
    - node2 (int): The second node.

    Returns:
    - bool: True if node1 trusts node2, False otherwise.
    """
    return node1 in trust_pairs and node2 in trust_pairs[node1]

def query_trusted_node(path):
    """
    Query a trusted node for channel information along a given path.

    Parameters:
    - trusted_node (int): The ID of the trusted node.
    - path (list): The path to query.

    Returns:
    - dict: A dictionary containing channel information (capacity and usage_frequency).
    - int: The minimum capacity along the path.
    - float: The maximum usage frequency along the path.
    """

    channel_info = {}
    min_capacity = float('inf')  # Initialize to a very large value
    max_usage_frequency_diff = float('-inf')  # Initialize to a very small value

    for i in range(0, len(path) - 1):
        u, v = path[i], path[i + 1]
        if i == 0 or is_trusted(trust_pairs, path[0], v) or is_trusted(trust_pairs, path[0], u) :
            capacity = G[u][v]['capacity']
            usage_frequency_diff = G[u][v].get('usage_frequency', 0) - G[v][u].get('usage_frequency', 0)

            # Update min_capacity and max_usage_frequency
            min_capacity = min(min_capacity, capacity)
            max_usage_frequency_diff = max(max_usage_frequency_diff, usage_frequency_diff)

            # Add channel information to the result
            channel_info[(u, v)] = {
                'capacity': capacity,
                'usage_frequency': usage_frequency_diff
            }

    # If no trusted channels are found, set min_capacity and max_usage_frequency to None
    if not channel_info:
        min_capacity = float('inf')
        max_usage_frequency_diff = 0

    return channel_info, min_capacity, max_usage_frequency_diff


def update_trust_scores_thread(G, trust_pairs, stop_event, update_interval=1, output_file="trust_nodes.txt"):
    """
    Thread function to periodically update trust scores based on recent transactions
    and output trust nodes to a file.

    Parameters:
    - G (nx.Graph): The network graph.
    - trust_pairs (dict): The dictionary storing trust relationships.
    - stop_event (threading.Event): Event to signal the thread to stop.
    - update_interval (int): The interval (in seconds) between updates.
    - output_file (str): The file to output trust nodes.
    """
    while not stop_event.is_set():
        try:
            # Update trust scores based on recent transactions
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    # Update trust relationship based on recent transactions
                    if node in transaction_timestamps and neighbor in transaction_timestamps[node]:
                        if transaction_timestamps[node][neighbor]:
                            add_trust_pair(trust_pairs, node, neighbor)
                        else:
                            remove_trust_pair(trust_pairs, node, neighbor)
                    else:
                        remove_trust_pair(trust_pairs, node, neighbor)  # if no transactions, remove trust

            # Output trust nodes to a file
            with open(output_file, "a") as f:
                for node, trusted_nodes in trust_pairs.items():
                    trusted_nodes_str = ", ".join(map(str, trusted_nodes))
                    f.write(f"Node {node}: {trusted_nodes_str}\n")

            # Sleep for the update interval
            time.sleep(update_interval)
        except Exception as e:
            print(f"Error in update_trust_scores_thread: {str(e)}")

def clear_log_table():
    """
    Clear the global log_table and all associated locks.
    """
    global log_table, log_table_locks
    with log_table_locks_lock:
        log_table.clear()
        log_table_locks.clear()

def get_log_table_lock(sender, receiver):
    """
    Ensure a lock exists for the specified [sender][receiver] combination.

    Parameters:
    - sender (int): The sender node ID.
    - receiver (int): The receiver node ID.

    Returns:
    - threading.Lock: The lock for the specified [sender][receiver].
    """
    global log_table_locks
    with log_table_locks_lock:
        if sender not in log_table_locks:
            log_table_locks[sender] = {}
        if receiver not in log_table_locks[sender]:
            log_table_locks[sender][receiver] = threading.Lock()
        return log_table_locks[sender][receiver]
    
def get_paths_from_log_table(sender, receiver):
    """
    Retrieve paths from the global log_table for a specific sender and receiver.

    Parameters:
    - sender (int): The sender node ID.
    - receiver (int): The receiver node ID.

    Returns:
    - list: A list of dictionaries containing path, flow, fee, and timestamp.
    """
    global log_table

    if sender in log_table and receiver in log_table[sender]:
        return log_table[sender][receiver]
    else:
        return []  # Return an empty list if no paths are found

def read_paths_from_log(log_file, destination):
    """
    Read paths from a node log file for a specific destination and store them in a list.

    Parameters:
    - log_file (str): Path to the log file.
    - destination (int): The destination node ID to filter paths.

    Returns:
    - paths (list): A list of dictionaries containing path, flow, fee, and timestamp.
    """
    paths = []
    if not os.path.exists(log_file):
        print(f"{log_file} does not exist.")
        return paths

    with open(log_file, "r") as f:
        lines = f.readlines()

    # Track whether we are processing the correct destination block
    is_correct_destination = False

    for line in lines:
        line = line.strip()
        if line.startswith("Paths from"):
            # Check if this block corresponds to the specified destination
            current_destination = int(line.split("to node")[1].strip(":"))
            is_correct_destination = (current_destination == destination)

        elif is_correct_destination and line.startswith("Path:"):
            try:
                # Extract path, flow, fee, and timestamp
                parts = line.split(", ")
                path_str = parts[0].split(":")[1].strip()  # Extract path
                flow = int(parts[1].split(":")[1].strip())  # Extract flow
                fee = int(parts[2].split(":")[1].strip())  # Extract fee
                timestamp = float(parts[3].split(":")[1].strip())  # Extract timestamp

                # Convert path to a list of integers
                path = [int(node) for node in path_str.split()]

                # Append to paths
                paths.append({
                    "path": path,
                    "flow": flow,
                    "fee": fee,
                    "timestamp": timestamp
                })
            except Exception as e:
                print(f"Failed to parse line: {line}. Reason: {e}")

    return paths

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
                Base_fee_str = parts[2].split(":")[1].strip()  # Extract base fee
                Fee_rate_str = parts[3].split(":")[1].strip()  # Extract fee rate
                
                #  Convert path to list of integers
                path = [int(node.replace("node", "")) for node in path_str.split()]
                flow = float(flow_str)
                Base_fee = float(Base_fee_str)
                Fee_rate = float(Fee_rate_str)
                paths.append((path, flow, Base_fee, Fee_rate))

    return paths

# Randomly select a sender and receiver
def random_sender_receiver(G, previous_transactions, repeat_ratio=0.75):
    """
    Select a sender and receiver pair with a bias towards repeated transactions
    and clustered transaction pairs.

    Parameters:
    - G (nx.Graph): The network graph.
    - previous_transactions (list): A list of previous sender-receiver pairs.
    - repeat_ratio (float): The ratio of transactions that are repeated (default: 0.86).

    Returns:
    - sender (int): The selected sender node.
    - receiver (int): The selected receiver node.
    """
    nodes = list(G.nodes())  # Get a list of all nodes

    # 86% of the time, select a repeated transaction
    if random.random() < repeat_ratio and previous_transactions:
        sender, receiver = random.choice(previous_transactions)
        if random.random() < 0.5:
            temp = sender
            sender = receiver
            receiver = temp
    else:
        # Generate a new sender-receiver pair
        sender = random.choice(nodes)
        receiver = random.choice(nodes)
        while sender == receiver:  # Ensure sender and receiver are different
            receiver = random.choice(nodes)

        # Add the new transaction to the previous transactions list
        previous_transactions.append((sender, receiver))

    return sender, receiver

#  Payment task class
class PaymentTask:
    def __init__(self, payment_id, sender, receiver, amount, path, arrival_time, fee=0):
        """
        Initialize a PaymentTask object.

        Parameters:
        - payment_id (int): Unique ID for the payment.
        - sender (int): Sender node ID.
        - receiver (int): Receiver node ID.
        - amount (int): Payment amount in satoshis.
        - path (list): Path for the payment.
        - arrival_time (float): Time when the payment arrives.
        - fee (float): Total fee for the payment (default: 0).
        """
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
        self.fee = fee  # Total fee for the payment

    def __str__(self):
        """
        String representation of the PaymentTask object.
        """
        status = "Success" if self.success else "Failed"
        return (f"Payment {self.payment_id}: {self.amount} satoshis from {self.sender} to {self.receiver}, "
                f"Status: {status}, Arrival: {self.arrival_time:.2f}s, Completion: {self.completion_time:.2f}s, "
                f"Processing: {self.processing_time:.2f}s, Fee: {self.fee:.6f}, Path: {self.path}")

    # For compatibility with PriorityQueue
    def __lt__(self, other):
        return self.arrival_time < other.arrival_time
    
# Probing task class
class ProbeTask:
    def __init__(self, path, arrival_time):
        self.task_id = str(uuid.uuid4())  # generate a unique task ID
        self.path = path
        self.amount = sys.maxsize  # Amount to probe
        self.arrival_time = arrival_time  # For compatibility with PaymentTask

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
        """Acquire locks for all channels on the path, including reverse channels"""
        # Generate a list of both forward and reverse channels
        channels = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            channels.append((u, v))  # Forward channel
            channels.append((v, u))  # Reverse channel

        # Sort the channels to avoid deadlocks
        channels.sort()

        # Acquire locks for all channels
        for channel in channels:
            self.acquire_channel_lock(channel)

        return channels
        
    def release_path_locks(self, channels):
        """Release locks for all channels on the path, including reverse channels"""
        # Generate a list of both forward and reverse channels
        all_channels = []
        for u, v in channels:
            all_channels.append((u, v))  # Forward channel
            all_channels.append((v, u))  # Reverse channel

        # Remove duplicates (if any) and sort to ensure consistency
        all_channels = sorted(set(all_channels))

        # Release locks for all channels
        for channel in all_channels:
            self.release_channel_lock(channel)

def update_log_table(sender, receiver, path, flow, fee, timestamp):
    """
    Update the global log_table with a new path entry.

    Parameters:
    - sender (int): The sender node ID.
    - receiver (int): The receiver node ID.
    - path (list): The path as a list of node IDs.
    - flow (int): The flow capacity of the path.
    - fee (int): The fee for the path.
    - timestamp (float): The timestamp of the path.
    """
    global log_table
    lock = get_log_table_lock(sender, receiver)  # aquire lock for this sender-receiver pair
    with lock:
        if sender not in log_table:
            log_table[sender] = {}
        if receiver not in log_table[sender]:
            log_table[sender][receiver] = []
        
        # Append the new path entry
        log_table[sender][receiver].append({
            "path": path,
            "flow": flow,
            "fee": fee,
            "timestamp": timestamp
        })            

def record_probe_results(node_id, probe_tasks, simulation_start_time):
    """
    Record the results of probing tasks to the global log_table.

    Parameters:
    - node_id (int): The ID of the source node.
    - probe_tasks (ProbeTask): The probing task containing path information.
    - simulation_start_time (float): The simulation start time in seconds.
    """
    destination = probe_tasks.path[-1]  # The last node in the path is the destination
    path = probe_tasks.path # The probed path
    flow = probe_tasks.amount  # The probed flow is the minimum capacity along the path
    fee = len(probe_tasks.path) - 1  # Fee is based on the number of hops
    timestamp = round(time.time() - simulation_start_time, 2)  # Time since the simulation started

    # Update the log_table
    update_log_table(node_id, destination, path, flow, fee, timestamp)
    update_log_table(node_id, path[0], path[::-1], flow, fee, timestamp)

def get_sorted_candidate_paths(payment_task, alpha=float(sys.argv[4]), beta=float(sys.argv[5]), gamma=float(sys.argv[6])):
    """
    Get sorted candidate paths based on the composite score formula.

    Parameters:
    - payment_task (PaymentTask): The payment task containing sender, receiver, and amount.
    - alpha (float): Weight for Total Channel Capacity.
    - beta (float): Weight for Base Fee.
    - epsilon (float): Weight for Channel Usage Frequency.

    Returns:
    - List of paths sorted by their composite score in descending order.
    """
    # Read candidate paths from the routing table
    candidate_paths = get_paths_from_routing_table(
        f"../routing_table/node{payment_task.sender}",
        payment_task.sender,
        payment_task.receiver
    )
    # initialize min and max values
    min_capacity, max_capacity = float('inf'), float('-inf')
    min_fee, max_fee = float('inf'), float('-inf')
    min_usage, max_usage = float('inf'), float('-inf')

    # compute min and max values for normalization
    for path, flow, Base_fee, Fee_rate in candidate_paths:
        total_channel_capacity = flow
        base_fee = Base_fee
        fee_rate = Fee_rate
        channel_info, min_capacity_path, max_usage_frequency_diff = query_trusted_node(path)

        # update min and max values
        min_capacity = min(min_capacity, total_channel_capacity, min_capacity_path)
        max_capacity = max(max_capacity, total_channel_capacity, min_capacity_path)

        total_fee = base_fee + (fee_rate * payment_task.amount)
        min_fee = min(min_fee, total_fee)
        max_fee = max(max_fee, total_fee)

        min_usage = min(min_usage, max_usage_frequency_diff)
        max_usage = max(max_usage, max_usage_frequency_diff)

    # normalize the values
    scored_paths = []
    for path, flow, Base_fee, Fee_rate in candidate_paths:
        total_channel_capacity = flow
        base_fee = Base_fee
        fee_rate = Fee_rate
        channel_info, min_capacity_path, max_usage_frequency_diff = query_trusted_node(path)

        # normalize the values
        normalized_capacity = (min(flow, min_capacity_path) - min_capacity) / (max_capacity - min_capacity) if max_capacity > min_capacity else 0
        total_fee = base_fee + (fee_rate * payment_task.amount)
        normalized_fee = (total_fee - min_fee) / (max_fee - min_fee) if max_fee > min_fee else 0
        normalized_usage = (max_usage_frequency_diff - min_usage) / (max_usage - min_usage) if max_usage > min_usage else 0

        # compute the composite score
        score = (
            alpha * normalized_capacity -
            beta * normalized_fee -
            gamma * normalized_usage
        )
        scored_paths.append((path, flow, score, total_fee))

    # sort paths by score in descending order
    scored_paths.sort(key=lambda x: x[2], reverse=True)

    # Return only the paths sorted by score
    return scored_paths

def probing_worker(stop_event, simulation_start_time):
    """
    Worker function for probing the network.
    """
    while not stop_event.is_set():
        try:
            # Get the next probing task from the queue
            probing_task = probing_task_queue.get(block=True, timeout=1)

            try:
                global G
                # Update channel capacities
                for i in range(len(probing_task.path) - 1):
                    time.sleep(0.06)
                    u, v = probing_task.path[i], probing_task.path[i + 1]
                    probing_task.amount = min(probing_task.amount, G[u][v]['capacity'])
                
            except Exception as e:
                probing_task.message = f"Error probing path: {str(e)}"
            
            # Record the probing results to log table
            record_probe_results(probing_task.path[0], probing_task, simulation_start_time)

            # Mark the task as done
            probing_task_queue.task_done()
                   
        except queue.Empty:
            # Continue if the queue is empty
            pass

def is_task_completed(task_id):
    """
    Check if a probing task with the given task_id has been completed.
    """

    global probing_results, probing_results_lock

    with probing_results_lock:
        return task_id in probing_results

def wait_for_all_tasks(task_ids):
    """
    wait for all probing tasks to complete.
    """
    global probing_results, probing_results_lock

    while True:
        with probing_results_lock:
            if all(task_id in probing_results for task_id in task_ids):
                break

def probing_worker_ps(stop_event, simulation_start_time):
    """
    Worker function for probing the network.
    """
    while not stop_event.is_set():
        try:
            # Get the next probing task from the queue
            probing_task = probing_task_queue.get(block=True, timeout=1)

            try:
                global G, probing_results, probing_results_lock
                # Update channel capacities
                for i in range(len(probing_task.path) - 1):
                    time.sleep(0.06)
                    u, v = probing_task.path[i], probing_task.path[i + 1]
                    probing_task.amount = min(probing_task.amount, G[u][v]['capacity'])

                # construct the probing result
                result = {
                    "task_id": probing_task.task_id,
                    "path": probing_task.path,
                    "amount": probing_task.amount,
                    "timestamp": time.time() - simulation_start_time
                }

                # save the probing result to the global dictionary
                with probing_results_lock:
                    probing_results[probing_task.task_id] = result

            except Exception as e:
                print(f"Error probing path: {str(e)}")
            
            finally:
                # Mark the task as done
                probing_task_queue.task_done()
                   
        except queue.Empty:
            # Continue if the queue is empty
            pass

def payment_worker(task_queue, result_queue, lock_manager, stop_event, simulation_start_time, split_rate = float(sys.argv[7])):
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

                # Check if argv is provided and set the flag
                execute_probing = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # default to 0

                if execute_probing == 1:
                    
                    log_paths = get_paths_from_log_table(payment_task.sender, payment_task.receiver)
                    if (log_paths):
                        
                        candidate_paths = get_paths_from_routing_table(f"../routing_table/node{payment_task.sender}", payment_task.sender, payment_task.receiver)
                        for path1 in candidate_paths:
                            probing_task = ProbeTask(path1[0], time.time() - simulation_start_time)
                            probing_task_queue.put(probing_task)

                        # Sort paths by flow
                        log_paths.sort(key=lambda x: x["flow"], reverse=True)

                        for log_path in log_paths:
                            payment_task.path = log_path["path"]
                            break
                    else:
                        candidate_paths = get_paths_from_routing_table(f"../routing_table/node{payment_task.sender}", payment_task.sender, payment_task.receiver)
                        for path1 in candidate_paths:
                            probing_task = ProbeTask(path1[0], time.time() - simulation_start_time)
                            probing_task_queue.put(probing_task)

                        probing_task_queue.join()

                        
                        log_paths = get_paths_from_log_table(payment_task.sender, payment_task.receiver)

                        # Sort paths by flow
                        log_paths.sort(key=lambda x: x["flow"], reverse=True)

                        # print( payment_task.path, log_paths[0]["path"])
                        for log_path in log_paths:
                            payment_task.path = log_path["path"]
                            break
                split = 0
                if execute_probing == 3:          

                    # Get the sorted candidate paths
                    candidate_paths = get_sorted_candidate_paths(payment_task)
                    # Select the best path
                    if candidate_paths:
                        payment_task.path = candidate_paths[0][0]
                        if (candidate_paths[0][1] - payment_task.amount) / candidate_paths[0][1] < split_rate:
                            split = 1  

                execute_paths = []
                execute_payment = []
                if execute_probing == 4:

                    candidate_paths = get_paths_from_routing_table(
                        f"../routing_table/node{payment_task.sender}",
                        payment_task.sender,
                        payment_task.receiver
                    )

                    # calculate the total fee for each path
                    paths_with_fees = []
                    for path, flow, base_fee, fee_rate in candidate_paths:
                        total_fee = base_fee + (fee_rate * payment_task.amount)
                        paths_with_fees.append((path, flow, total_fee))

                    # sort paths by fee
                    paths_with_fees.sort(key=lambda x: x[2]) 

                    # upadate the path with the lowest fee
                    candidate_paths = [(path, flow) for path, flow, _ in paths_with_fees]
                    
                    remain_amount = payment_task.amount
                    iter = 0
                    
                    for path, flow in candidate_paths:  

                        if (remain_amount < 0):
                            break

                        try:
                            global G
                            # Update channel capacities
                            tpamount = float('inf')
                            for i in range(len(path) - 1):
                                time.sleep(0.06)
                                u, v = path[i], path[i + 1]
                                tpamount = min(tpamount, G[u][v]['capacity'])

                            execute_paths.append(path)
                            execute_payment.append(tpamount)
                            remain_amount -= tpamount

                        except Exception as e:
                            probing_task.message = f"Error probing path: {str(e)}"
                            break
                    
                    split = 2
                        


                # Acquire locks for all channels on the path
                channels = []
                has_capacity = True
                rollback_channels = []  # record channels that need to be rolled back

                try:
                    total_fee = 0
                    alpha = 0.5  # EWMA weight for channel capacity
                    if split == 0:
                        for i in range(len(payment_task.path) - 1):

                            time.sleep(0.03)
                            u, v = payment_task.path[i], payment_task.path[i + 1]
                            
                            lock_manager.acquire_channel_lock((u, v))
                            channels.append((u, v))  # record the channel that has been locked
                            
                            # check if the channel has enough capacity
                            if G[u][v]['capacity'] < payment_task.amount:
                                has_capacity = False
                                payment_task.message = f"Channel {u}-{v} has insufficient capacity: {G[u][v]['capacity']} < {payment_task.amount}"
                                lock_manager.release_channel_lock((u, v))
                                break
                            
                            # temporarily deduct the amount from the channel
                            G[u][v]['capacity'] -= payment_task.amount
                            G[v][u]['capacity'] += payment_task.amount
                            rollback_channels.append((u, v))  # record the channel that has been deducted
                            lock_manager.release_channel_lock((u, v))

                            # Update the usage frequency of the channel using EWMA
                            now = time.time()
                            if G[u][v]['last_used'] is not None:
                                # Calculate the time difference since the last usage
                                time_diff = now - G[u][v]['last_used']
                                if time_diff > 0:
                                    # Apply EWMA formula
                                    previous_frequency = G[u][v].get('usage_frequency', 0)
                                    current_frequency = 1 / time_diff
                                    G[u][v]['usage_frequency'] = alpha * current_frequency + (1 - alpha) * previous_frequency
                            else:
                                # If this is the first usage, set the frequency to a default value
                                G[u][v]['usage_frequency'] = 1
                            
                            # Update the last used timestamp
                            G[u][v]['last_used'] = now
                            
                            # Add the fees for both directions
                            total_fee += (G[u][v]['fee'] + G[u][v]['rate'] * payment_task.amount)

                        # If all channels have enough capacity, the payment is successful
                        if has_capacity:
                            payment_task.success = True
                            payment_task.fee = total_fee
                            payment_task.message = "Payment successful"

                            # Record the transaction timestamp for each channel in the path
                            timestamp = time.time()
                            
                            u, v = payment_task.path[0], payment_task.path[-1]
                            if u not in transaction_timestamps:
                                transaction_timestamps[u] = {}
                            if v not in transaction_timestamps[u]:
                                transaction_timestamps[u][v] = []
                            
                            # Append the timestamp
                            transaction_timestamps[u][v].append(timestamp)

                            # Limit the number of stored timestamps to avoid memory issues
                            if len(transaction_timestamps[u][v]) > 100:
                                transaction_timestamps[u][v].pop(0)
                            
                            v, u = payment_task.path[0], payment_task.path[-1]
                            if u not in transaction_timestamps:
                                transaction_timestamps[u] = {}
                            if v not in transaction_timestamps[u]:
                                transaction_timestamps[u][v] = []
                            
                            # Append the timestamp
                            transaction_timestamps[u][v].append(timestamp)

                            # Limit the number of stored timestamps to avoid memory issues
                            if len(transaction_timestamps[u][v]) > 100:
                                transaction_timestamps[u][v].pop(0)
                    
                        else:
                            payment_task.fee = 0
                            # If any channel does not have enough capacity, the payment fails
                            for u, v in rollback_channels:
                                lock_manager.acquire_channel_lock((u, v))
                                time.sleep(0.03)
                                G[u][v]['capacity'] += payment_task.amount
                                G[v][u]['capacity'] -= payment_task.amount
                                lock_manager.release_channel_lock((u, v))

                    elif split == 2:
                        rollback_channels_all = []  # record channels that need to be rolled back   
                        remain_amount = payment_task.amount
                        for p in range(len(execute_paths)):

                            if has_capacity == False:
                                break

                            for i in range(len(execute_paths[p]) - 1):

                                time.sleep(0.03)
                                u, v = execute_paths[p][i], execute_paths[p][i+1]
                                
                                lock_manager.acquire_channel_lock((u, v))
                                channels.append((u, v))  # record the channel that has been locked
                                
                                if remain_amount < execute_payment[p]:
                                    execute_payment[p] = remain_amount

                                # check if the channel has enough capacity
                                if G[u][v]['capacity'] < execute_payment[p]:
                                    has_capacity = False
                                    payment_task.message = f"Channel {u}-{v} has insufficient capacity: {G[u][v]['capacity']} < {payment_task.amount}"
                                    lock_manager.release_channel_lock((u, v))
                                    break
                                
                                # temporarily deduct the amount from the channel
                                G[u][v]['capacity'] -= execute_payment[p]
                                G[v][u]['capacity'] += execute_payment[p]

                                remain_amount -= execute_payment[p]

                                if len(rollback_channels_all) < p + 1:
                                    rollback_channels_all.append([])

                                rollback_channels_all[p].append((u, v))  # record the channel that has been deducted

                                lock_manager.release_channel_lock((u, v))
                                
                                # Add the fees for both directions
                                total_fee += (G[u][v]['fee'] + G[u][v]['rate'] * execute_payment[p])
                                if remain_amount < 0:
                                    break  

                        # If all channels have enough capacity, the payment is successful
                        if has_capacity:
                            payment_task.success = True
                            payment_task.fee = total_fee
                            payment_task.message = "Payment successful"
                    
                        else:
                            payment_task.fee = 0

                            pos = 0  # current position in the rollback channels
                            roll_back_pos = []  # record channels that have been rolled back

                            while True:
                                time.sleep(0.03)  # simulate processing time

                                for i in range(len(rollback_channels_all)):
                                    if i in roll_back_pos:
                                        continue  # skip already rolled back channels

                                    if pos >= len(rollback_channels_all[i]):
                                        roll_back_pos.append(i)  # if the channel has been rolled back, add it to the list
                                        continue

                                    u, v = rollback_channels_all[i][pos]  # aquire the channel to be rolled back
                                    try:
                                        lock_manager.acquire_channel_lock((u, v))

                                        # roll back the channel
                                        G[u][v]['capacity'] += execute_payment[i]
                                        G[v][u]['capacity'] -= execute_payment[i]
                                    finally:
                                        lock_manager.release_channel_lock((u, v))

                                # if all channels have been rolled back, break the loop
                                if len(roll_back_pos) == len(rollback_channels_all):
                                    break

                                pos += 1  # next channel to roll back

                    else:
                        split_cont = [0, 1]
                        payment_split_amount = []
                        rollback_channels_all = []  # record channels that need to be rolled back
                        min_succ_rate = float('inf')

                        while True:
                            total_flow = sum(path[1] for path in candidate_paths[:split_cont[-1] + 1] if path[1] > 0)
                            if total_flow == 0:
                                raise ValueError("Total flow is zero, cannot split payment.")
                            
                            for i in split_cont: 

                                # caulculate the split amount for each path
                            
                                split_amount = payment_task.amount * (candidate_paths[i][1] / total_flow)
                                payment_split_amount.append(split_amount)

                                # caulculate the success rate
                                min_succ_rate = min(
                                    min_succ_rate,
                                    (candidate_paths[i][1] - split_amount) / candidate_paths[i][1]
                                )

                            if min_succ_rate < float(sys.argv[7]) and len(split_cont) < len(candidate_paths) and len(split_cont) < float(sys.argv[8]):
                                payment_split_amount = []
                                min_succ_rate = float('inf')
                                split_cont.append(len(split_cont))  # add a new path to the split_cont list
                                continue
                            else:
                                break
                        

                        # Calculate the total fee for each path
                        paths_with_fees = []
                        # Start iterating from candidate_paths[split_amount[-1] + 1]
                        start_index = split_cont[-1] + 1 if split_cont else 0  # Ensure split_amount is not empty
                        for path, flow, scores, total_fees in candidate_paths[start_index:]:
                            paths_with_fees.append((path, flow, total_fees))

                        # Sort paths by fee (cheapest first)
                        paths_with_fees.sort(key=lambda x: x[2])

                        # Select the top 5 cheapest paths
                        top_cheapest_paths = paths_with_fees[:5]

                        task_ids = []
                        # Create probing tasks for the top 5 cheapest paths
                        for path, flow, total_fees in top_cheapest_paths:
                            probing_task = ProbeTask(path, time.time() - simulation_start_time)
                            task_ids.append(probing_task.task_id)
                            probing_task_queue.put(probing_task)  
                        
                        pos = 0  
                        roll_back_pos = []
                        remain_amount = 0
                        rollback_channels_all = [[] for _ in range(len(split_cont))]  # initialize rollback channels for each path
                        while True:
                            time.sleep(0.03)
                            for i in split_cont:

                                if len(split_cont) == 0:       
                                        break
                                if pos >= len(candidate_paths[i][0]) - 1:
                                    split_cont.remove(i)
                                    continue

                                u, v = candidate_paths[i][0][pos], candidate_paths[i][0][pos + 1]
                                
                                lock_manager.acquire_channel_lock((u, v))
                                channels.append((u, v))  # record the channel that has been locked
                                # check if the channel has enough capacity
                                if G[u][v]['capacity'] < payment_split_amount[i]:
                                    has_capacity = False
                                    payment_task.message = f"Channel {u}-{v} has insufficient capacity: {G[u][v]['capacity']} < {payment_task.amount}"
                                    lock_manager.release_channel_lock((u, v))
                                    roll_back_pos.append(i)
                                    split_cont.remove(i)
                                    continue

                                # temporarily deduct the amount from the channel
                                G[u][v]['capacity'] -= payment_split_amount[i]
                                G[v][u]['capacity'] += payment_split_amount[i]

                                rollback_channels_all[i].append((u, v))  # record the channel that has been deducted
                                lock_manager.release_channel_lock((u, v))

                                # Update the usage frequency of the channel using EWMA
                                now = time.time()
                                if G[u][v]['last_used'] is not None:
                                    # Calculate the time difference since the last usage
                                    time_diff = now - G[u][v]['last_used']
                                    if time_diff > 0:
                                        # Apply EWMA formula
                                        previous_frequency = G[u][v].get('usage_frequency', 0)
                                        current_frequency = 1 / time_diff
                                        G[u][v]['usage_frequency'] = alpha * current_frequency + (1 - alpha) * previous_frequency
                                else:
                                    # If this is the first usage, set the frequency to a default value
                                    G[u][v]['usage_frequency'] = 1
                                
                                # Update the last used timestamp
                                G[u][v]['last_used'] = now
                                
                                # Add the fees for both directions
                                total_fee += (G[u][v]['fee'] + G[u][v]['rate'] * payment_split_amount[i])

                            if len(split_cont) == 0:  
                                    break
                            pos += 1

                        # If all channels have enough capacity, the payment is successful
                        if has_capacity:
                            payment_task.success = True
                            payment_task.fee = total_fee
                            payment_task.message = "Payment successful"

                            # Record the transaction timestamp for each channel in the path
                            timestamp = time.time()
                            
                            u, v = payment_task.path[0], payment_task.path[-1]
                            if u not in transaction_timestamps:
                                transaction_timestamps[u] = {}
                            if v not in transaction_timestamps[u]:
                                transaction_timestamps[u][v] = []
                            
                            # Append the timestamp
                            transaction_timestamps[u][v].append(timestamp)

                            # Limit the number of stored timestamps to avoid memory issues
                            if len(transaction_timestamps[u][v]) > 100:
                                transaction_timestamps[u][v].pop(0)
                            
                            v, u = payment_task.path[0], payment_task.path[-1]
                            if u not in transaction_timestamps:
                                transaction_timestamps[u] = {}
                            if v not in transaction_timestamps[u]:
                                transaction_timestamps[u][v] = []
                            
                            # Append the timestamp
                            transaction_timestamps[u][v].append(timestamp)

                            # Limit the number of stored timestamps to avoid memory issues
                            if len(transaction_timestamps[u][v]) > 100:
                                transaction_timestamps[u][v].pop(0)
                        else:
                            has_capacity = True
                            
                            # If any channel does not have enough capacity, the payment fails
                            pos = 0  # current position in the rollback channels

                            while True:
                                time.sleep(0.03)  # simulate processing time

                                for i in roll_back_pos:  # only roll back the channels that have been rolled back
                                    if pos >= len(rollback_channels_all[i]):
                                        continue  # if the channel has been rolled back, skip it

                                    u, v = rollback_channels_all[i][pos]  # aquire the channel to be rolled back
                                    try:
                                        lock_manager.acquire_channel_lock((u, v))

                                        if i < len(payment_split_amount):
                                            G[u][v]['capacity'] += payment_split_amount[i]
                                            G[v][u]['capacity'] -= payment_split_amount[i]
                                        else:
                                            G[u][v]['capacity'] += execute_payment[i - len(payment_split_amount)]
                                            G[v][u]['capacity'] -= execute_payment[i - len(payment_split_amount)]
                                    finally:
                                        lock_manager.release_channel_lock((u, v))

                                # if all channels have been rolled back, break the loop 
                                if all(pos >= len(rollback_channels_all[i]) for i in roll_back_pos):
                                    break

                                pos += 1  # proceed to the next channel to roll back
                            
                            # wait for all probing tasks to complete
                            wait_for_all_tasks(task_ids)

                            # aquire the probing results
                            with probing_results_lock:
                                results = [probing_results[task_id] for task_id in task_ids]
                            # print("Probing results:", len(results))

                            for path in results:
                                execute_paths.append(path["path"])
                                execute_payment.append(path["amount"])
                                
                            for p in range(len(execute_paths)):
                                rollback_channels_all.append([])
                                if has_capacity == False:
                                    break

                                for i in range(len(execute_paths[p]) - 1):

                                    time.sleep(0.03)
                                    u, v = execute_paths[p][i], execute_paths[p][i+1]
                                    
                                    lock_manager.acquire_channel_lock((u, v))
                                    channels.append((u, v))  # record the channel that has been locked
                                    
                                    if remain_amount < execute_payment[p]:
                                        execute_payment[p] = remain_amount

                                    # check if the channel has enough capacity
                                    if G[u][v]['capacity'] < execute_payment[p]:
                                        has_capacity = False
                                        payment_task.message = f"Channel {u}-{v} has insufficient capacity: {G[u][v]['capacity']} < {payment_task.amount}"
                                        lock_manager.release_channel_lock((u, v))
                                        break
                                    
                                    # temporarily deduct the amount from the channel
                                    G[u][v]['capacity'] -= execute_payment[p]
                                    G[v][u]['capacity'] += execute_payment[p]

                                    remain_amount -= execute_payment[p]

                                    rollback_channels_all[-1].append((u, v))  # record the channel that has been deducted

                                    lock_manager.release_channel_lock((u, v))
                                    
                                    # Add the fees for both directions
                                    total_fee += (G[u][v]['fee'] + G[u][v]['rate'] * execute_payment[p])
                                    if remain_amount < 0:
                                        break  

                            # If all channels have enough capacity, the payment is successful
                            if has_capacity:
                                payment_task.success = True
                                payment_task.fee = total_fee
                                payment_task.message = "Payment successful"
                        
                            else:
                                payment_task.fee = 0

                                pos = 0  # current position in the rollback channels
                                roll_back_pos = []  # record channels that have been rolled back

                                while True:
                                    time.sleep(0.03)  # simulate processing time

                                    for i in range(len(rollback_channels_all)):
                                        if i in roll_back_pos:
                                            continue  # skip already rolled back channels

                                        if pos >= len(rollback_channels_all[i]):
                                            roll_back_pos.append(i)  # if the channel has been rolled back, add it to the list
                                            continue

                                        u, v = rollback_channels_all[i][pos]  # aquire the channel to be rolled back
                                        
                                        try:
                                            if i < len(payment_split_amount):
                                                    lock_manager.acquire_channel_lock((u, v))
                                                    G[u][v]['capacity'] += payment_split_amount[i]
                                                    G[v][u]['capacity'] -= payment_split_amount[i]
                                            else:
                                                lock_manager.acquire_channel_lock((u, v))
                                                G[u][v]['capacity'] += execute_payment[i - len(payment_split_amount)]
                                                G[v][u]['capacity'] -= execute_payment[i- len(payment_split_amount)]
                                        finally:
                                            lock_manager.release_channel_lock((u, v))

                                    # if all channels have been rolled back, break the loop
                                    if len(roll_back_pos) == len(rollback_channels_all):
                                        break

                                    pos += 1  # next channel to roll back

                           
                finally:
                    # Release locks for all channels on the path
                    '''
                    for channel in channels:
                        lock_manager.release_channel_lock(channel)
                    '''
                    
                
            except Exception as e:
                print(f"Error processing payment: {str(e)}")    
                payment_task.message = f"Error processing payment: {str(e)}"
                import traceback
                error_traceback = traceback.format_exc()  # aquires the traceback
                print("Traceback details:")
                print(error_traceback)
    
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
def simulate_threaded_payments_poisson(payment_tasks, probing_task, num_threads=20, simulation_duration=60):
    """
    Simulate parallel payments in the Lightning Network using multiple threads.
    Payments arrive according to a Poisson process.
    simulation_duration: simulation duration in seconds
    """
    # Create task and result queues
    task_queue = queue.PriorityQueue()
    result_queue = queue.Queue()

    # Create a lock manager
    lock_manager = ChannelLockManager()

    # Create a stop event
    stop_event = threading.Event()

    # Record the simulation start time
    simulation_start_time = time.time()

    # set the simulation end time
    simulation_end_time = simulation_start_time + simulation_duration

    # Create and start worker threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=payment_worker,
            args=(task_queue, result_queue, lock_manager, stop_event, simulation_start_time)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)

    # Add payment tasks to the task queue
    for task in payment_tasks:
        task_queue.put(task)

    # Create and start probing worker threads
    threads1 = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=probing_worker_ps,
            args=(stop_event, simulation_start_time)
        )
        thread.daemon = True
        thread.start()
        threads1.append(thread)

    # Add probing tasks to the probing task queue
    for task in probing_task:
        probing_task_queue.put(task)

    # wait for all tasks to be completed
    while time.time() < simulation_end_time:
        time.sleep(1)  # check every second

    # Set the stop event
    stop_event.set()

    # Join all threads
    for thread in threads:
        thread.join(timeout=1)

    for thread in threads1:
        thread.join(timeout=1)

    # Get results from the result queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Count successful payments
    successful_payments = sum(1 for task in results if task.success)
    total_payments = len(results)
    avg_fee = sum(task.fee for task in results) / successful_payments if total_payments > 0 else 0

    # Calculate statistics
    if results:
        avg_processing_time = sum(task.processing_time for task in results) / len(results)
        avg_completion_time = sum(task.completion_time for task in results) / len(results)
        print(f"Average processing time: {avg_processing_time:.8f} seconds")
        print(f"Average completion time: {avg_completion_time:.2f} seconds")

    # Print payment results
    print(f"\nPayment results:")
    for task in sorted(results, key=lambda x: x.payment_id):
        print(task)

    return successful_payments, total_payments, results, avg_fee, avg_processing_time

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
def prepare_payment_tasks_poisson(payment_amounts, rate):
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
    previous_transactions = []
    for i, (amount, arrival_time) in enumerate(zip(payment_amounts, arrival_times)):
        sender, receiver = random_sender_receiver(G, previous_transactions)
        
        # Try to find a path in the routing table
        try:
            route_file = f"../routing_table/node{sender}"
            candidate_paths = get_paths_from_routing_table(route_file, sender, receiver)
            
            # Sort paths by flow
            candidate_paths.sort(key=lambda x: x[1], reverse=True)
            path = candidate_paths[0][0]  # Select the path with the highest flow
            execute_probing = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # default to 0
            if execute_probing == 2:
                path = nx.shortest_path(G, sender, receiver)
                
            # Create a payment task with arrival time
            task = PaymentTask(i+1, sender, receiver, amount, path, arrival_time)
            tasks.append(task)
    
        except nx.NetworkXNoPath:  
            print(f"Cannot find a path from {sender} to {receiver}, skipping payment {i+1}")  
        except Exception as e:  
            print(f"Error preparing payment {i+1}: {str(e)}")


    return tasks, probing_tasks

# Load payment amounts
def load_payment_amounts(file_path, num_payments):
    """
    Load random transaction amounts from creditcard.csv and map them to payment simulation.
    """
    df = pd.read_csv(file_path)
    
    # Randomly select `num_payments` rows from the 'Amount' column
    amounts = np.random.choice(df['Amount'].values, size=num_payments, replace=False)
    
    # Convert amounts to satoshis (assuming 1 USD = 11168.89 satoshis)
    amounts = amounts * 11168.89
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

def main():
    file_path = "../creditcard.csv"  # CSV file path

    clear_log_table() # Clear the log table

    # Clear the old routing table
    file_to_delete = "./trust_nodes.txt"
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)

    # Load the Lightning Network graph
    with open("../scripts/lightning_network.txt", "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=2)
            if len(parts) != 3:
                continue
            u, v, attr_str = parts
            u, v = int(u), int(v)

            # Parse capacity
            capacity_match = re.search(r"np\.int64\((\d+)\)", attr_str)
            capacity = int(capacity_match.group(1)) if capacity_match else 0

            # Parse fee
            fee_match = re.search(r"'fee': np\.float64\(([\d.]+)\)", attr_str)
            fee = float(fee_match.group(1)) if fee_match else 1.0  # Default fee is 1.0

            # Parse fee rate
            rate_match = re.search(r"'rate': np\.float64\(([\d.]+)\)", attr_str)
            rate = float(rate_match.group(1)) if rate_match else 0.00022  # Default rate is 0.00022

            # Add edge with attributes
            attrs = {
                'capacity': capacity,
                'fee': fee,
                'rate': rate,
                'usage': 0,
                'last_used': None,  # Initialize to None
                'usage_frequency': 0,  # Initialize to 0
            }
            G.add_edge(u, v, **attrs)

    # Output statistics  
    print(f"Number of nodes: {G.number_of_nodes()}")  
    print(f"Number of channels: {G.number_of_edges()}")  
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]  
    print(f"Average channel capacity: {np.mean(capacities):.2f} satoshis")  
    print(f"Median channel capacity: {np.median(capacities):.2f} satoshis")

    # Load payment amounts from creditcard.csv
    num_payments = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # default to 100
    payment_per_second = int(sys.argv[3]) if len(sys.argv) > 3 else 100 # default to 100
    payment_amounts = load_payment_amounts(file_path, num_payments)

    # Prepare payment tasks
    payment_tasks, probing_task = prepare_payment_tasks_poisson(payment_amounts, payment_per_second)
    
    print(f"Prepared {len(payment_tasks)} payment tasks and {len(probing_task)} probing tasks.")
    
    # Create a stop event for the trust score update thread
    stop_event = threading.Event()

    # Start the trust score update thread
    trust_thread = threading.Thread(
        target=update_trust_scores_thread,
        args=(G, trust_pairs, stop_event, 1)  # update_interval=1s
    )
    trust_thread.daemon = True
    trust_thread.start()

    # Simulate threaded payments
    start_time = time.time()
    successful_payments, total_payments, results, avg_fee, avg_processing_time = simulate_threaded_payments_poisson(
        payment_tasks, probing_task, num_threads=20
    )
    end_time = time.time()

    # Stop the trust score update thread
    stop_event.set()
    trust_thread.join()

    clear_log_table() # Clear the log table
    
    # Redirect print output to a log file
    log_file = open("simulation_output.log", "w")
    original_stdout = sys.stdout # Save a reference to the original standard output
    sys.stdout = log_file

    # Print the results
    success_rate = (successful_payments / total_payments) * 100
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    print(f"Successful Payments: {successful_payments}/{total_payments}")
    print(f"Average Fee: {avg_fee:.6f} satoshis")
    print(f"Execution Time: {avg_processing_time:.8f} seconds")

    # Close the log file
    log_file.close()
    sys.stdout = original_stdout # Reset standard output to original
    # visualize_network(G)
    # plot_payment_statistics(results)


print(sys.argv[0])
print(sys.argv[1])
print(sys.argv[2])
print(sys.argv[3])
print(sys.argv[4])
print(sys.argv[5])
print(sys.argv[6])
print(sys.argv[7])
print(sys.argv[8])

# Call the main function of simulator_thread.py
main()

