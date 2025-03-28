import os
import shutil
import re  # Used for parsing numbers
from collections import deque

def delete_directory(directory):
    """Delete a directory (if it exists)"""
    if os.path.exists(directory):
        shutil.rmtree(directory)

def bfs(graph, residual_graph, vs, vd):
    """Perform BFS to search for available paths"""
    queue = deque([vs])
    parent = {vs: None}
    while queue:
        node = queue.popleft()
        for neighbor, capacity in residual_graph[node].items():
            if neighbor not in parent and capacity > 0:
                parent[neighbor] = node
                if neighbor == vd:
                    return parent
                queue.append(neighbor)
    return None

def candidate_path_computation(graph, capacity, base_fee, vs):
    """Compute candidate paths"""
    candidate_paths = {}
    for vd in graph:
        if vd == vs:
            continue
        residual_graph = {node: {neighbor: capacity.get((node, neighbor), 0) for neighbor in graph[node]} for node in graph}
        candidate_paths[vd] = []
        
        while True:
            parent = bfs(graph, residual_graph, vs, vd)
            if not parent:
                break
            
            path = []
            node = vd
            while node != vs:
                path.append(node)
                node = parent[node]
            path.append(vs)
            path.reverse()

            Y_p = min(residual_graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
            base_fee_sum = sum(base_fee.get((path[i], path[i + 1]), 0) for i in range(len(path) - 1))
            
            if Y_p > 0:
                candidate_paths[vd].append((path, Y_p, base_fee_sum))
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if v not in residual_graph:
                        residual_graph[v] = {}
                    if u not in residual_graph[v]:
                        residual_graph[v][u] = 0
                    residual_graph[u][v] -= Y_p
                    residual_graph[v][u] += Y_p
            else:
                break
        for u in list(residual_graph):
            for v in list(residual_graph[u]):
                if residual_graph[u][v] <= 0:
                    del residual_graph[u][v]
    
    return candidate_paths

# Clear the old routing table
directory_to_delete = "../routing_table"
delete_directory(directory_to_delete)

# Initialize graph, capacity, base_fee
graph = {}
capacity = {}
base_fee = {}

# Read lightning_network.txt
with open("lightning_network.txt", "r") as file:
    for line in file:
        if not line.strip():
            continue
        parts = line.split()
        node1, node2 = int(parts[0]), int(parts[1])
        
        # Use regular expressions to parse `capacity`
        match = re.search(r"np\.int64\((\d+)\)", line)
        if match:
            capacity_value = int(match.group(1))  # Extract the numeric part
        else:
            print(f"Parsing error: {line.strip()}")
            continue
        
        fee = 1  # Assume base fee is 1, can be adjusted
        
        # Build the graph structure
        if node1 not in graph:
            graph[node1] = []
        if node2 not in graph[node1]:
            graph[node1].append(node2)
        if node2 not in graph:
            graph[node2] = []
        if node1 not in graph[node2]:
            graph[node2].append(node1)

        # Set capacity
        capacity[(node1, node2)] = capacity_value
        capacity[(node2, node1)] = capacity_value

        # Set base fee
        base_fee[(node1, node2)] = fee
        base_fee[(node2, node1)] = fee

# Create the routing table
folder_name = "../routing_table"
os.makedirs(folder_name, exist_ok=True)

for i in graph:
    vs = i
    all_paths = candidate_path_computation(graph, capacity, base_fee, vs)
    with open(f"{folder_name}/node{vs}", "w") as file:
        for vd, paths in all_paths.items():
            file.write(f"Paths from node{vs} to node{vd}:\n")
            for path, flow, fee in paths:
                path_str = ' '.join(f"node{n}" for n in path)
                file.write(f"  Path: {path_str}, Flow: {flow}, Fee: {fee}\n")

print("Routing table has been created!")