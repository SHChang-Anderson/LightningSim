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
        if node not in residual_graph: 
            continue
        for neighbor, capacity in residual_graph[node].items():
            if neighbor not in parent and capacity > 0:
                parent[neighbor] = node
                if neighbor == vd:
                    return parent
                queue.append(neighbor)
    return None

def candidate_path_computation(graph, capacity, base_fee, fee_rate, vs):
    """Compute edge-disjoint candidate paths (based on Algorithm 1)"""
    candidate_paths = {}
    for vd in graph:
        if vd == vs:
            continue

        # Initialize residual graph
        residual_graph = {node: {neighbor: capacity.get((node, neighbor), 0) for neighbor in graph[node]} for node in graph}
        candidate_paths[vd] = []

        while True:
            parent = bfs(graph, residual_graph, vs, vd)
            if not parent:
                break

            # from parent dictionary, reconstruct the path from vs to vd
            path = []
            node = vd
            while node != vs:
                path.append(node)
                node = parent[node]
            path.append(vs)
            path.reverse()

            # compute Y(p)
            min_capacities = [residual_graph[path[i]][path[i + 1]] for i in range(len(path) - 1)]
            Y_p = min(min_capacities)

            # if Y(p) > 0, add the path to candidate_paths
            if Y_p > 0:
                base_fee_sum = sum(base_fee.get((path[i], path[i + 1]), 0) for i in range(len(path) - 1))
                fee_rate_sum = sum(fee_rate.get((path[i], path[i + 1]), 0) for i in range(len(path) - 1))
                candidate_paths[vd].append((path, Y_p, base_fee_sum, fee_rate_sum))

                # update the residual graph
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    residual_graph[u][v] -= Y_p
                    residual_graph[v][u] = residual_graph.get(v, {}).get(u, 0) + Y_p

            else:
                # Y(p) <= 0, find the bottleneck edge
                bottleneck_index = min(range(len(min_capacities)), key=lambda i: min_capacities[i])
                u, v = path[bottleneck_index], path[bottleneck_index + 1]
                residual_graph[u][v] = 0
                residual_graph[v][u] = capacity.get((v, u), 0)

            # remove edges with zero capacity
            for u in list(residual_graph):
                for v in list(residual_graph[u]):
                    if residual_graph[u][v] <= 0:
                        del residual_graph[u][v]
                if not residual_graph[u]:
                    del residual_graph[u]

    return candidate_paths

# Clear the old routing table
directory_to_delete = "../routing_table"
delete_directory(directory_to_delete)

# Initialize graph, capacity, base_fee, and fee_rate
graph = {}
capacity = {}
base_fee = {}
fee_rate = {}

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
        
        # Parse fee and fee rate
        fee_match = re.search(r"'fee': np\.float64\(([\d.]+)\)", line)
        rate_match = re.search(r"'rate': np\.float64\(([\d.]+)\)", line)
        fee = float(fee_match.group(1)) if fee_match else 1.0  # Default fee is 1.0
        rate = float(rate_match.group(1)) if rate_match else 0.00022  # Default rate is 0.00022

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

        # Set base fee and fee rate
        base_fee[(node1, node2)] = fee
        base_fee[(node2, node1)] = fee
        fee_rate[(node1, node2)] = rate
        fee_rate[(node2, node1)] = rate

# Create the routing table
folder_name = "../routing_table"
os.makedirs(folder_name, exist_ok=True)

for i in graph:
    vs = i
    all_paths = candidate_path_computation(graph, capacity, base_fee, fee_rate, vs)
    with open(f"{folder_name}/node{vs}", "w") as file:
        for vd, paths in all_paths.items():
            file.write(f"Paths from node{vs} to node{vd}:\n")
            for path, flow, base_fee_sum, fee_rate_sum in paths:
                path_str = ' '.join(f"node{n}" for n in path)
                file.write(f"  Path: {path_str}, Flow: {flow}, Base Fee Sum: {base_fee_sum:.6f}, Fee Rate Sum: {fee_rate_sum:.6f}\n")

print("Routing table has been created!")