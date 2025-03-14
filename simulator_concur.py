import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import ast
import re
from collections import deque

# 設定隨機種子以獲得可重現的結果
np.random.seed(42)

def get_paths_from_routing_table(filename, source, destination):
    """
    從路由表檔案中提取從指定源節點到目標節點的所有路徑。

    參數:
    - filename (str): 路由表檔案的路徑。
    - source (int): 源節點。
    - destination (int): 目標節點。

    返回:
    - 包含 (path, flow) 的列表，其中:
      - path (list): 路徑中的節點列表。
      - flow (int): 該路徑的最大流量。
    """
    paths = []
    if not os.path.exists(filename):
        print(f"找不到路由表檔案 {filename}.")
        return paths

    with open(filename, "r") as file:
        is_target_section = False  # 追蹤我們是否在正確的部分
        for line in file:
            line = line.strip()
            
            # 識別部分標頭
            if line.startswith(f"Paths from node{source} to node{destination}:"):
                is_target_section = True
                continue  # 移至下一行

            # 如果新部分開始，停止處理
            if is_target_section and line.startswith("Paths from node"):
                break

            # 處理相關部分內的路徑
            if is_target_section and "Path:" in line:
                parts = line.split(", ")
                path_str = parts[0].split(":")[1].strip()  # 提取路徑
                flow_str = parts[1].split(":")[1].strip()  # 提取流量
                
                # 將路徑轉換為整數列表
                path = [int(node.replace("node", "")) for node in path_str.split()]
                flow = int(flow_str)
                paths.append((path, flow))

    return paths

# 隨機選擇發送者和接收者的函數
def random_sender_receiver(G):
    nodes = list(G.nodes())  # 獲取所有節點的列表
    sender = random.choice(nodes)  # 隨機選擇發送者
    receiver = random.choice(nodes)  # 隨機選擇接收者
    while sender == receiver:  # 確保發送者和接收者不同
        receiver = random.choice(nodes)
    return sender, receiver

# 定義一個表示待處理支付的類
class PendingPayment:
    def __init__(self, payment_id, sender, receiver, amount, path):
        self.payment_id = payment_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.path = path
        
    def __str__(self):
        return f"Payment {self.payment_id}: {self.amount} satoshis from {self.sender} to {self.receiver} via {self.path}"

# 檢查路徑是否有足夠的容量
def check_path_capacity(G, path, amount):
    """
    檢查路徑中的所有通道是否有足夠的容量。
    
    參數:
    - G: NetworkX 圖
    - path: 節點路徑
    - amount: 支付金額
    
    返回:
    - 布林值，表示路徑是否有足夠的容量
    """
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G[u][v]['capacity'] < amount:
            return False
    return True

# 更新通道容量
def update_channel_capacities(G, path, amount):
    """
    更新路徑中的通道容量。
    
    參數:
    - G: NetworkX 圖
    - path: 節點路徑
    - amount: 支付金額
    """
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        G[u][v]['capacity'] -= amount
        G[v][u]['capacity'] += amount

# 模擬並行支付
def simulate_parallel_payments(G, payment_batches):
    """
    模擬閃電網路中的並行支付。
    
    參數:
    - G: NetworkX 圖
    - payment_batches: 支付批次列表，每個批次包含多個支付
    
    返回:
    - 成功支付的數量
    - 支付總數
    """
    successful_payments = 0
    total_payments = 0
    
    for batch_idx, batch in enumerate(payment_batches):
        print(f"\n處理批次 {batch_idx+1}/{len(payment_batches)} (包含 {len(batch)} 筆支付)")
        
        # 初始化待處理支付隊列
        pending_payments = deque(batch)
        total_payments += len(batch)
        
        # 處理此批次中的所有支付
        while pending_payments:
            current_payment = pending_payments.popleft()
            
            # 檢查路徑是否有足夠的容量
            if check_path_capacity(G, current_payment.path, current_payment.amount):
                # 更新通道容量
                update_channel_capacities(G, current_payment.path, current_payment.amount)
                successful_payments += 1
                print(f"支付成功! {current_payment}")
            else:
                print(f"支付失敗! {current_payment} (容量不足)")
        
        # 在處理下一批次之前，可以添加一些延遲或其他邏輯
    
    return successful_payments, total_payments

# 加載支付金額
def load_payment_amounts(file_path, num_payments):
    """
    從 creditcard.csv 加載交易金額並映射到支付模擬。
    
    參數:
    - file_path: CSV 檔案路徑
    - num_payments: 要模擬的支付數量
    
    返回:
    - amounts: 支付金額列表 (satoshis)
    """
    df = pd.read_csv(file_path)
    amounts = df['Amount'].head(num_payments).values * 100000  # 轉換為 satoshis (假設 1 USD = 100,000 satoshis)
    amounts = np.round(amounts).astype(int)
    return amounts

# 可視化網路
def visualize_network(G):
    """
    可視化網路拓撲。
    
    參數:
    - G: NetworkX 圖
    """
    pos = nx.spring_layout(G)
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color=capacities, edge_cmap=plt.cm.Blues)
    plt.show()

# 準備支付批次
def prepare_payment_batches(G, payment_amounts, batch_size=10):
    """
    準備支付批次。
    
    參數:
    - G: NetworkX 圖
    - payment_amounts: 支付金額列表
    - batch_size: 每批次的支付數量
    
    返回:
    - 支付批次列表
    """
    batches = []
    current_batch = []
    
    for i, amount in enumerate(payment_amounts):
        sender, receiver = random_sender_receiver(G)
        
        # 嘗試獲取路由表中的路徑
        try:
            route_file = f"routing_table/node{sender}"
            candidate_paths = get_paths_from_routing_table(route_file, sender, receiver)
            
            if candidate_paths:
                # 按流量排序路徑
                candidate_paths.sort(key=lambda x: x[1], reverse=True)
                path = candidate_paths[0][0]  # 選擇流量最高的路徑
            else:
                # 如果找不到路由表路徑，使用最短路徑
                path = nx.shortest_path(G, sender, receiver)
                
            # 創建待處理支付對象
            payment = PendingPayment(i+1, sender, receiver, amount, path)
            current_batch.append(payment)
            
            # 如果達到批次大小或是最後一個支付，開始新批次
            if len(current_batch) >= batch_size or i == len(payment_amounts) - 1:
                batches.append(current_batch)
                current_batch = []
                
        except nx.NetworkXNoPath:
            print(f"無法找到從 {sender} 到 {receiver} 的路徑，跳過支付 {i+1}")
        except Exception as e:
            print(f"處理支付 {i+1} 時出錯: {e}")
    
    return batches

# 主程式
if __name__ == "__main__":
    file_path = "creditcard.csv"  # CSV 檔案路徑

    # 生成網路
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
                print(f"跳過格式不正確的行: {line.strip()}")
    
    # 輸出統計信息
    print(f"節點數量: {G.number_of_nodes()}")
    print(f"通道數量: {G.number_of_edges()}")
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]
    print(f"平均通道容量: {np.mean(capacities):.2f} satoshis")
    print(f"中位數通道容量: {np.median(capacities):.2f} satoshis")

    # 從 creditcard.csv 加載支付金額
    num_payments = 1000  # 模擬 1000 筆支付
    payment_amounts = load_payment_amounts(file_path, num_payments)

    # 準備支付批次，每批次 10 筆支付
    payment_batches = prepare_payment_batches(G, payment_amounts, batch_size=10)
    
    # 模擬並行支付
    successful_payments, total_payments = simulate_parallel_payments(G, payment_batches)

    # 計算並打印成功率
    success_rate = (successful_payments / total_payments) * 100
    print(f"\n支付成功率: {success_rate:.2f}%")
    print(f"成功支付: {successful_payments}/{total_payments}")

    # 可視化網路 (可選)
    # visualize_network(G)