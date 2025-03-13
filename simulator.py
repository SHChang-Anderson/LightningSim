import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# 設置隨機種子以確保可重複性
np.random.seed(42)

# 隨機選擇發送者和接收者的函數
def random_sender_receiver(G):
    nodes = list(G.nodes())  # 獲取所有節點的列表
    sender = random.choice(nodes)  # 隨機選取發送者
    receiver = random.choice(nodes)  # 隨機選取接收者
    while sender == receiver:  # 確保發送者和接收者不同
        receiver = random.choice(nodes)
    return sender, receiver

# 函數：生成對數正態分布的通道容量
def generate_channel_capacities(num_channels, mean_capacity, median_capacity):
    """
    根據對數正態分布生成通道容量。
    參數：
    - num_channels: 通道數量
    - mean_capacity: 通道容量的平均值（satoshis）
    - median_capacity: 通道容量的中位數（satoshis）
    返回：
    - capacities: 通道容量列表
    """
    mu = np.log(median_capacity)
    sigma = np.sqrt(2 * (np.log(mean_capacity) - mu))
    capacities = np.random.lognormal(mean=mu, sigma=sigma, size=num_channels)
    capacities = np.round(capacities).astype(int)
    return capacities

# 函數：生成 scale-free 網絡並分配通道容量
def generate_lightning_network(num_nodes, m, mean_capacity, median_capacity):
    """
    生成一個模擬的 Lightning Network 拓撲環境。
    參數：
    - num_nodes: 節點數量
    - m: BA 模型中每個新節點的連接數
    - mean_capacity: 通道容量的平均值（satoshis）
    - median_capacity: 通道容量的中位數（satoshis）
    返回：
    - G: 包含節點和帶有容量屬性的邊的 networkx 圖
    """
    G = nx.barabasi_albert_graph(num_nodes, m)
    num_channels = G.number_of_edges()
    capacities = generate_channel_capacities(num_channels, mean_capacity, median_capacity)
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['capacity'] = capacities[i]
    return G

# 函數：模擬支付
def simulate_payment(G, sender, receiver, amount):
    """
    模擬 Lightning Network 中的支付。
    參數：
    - G: networkx 圖
    - sender: 發送者節點
    - receiver: 接收者節點
    - amount: 支付金額（satoshis）
    返回：
    - success: 支付是否成功
    - path: 支付路徑（如果成功）
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

# 函數：從 creditcard.csv 讀取數據並映射到支付金額
def load_payment_amounts(file_path, num_payments):
    """
    從 creditcard.csv 讀取交易金額並映射到支付模擬。
    參數：
    - file_path: CSV 文件路徑
    - num_payments: 需要模擬的支付次數
    返回：
    - amounts: 支付金額列表（satoshis）
    """
    df = pd.read_csv(file_path)
    amounts = df['Amount'].head(num_payments).values * 100000  # 將金額轉換為 satoshis（假設 1 USD = 100,000 satoshis）
    amounts = np.round(amounts).astype(int)
    return amounts

# 函數：可視化網絡
def visualize_network(G):
    """
    可視化網絡拓撲。
    參數：
    - G: networkx 圖
    """
    pos = nx.spring_layout(G)
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color=capacities, edge_cmap=plt.cm.Blues)
    plt.show()

# 主程式
if __name__ == "__main__":
    # 設定參數
    num_nodes = 1000  # 節點數量
    m = 2  # BA 模型參數
    mean_capacity = 20000000  # 平均通道容量為 2000 萬 satoshis
    median_capacity = 5000000  # 中位數為 500 萬 satoshis
    file_path = "creditcard.csv"  # CSV 文件路徑

    # 生成網絡
    G = generate_lightning_network(num_nodes, m, mean_capacity, median_capacity)

    # 輸出統計信息
    print(f"節點數量: {G.number_of_nodes()}")
    print(f"通道數量: {G.number_of_edges()}")
    capacities = [G[u][v]['capacity'] for u, v in G.edges()]
    print(f"平均通道容量: {np.mean(capacities):.2f} satoshis")
    print(f"中位數通道容量: {np.median(capacities):.2f} satoshis")

    # 從 creditcard.csv 讀取支付金額
    num_payments = 1000  # 模擬 5 次支付
    payment_amounts = load_payment_amounts(file_path, num_payments)

    # 模擬多次支付
    sender = 0
    receiver = 99
    for i, amount in enumerate(payment_amounts):
        sender, receiver = random_sender_receiver(G)
        success, path = simulate_payment(G, sender, receiver, amount)
        if success:
            print(f"第 {i+1} 次支付成功！金額: {amount} satoshis, 路徑: {path}")
        else:
            print(f"第 {i+1} 次支付失敗！金額: {amount} satoshis")

    # 可視化網絡（可選）
    visualize_network(G)