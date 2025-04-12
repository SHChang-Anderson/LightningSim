import optuna
import matplotlib.pyplot as plt
import random
import time
from test_simulation import run_simulation

# simulate a function that takes parameters and returns two values
def run_your_code(param1, param2, param3):
    # Run the simulation with the given parameters
    success_rate, execution_time, avg_fee = run_simulation(10000, 7000, 3, param1, param2, param3)
    return success_rate, avg_fee

# Multi-objective optimization function
def objective(trial):
    param1 = trial.suggest_float("param1", 0.0, 100.0)
    param2 = trial.suggest_float("param2", 0.0, 10.0)
    param3 = trial.suggest_float("param3", 0.0, 10.0)
    
    result1, result2 = run_your_code(param1, param2, param3)
    return result1, result2  

# Create a study object
study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler()
)

# Excute the optimization
study.optimize(objective, n_trials=10)

# Show the best trials
print("Pareto Front:")
for i, trial in enumerate(study.best_trials):
    print(f"\n---  {i+1} ---")
    print(f"{trial.params}")
    print(f"Target: {trial.values}")

# Plotting the results
result1_list = [trial.values[0] for trial in study.trials if trial.values is not None]
result2_list = [trial.values[1] for trial in study.trials if trial.values is not None]

plt.figure(figsize=(8,6))
plt.scatter(result1_list, result2_list, c="blue", label="All Trials")
plt.xlabel("Objective 1: Error Rate ↓")
plt.ylabel("Objective 2: Runtime ↓")
plt.title("Pareto Front (NSGA-II)")
plt.grid(True)
plt.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 收集參數與目標值
param1_list = [trial.params["param1"] for trial in study.trials if trial.values is not None]
param2_list = [trial.params["param2"] for trial in study.trials if trial.values is not None]
result1_list = [trial.values[0] for trial in study.trials if trial.values is not None]
result2_list = [trial.values[1] for trial in study.trials if trial.values is not None]

# 將數據轉換為網格格式
param1_grid, param2_grid = np.meshgrid(
    np.linspace(min(param1_list), max(param1_list), 50),
    np.linspace(min(param2_list), max(param2_list), 50)
)
result1_grid = np.interp(param1_grid, param1_list, result1_list)
result2_grid = np.interp(param2_grid, param2_list, result2_list)

# 繪製 3D 曲面圖
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用 result1 作為 z 軸
surf = ax.plot_surface(param1_grid, param2_grid, result1_grid, cmap="viridis", alpha=0.8)
ax.set_xlabel("Param1")
ax.set_ylabel("Param2")
ax.set_zlabel("Objective 1: Error Rate ↓")
ax.set_title("3D Surface Plot of Objective 1")
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

# 繪製第二個目標值的 3D 曲面圖
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用 result2 作為 z 軸
surf = ax.plot_surface(param1_grid, param2_grid, result2_grid, cmap="plasma", alpha=0.8)
ax.set_xlabel("Param1")
ax.set_ylabel("Param2")
ax.set_zlabel("Objective 2: Runtime ↓")
ax.set_title("3D Surface Plot of Objective 2")
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
