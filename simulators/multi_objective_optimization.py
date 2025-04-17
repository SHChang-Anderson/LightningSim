import optuna
import matplotlib.pyplot as plt
import random
import time
from test_simulation import run_simulation

# simulate a function that takes parameters and returns two values
def run_your_code(param1, param2, param3, param4, param5):
    # Run the simulation with the given parameters
    success_rate, execution_time, avg_fee = run_simulation(10000, 5000, 3, param1, param2, param3, param4, param5)
    return success_rate, avg_fee

# Multi-objective optimization function
def objective(trial):
    param1 = trial.suggest_float("param1", 0.0, 100.0)
    param2 = trial.suggest_float("param2", 0.0, 10.0)
    param3 = trial.suggest_float("param3", 0.0, 10.0)
    param4 = trial.suggest_float("param4", 0.0, 0.5)
    param5 = trial.suggest_int("param5", 5, 10)
    result1, result2 = run_your_code(param1, param2, param3, param4, param5)
    return result1, result2  

# Create a study object
study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler()
)

# Excute the optimization
study.optimize(objective, n_trials=10)

# Get the best trials
sorted_best = sorted(study.best_trials, key=lambda t: (-t.values[0], t.values[1]))  # first by success rate, then by fee

# display the best trials
print("Top 5 Preferred Solutions (High Success Rate, Low Fee):")
for i, trial in enumerate(sorted_best[:5]):
    print(f"\n--- Rank #{i+1} ---")
    print(f"Success Rate: {trial.values[0]:.4f}, Fee: {trial.values[1]:.4f}")
    print(f"Params: {trial.params}")
    
'''
import plotly.express as px
import pandas as pd

# transform the study results into a DataFrame for visualization
params = ["param1", "param2", "param3", "param4", "param5"]
trials_data = [
    {**trial.params, "success_rate": trial.values[0], "avg_fee": trial.values[1]}
    for trial in study.trials if trial.values is not None
]
df = pd.DataFrame(trials_data)

# draw the Pareto front
for param in params:
    # param vs success_rate
    fig1 = px.scatter(
        df,
        x=param,
        y="success_rate",
        title=f"{param} vs Success Rate",
        labels={param: param, "success_rate": "Success Rate"},
        hover_data=["avg_fee"] + params
    )
    fig1.update_traces(marker=dict(color="teal", size=8), selector=dict(mode='markers'))
    fig1.show()

    # param vs avg_fee
    fig2 = px.scatter(
        df,
        x=param,
        y="avg_fee",
        title=f"{param} vs Average Fee",
        labels={param: param, "avg_fee": "Average Fee"},
        hover_data=["success_rate"] + params
    )
    fig2.update_traces(marker=dict(color="coral", size=8), selector=dict(mode='markers'))
    fig2.show()

import plotly.express as px
import pandas as pd
import numpy as np

# Let's assume you have a study object from Optuna
records = []
for trial in study.trials:
    if trial.values is not None:
        records.append({
            "param1": trial.params["param1"],
            "param2": trial.params["param2"],
            "param3": trial.params["param3"],
            "param4": trial.params["param4"],
            "param5": trial.params["param5"],
            "success_rate": trial.values[0],
            "avg_fee": trial.values[1]
        })

df = pd.DataFrame(records)

# Transform success_rate to [0, 1]
min_fee = df["avg_fee"].min()
df["fee_size"] = df["avg_fee"] - min_fee + 1  # avoid zero size

# draw 3D scatter plot
fig = px.scatter_3d(
    df,
    x="param1",
    y="param2",
    z="param3",
    color="success_rate",      
    size="fee_size",           
    hover_data=["param4", "param5", "success_rate", "avg_fee"],
    title="Multi-Objective Optimization - Param1/2/3 vs. Success/Cost",
    color_continuous_scale="Viridis"
)

fig.update_traces(marker=dict(opacity=0.8))
fig.update_layout(scene=dict(
    xaxis_title='Param1',
    yaxis_title='Param2',
    zaxis_title='Param3'
))
fig.show()
'''
