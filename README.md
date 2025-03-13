### DESCRIPTION

# LightningSim
**LightningSim** is a tool designed to simulate Lightning Network payments, with an emphasis on channel capacity management. It creates a scale-free network topology, allocates channel capacities, and simulates payment processes to help users explore the mechanics of the Lightning Network.

## Core Features
- **Network Generation**: Uses the BA model to create a scale-free network.
- **Capacity Allocation**: Assigns channel capacities following a log-normal distribution.
- **Payment Simulation**: Identifies payment paths using the shortest path algorithm and updates channel capacities accordingly.
- **Data Reading**: Imports payment amounts from a CSV file.
- **Visualization**: Displays the network topology and capacity changes.

## Use Cases
- Analyze channel capacity management in the Lightning Network.
- Simulate payment processes and study capacity dynamics.
- Investigate how network topology affects payment success rates.

## Dependencies
- Python 3.x
- `networkx`
- `numpy`
- `matplotlib`
- `pandas`

## Installation
Install the required libraries using:
```bash
pip install networkx numpy matplotlib pandas
