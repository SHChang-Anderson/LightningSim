import subprocess
import matplotlib.pyplot as plt

def run_simulation(num_payments, payments_per_sec, execute_probing):
    """
    Run the simulator_thread.py script with the specified parameters.

    Parameters:
    - num_payments (int): Number of payments to simulate.
    - execute_probing (int): Whether to enable probing (0 or 1).

    Returns:
    - success_rate (float): The success rate of the simulation.
    """
    try:
        # Run the simulator_thread.py script with subprocess
        result = subprocess.run(
            ["python3", "simulator_thread.py", str(execute_probing), str(num_payments), str(payments_per_sec)],
            capture_output=True,
            text=True
        )
        
        # Parse the output to extract the success rate
        output = result.stdout
        success_rate = -1
        execution_time = -1
        for line in output.split("\n"):
            if "Success Rate" in line:
                success_rate = float(line.split(":")[1].strip().replace("%", ""))
            if "Average processing time:" in line: # Extract the execution time
                execution_time = float(line.split(":")[1].strip().replace("seconds", ""))    
            if success_rate != -1 and execution_time != -1:
                return success_rate, execution_time
    except Exception as e:
        print(f"Error running simulation: {e}")
        return None

def test_simulation():
    """
    Test the simulator_thread.py script with different payment counts and probing settings.
    """
    payment_counts = range(1000, 5000, 1000)  # From 1000 to 9000 payments, step 1000
    success_rates_probing_0 = []
    success_rates_probing_1 = []

    # Run simulations for execute_probing = 0
    print("Running simulations with execute_probing = 0...")
    for num_payments in payment_counts:
        success_rate, execution_time = run_simulation(num_payments, 1000, 0)
        if success_rate is not None:
            success_rates_probing_0.append(success_rate)
            print(f"Payments: {num_payments}, Success Rate (Probing=0): {success_rate:.2f}%, Execution Time: {execution_time:.8f}s")
        else:
            success_rates_probing_0.append(0)  # Default to 0 if simulation fails

    # Run simulations for execute_probing = 1
    print("\nRunning simulations with execute_probing = 1...")
    for num_payments in payment_counts:
        success_rate, execution_time = run_simulation(num_payments, 1000, 1)
        if success_rate is not None:
            success_rates_probing_1.append(success_rate)
            print(f"Payments: {num_payments}, Success Rate (Probing=1): {success_rate:.2f}%, Execution Time: {execution_time:.8f}s")
        else:
            success_rates_probing_1.append(0)  # Default to 0 if simulation fails

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(payment_counts, success_rates_probing_0, label="Probing=0", marker="o")
    plt.plot(payment_counts, success_rates_probing_1, label="Probing=1", marker="o")
    plt.xlabel("Number of Payments")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate vs Number of Payments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("success_rate_comparison.png")  # Save the plot as an image
    plt.show()

if __name__ == "__main__":
    test_simulation()