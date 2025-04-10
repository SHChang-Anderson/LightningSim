import subprocess
import matplotlib.pyplot as plt
from simulator_thread import main as simulator_main

def run_simulation(num_payments, payments_per_sec, execute_probing):
    """
    Run the simulator_thread.py script directly by calling its main function.

    Parameters:
    - num_payments (int): Number of payments to simulate.
    - payments_per_sec (int): Number of payments per second.
    - execute_probing (int): Whether to enable probing (0, 1, or 2).

    Returns:
    - success_rate (float): The success rate of the simulation.
    - execution_time (float): The average execution time per payment.
    """
    try:
        # Simulate command-line arguments
        import sys
        sys.argv = [
            "simulator_thread.py",  # Script name
            str(execute_probing),   # Probing mode
            str(num_payments),      # Number of payments
            str(payments_per_sec)   # Payments per second
        ]

        # Call the main function of simulator_thread.py
        simulator_main()

        # Parse the output from the global variables or logs
        # Assuming the main function prints the success rate and execution time
        # You can modify this part to directly return values from the main function
        success_rate = None
        execution_time = None
        avg_fee = None
        with open("simulation_output.log", "r") as log_file:
            for line in log_file:
                if "Success Rate" in line:
                    success_rate = float(line.split(":")[1].strip().replace("%", ""))
                if "Execution Time" in line:
                    execution_time = float(line.split(":")[1].strip().replace("seconds", ""))
                if "Average Fee" in line:
                    avg_fee = float(line.split(":")[1].strip().replace("satoshis", ""))
                if success_rate is not None and execution_time is not None:
                    break
        
        return success_rate, execution_time, avg_fee

    except Exception as e:
        print(f"Error running simulation: {e}")
        return None, None

def test_simulation():
    """
    Test the simulator_thread.py script with different payment counts and probing settings.
    """
    payment_counts = range(1000, 6000, 1000)  # From 1000 to 9000 payments, step 1000
    success_rates_probing_0 = []
    success_rates_probing_1 = []
    success_rates_probing_2 = []

    exe_time_probing_0 = []
    exe_time_probing_1 = []
    exe_time_probing_2 = []

    avg_fee_0 = []
    avg_fee_1 = []
    avg_fee_2 = []

    # Run simulations for execute_probing = 0
    print("Running simulations with execute_probing = 0...")
    for num_payments in payment_counts:
        success_rate, execution_time, avg_fee = run_simulation(num_payments, 5000, 0)
        if success_rate is not None:
            success_rates_probing_0.append(success_rate)
            exe_time_probing_0.append(execution_time)
            avg_fee_0.append(avg_fee)
            print(f"Payments: {num_payments}, Success Rate (Probing=0): {success_rate:.2f}%, Execution Time: {execution_time:.8f}s, print Avg Fee: {avg_fee:.2f} satoshis")
        else:
            success_rates_probing_0.append(0)  # Default to 0 if simulation fails
            exe_time_probing_0.append(0)
            avg_fee_0.append(0)

    # Run simulations for execute_probing = 1
    print("\nRunning simulations with execute_probing = 1...")
    for num_payments in payment_counts:
        success_rate, execution_time, avg_fee = run_simulation(num_payments, 5000, 1)
        if success_rate is not None:
            success_rates_probing_1.append(success_rate)
            exe_time_probing_1.append(execution_time)
            avg_fee_1.append(avg_fee)
            print(f"Payments: {num_payments}, Success Rate (Probing=1): {success_rate:.2f}%, Execution Time: {execution_time:.8f}s, print Avg Fee: {avg_fee:.2f} satoshis")
        else:
            success_rates_probing_1.append(0)  # Default to 0 if simulation fails
            exe_time_probing_1.append(0)
            avg_fee_1.append(0)

    # Run simulations for execute_probing = 3
    print("\nRunning simulations with execute_probing = 3...")
    for num_payments in payment_counts:
        success_rate, execution_time, avg_fee = run_simulation(num_payments, 5000, 3)
        if success_rate is not None:
            success_rates_probing_2.append(success_rate)
            exe_time_probing_2.append(execution_time)
            avg_fee_2.append(avg_fee)
            print(f"Payments: {num_payments}, Success Rate (Probing=1): {success_rate:.2f}%, Execution Time: {execution_time:.8f}s, avg_fee: {avg_fee:.2f} satoshis")
        else:
            success_rates_probing_2.append(0)  # Default to 0 if simulation fails
            exe_time_probing_2.append(0)
            avg_fee_2.append(0)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(payment_counts, success_rates_probing_0, label="Probing=0", marker="o")
    plt.plot(payment_counts, success_rates_probing_1, label="Probing=1", marker="o")
    plt.plot(payment_counts, success_rates_probing_2, label="Use cal", marker="o")
    plt.xlabel("Number of Payments")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate vs Number of Payments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("success_rate_comparison.png")  # Save the plot as an image
    plt.show()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(payment_counts, exe_time_probing_0, label="Probing=0", marker="o")
    plt.plot(payment_counts, exe_time_probing_1, label="Probing=1", marker="o")
    plt.plot(payment_counts, exe_time_probing_2, label="Use cal", marker="o")
    plt.xlabel("Number of Payments")
    plt.ylabel("Execution Time (s) / Payment")
    plt.title("Execution Time vs Number of Payments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Exe.png")  # Save the plot as an image
    plt.show()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(payment_counts, avg_fee_0, label="Probing=0", marker="o")
    plt.plot(payment_counts, avg_fee_1, label="Probing=1", marker="o")
    plt.plot(payment_counts, avg_fee_2, label="Use cal", marker="o")
    plt.xlabel("Number of Payments")
    plt.ylabel("Average Fee (satoshis)")
    plt.title("Average Fee vs Number of Payments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Exe.png")  # Save the plot as an image
    plt.show()

if __name__ == "__main__":
    test_simulation()