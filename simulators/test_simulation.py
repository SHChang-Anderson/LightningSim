import subprocess
import matplotlib.pyplot as plt
# {'param1': 8.163282840531622, 'param2': 3.164842908222707, 'param3': 2.5196407699778964, 'param4': 4.799002018614012}
# {'param1': 0.1467240216646526, 'param2': 88.60465960039095, 'param3': 92.81928625519423}
# {'param1': 7.540066433454607, 'param2': 1.5458277166017909, 'param3': 8.53511340709082}
# {'param1': 69.9121832103292, 'param2': 9.568959148843271, 'param3': 0.8334971354417864}
# 'param1': 92.57303178668649, 'param2': 0.0, 'param3': 0.0, 'param4': 0.41628467118225165, 'param5': 8
# {'param1': 96.09985339519194, 'param2': 2.10183642984912, 'param3': 0.8338142960594785, 'param4': 0.19995197310808, 'param5': 9}

# cheap
# 'param1': 1.0611725984543918, 'param2': 60.27367308783179, 'param3': 7.360643237372914, 'param4': 0.3626886219630006, 'param5': 2} 

def run_simulation(num_payments, payments_per_sec, execute_probing, param1=1.0611725984543918,param2 = 60.27367308783179, param3=7.360643237372914, param4=0.3626886219630006, param5=2): 
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
        print(f"Running simulation with {num_payments} payments, {payments_per_sec} payments/sec, execute_probing={execute_probing}...")    
        # create a timeout for the subprocess

        timeout = 500  # seconds
        cmd = [
            "python3", "simulator_thread.py",
            str(execute_probing),
            str(num_payments),
            str(payments_per_sec),
            str(param1),
            str(param2),    
            str(param3),
            str(param4),
            str(param5)
        ]

        # Run the command and capture the output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        print(result.stdout)
        print(result.stderr)
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
        
        return success_rate, execution_time , avg_fee

    except subprocess.TimeoutExpired:
        print(f"Timeout: The simulation took too long to complete.")
        return None, None, None
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, None, None

def test_simulation():
    """
    Test the simulator_thread.py script with different payment counts and probing settings.
    """
    payment_counts = range(1000, 5100, 1000)  # From 1000 to 9000 payments, step 1000
    TPS1 = range(10, 60, 10)  # From 1 to 100 TPS, step 10
    success_rates_probing_0 = []
    success_rates_probing_1 = []
    success_rates_probing_2 = []

    exe_time_probing_0 = []
    exe_time_probing_1 = []
    exe_time_probing_2 = []

    avg_fee_0 = []
    avg_fee_1 = []
    avg_fee_2 = []
    '''
    # Run simulations for execute_probing = 0
    print("Running simulations with execute_probing = 0...")
    for tps in TPS1:
        print("1")
        success_rate, execution_time, avg_fee = run_simulation(100000, tps, 0)
        if success_rate is not None:
            success_rates_probing_0.append(success_rate)
            exe_time_probing_0.append(execution_time)
            avg_fee_0.append(avg_fee)
            print(f"TPS: {tps}, Success Rate (Deter Pay): {success_rate:.2f}%, Execution Time: {execution_time:.8f}s, print Avg Fee: {avg_fee:.2f} satoshis")
        else:
            success_rates_probing_0.append(0)  # Default to 0 if simulation fails
            exe_time_probing_0.append(0)
            avg_fee_0.append(0)
    '''
    # Run simulations for execute_probing = 1
    print("\nRunning simulations with execute_probing = 1...")
    for tps in TPS1:
        print("2")
        success_rate, execution_time, avg_fee = run_simulation(100000, tps, 0)
        if success_rate is not None:
            success_rates_probing_1.append(success_rate)
            exe_time_probing_1.append(execution_time)
            avg_fee_1.append(avg_fee)
            print(f"Payments: {tps}, Success Rate (Probing=1): {success_rate:.2f}%, Execution Time: {execution_time:.8f}s, print Avg Fee: {avg_fee:.2f} satoshis")
        else:
            success_rates_probing_1.append(0)  # Default to 0 if simulation fails
            exe_time_probing_1.append(0)
            avg_fee_1.append(0)
    

    # Run simulations for execute_probing = 3
    print("\nRunning simulations with execute_probing = 3...")
    for tps in TPS1:
        print("3")
        success_rate, execution_time, avg_fee = run_simulation(100000, tps, 2)
        if success_rate is not None:
            success_rates_probing_2.append(success_rate)
            exe_time_probing_2.append(execution_time)
            avg_fee_2.append(avg_fee)
            print(f"TPS: {tps}, Success Rate (DReP): {success_rate:.2f}%, Execution Time: {execution_time:.8f}s, avg_fee: {avg_fee:.2f} satoshis")
        else:
            success_rates_probing_2.append(0)  # Default to 0 if simulation fails
            exe_time_probing_2.append(0)
            avg_fee_2.append(0)
    
    
    # simulation results
    # TPS = [10, 20, 30, 40, 50, 60]

    # Deter Pay results
    success_rates_probing_0 = [89.85, 85.88, 86.66, 87.45, 87.57]
    exe_time_probing_0 = [0.95304964, 1.06003440, 0.97741990, 0.93625529, 0.99946868]
    avg_fee_0 = [145.91, 141.61, 139.31, 140.90, 142.35]
    '''
    # Proposed results
    success_rates_probing_2 = [60.34, 62.05, 54.89, 55.17, 56.17, 51.87]
    exe_time_probing_2 = [0.10802969, 0.10447014, 0.10663722, 0.10894671, 0.10929702, 0.11170397]
    avg_fee_2 = [182.76, 176.14, 139.12, 133.44, 153.09, 123.06]
    '''
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(TPS1, success_rates_probing_0, label="Deter Pay", marker="o")
    plt.plot(TPS1, success_rates_probing_1, label="Paying via the Maximum Flow Path", marker="o")
    plt.plot(TPS1, success_rates_probing_2, label="Shortest Path", marker="o")
    plt.xlabel("Workload (TPS)")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate vs Workload (TPS)")
    plt.ylim(0, 100)  # Set y-axis range from 0 to 100%
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("success_rate_comparison.png")  # Save the plot as an image
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(TPS1, exe_time_probing_0, label="Deter Pay", marker="o")
    plt.plot(TPS1, exe_time_probing_1, label="Paying via the Maximum Flow Path", marker="o")
    plt.plot(TPS1, exe_time_probing_2, label="Shortest Path", marker="o")
    plt.xlabel("Workload (TPS)")
    plt.ylabel("Execution Time (s) / Payment")
    plt.title("Execution Time vs Workload (TPS)")
    plt.ylim(0, 2)  # Set y-axis range from 0 to 2 seconds
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Exe_T.png")  # Save the plot as an image

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(TPS1, avg_fee_0, label="Deter Pay", marker="o")
    plt.plot(TPS1, avg_fee_1, label="Paying via the Maximum Flow Path", marker="o")
    plt.plot(TPS1, avg_fee_2, label="Shortest Path", marker="o")
    plt.xlabel("Workload (TPS)")
    plt.ylabel("Average Fee (satoshis)")
    plt.title("Average Fee vs Workload (TPS)")
    plt.ylim(0, 600)  # Set y-axis range from 0 to 600 satoshis
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Fee.png")  # Save the plot as an image

if __name__ == "__main__":
    test_simulation()