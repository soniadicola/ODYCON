"""
Python code of the multi-objective strategic planning support tool (SYPL).

"""

from SYPL.simulation import Simulation

if __name__ == "__main__":
    # Define simulation parameters
    runs = 200
    w1 = 0.30  # project duration
    w2 = 0.35  # cost
    w3 = 0.15  # fleet utilization
    w4 = 0.20  # sustainability

    # Display resulting preference curves for every iteration
    info_setting = False

    # Create an instance of the Simulation class
    simulation_instance = Simulation(runs, w1, w2, w3, w4, info_setting)

    # Plot the initial preference curves
    simulation_instance.plot_preference_curves()

    # Run the simulation
    simulation_instance.run_simulation()