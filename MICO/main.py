"""
Python code of the multi-objective dynamic control support tool (MiCo).

Copyright (c) Lukas Teuber, 2024.
"""

from MICO.simulation import MiCo

def main():
    # Define options for MiCo
    options = {
        "filename": "MICO/examples/example_case3_correlation.xlsx",
        "runs": 100,
        "target_duration": None,
        "min_duration": 500,
        "penalty": 3000,
        "incentive": 3000,
        "penalty_nuisance": 0.0,
        "incentive_nuisance": 0,
        "w1": 1.0, # cost
        "w2": 0.0, # time
        "w3": 0.0, # traffic user nuisance
        "info_setting": False,
    }

    # Create an instance of MiCo
    mico_instance = MiCo(options)

    # Run the simulation
    mico_instance.run_simulation()

if __name__ == "__main__":
    main()
