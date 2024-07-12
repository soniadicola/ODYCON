# MiCo - Dynamic Project Mitigation Control

This repository contains the code of the state of the art multi-objective mitigation control tool named MiCo.
MiCo is a probabilistic optimisation tool for dynamic control of construction projects.
It is based upon the MitC concept ([Kammouh et al., 2021](https://doi.org/10.1061/(ASCE)CO.1943-7862.0002126); 
[Kammouh et al., 2022](https://doi.org/10.1016/j.autcon.2022.104450)).  It integrates preference based IMAP optimisation 
and probabilistic network planning, incorporating activity uncertainties and correlation, risk events and contractual performance schemes.

## Getting Started
1. **Examples Folder:**
    The examples folder contains demonstrative cases that can use with the software. Note, this version of the framework is specifically build towards example case 3 and serves as a proof of concept, demonstrating the ability to optimise towards multiple objectives.

2. **Configuration:**
    Before running the software, you need to adapt options to match your project requirements.
    - filename (str): The name of the project Excel file.
    - runs (int): Number of simulations.
    - target_duration (float): Target duration of the project. If None it the critical_path duration it taken.
    - min_duration (float): Optimistic duration of project according to BETA-pert.
    - penalty (float): Daily penalty towards objective cost.
    - incentive (float): Daily incentive towards objective cost.
    - penalty_nuisance (float): Daily penalty towards the traffic nuisance.
    - incentive_nuisance (float): Daily incentive towards the traffic nuisance.
    - w1 (float): Weight of objective cost.
    - w2 (float): Weight of objective time.
    - w3 (float): Weight of objective traffic nuisance.
    - info_setting (bool): Whether to display preference curves during simulation (default: False).

    The structure of the Excel input file should the same as for the examples.
    
3. Running the Software:
    To run the software, execute the main.py file.
    ```bash
    python -m MICO.main
    ```

## Build
Note, since this tool is supposed to be built towards the individual project and stakeholder-oriented needs it requires individual code adaptation towards different or further objectives. This script is a proof of concept.