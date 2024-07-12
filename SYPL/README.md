# SyPl - Strategic Planning

This folder contains the core elements of the strategic planning tool named SyPl. SyPl is a probabilistic optimization tool for a pure strategic planning application (no control variables). This exemplar case is based upon prior publication ([Wolfert, 2023](https://doi.org/10.3233/RIDS10); [Van Heukelum et al., 2024](https://doi.org/10.1080/15732479.2023.2297891)). 

It integrates preference based IMAP optimisation and probabilistic Monte-Carlo simulation, incorporating activity uncertainties and correlation and risk events enabling best-fit for common purpose project planning.

## Getting Started
1. **Configuration:**
    Before running the software, you need to adapt options to match your project requirements.
    - runs (int): Number of MS simulations.
    - w1 (float): Weight of objective project duration.
    - w2 (float): Weight of objective cost.
    - w3 (float): Weight of objective fleet utilization.
    - w4 (flaot): Weight of objective sustainability.
    - info_setting (bool): Whether to display preference curves during simulation (default: False).
    
2. **Running the Software:**
    To run the software, execute the main.py file.
    ```bash
    python -m SYPL.main
    ```

## Build
Note, this repository contains only one optimization example for proof of concept. The general methodology can however be build upon to suit all different kind or projects given the individual project objectives and constraints.