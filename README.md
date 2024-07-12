# Odycon - Open Design & Dynamic Control

Project management is  becoming increasingly complex due to a shift of contractual responsibilities to 
contractors, broader project scopes with associated increase in interfaces, and the 
increasing influence of (local) stakeholders. This calls for adaptive decision support 
both for the planning and execution phase in order to find the best fitting solution for 
multiple (and sometimes changing) objectives. A stochastic simulation and
multi-objective optimisation framework (Odycon - Open Design & DYNamic CONtrol) has been developed that 
enables optimisation of both strategic planning and/or dynamic control cases. Odycon
is a decision-support framework that automates the selection of planning and control
variables, taking into consideration multiple stakeholder objectives and constraints.
To enable this, both Monte-Carlo simulation (MCS) and the multi-objective preference based 
IMAP optimisation are integrated. Odycon takes a next step in computer aided design for 
operations management into the future.

To this end, two Python based models were developed: one for a pure strategic planning application 
in an offshore transport and installation project (see [SYPL](SYPL/README.md)), and another for a pure mitigation 
control application in an  inland-infrastructure construction project (see [MICO](MICO/README.md)). Both applications 
prove their advances towards concurrent and associative design and decision-making, offering best fit-for common purpose 
synthesis for different complex project phases.

The optimisation is based on the Preferendus principles ([Wolfert, 2023](https://doi.org/10.3233/RIDS10); 
[Van Heukelum et al., 2024](https://doi.org/10.1080/15732479.2023.2297891)). 
For preference aggregation as part of the optimisation, the [A-fine Aggregator](https://github.com/Boskalis-python/a-fine-aggregator) algorithm is used.

The Odycon concept, and the associated Python models are developed by Lukas Teuber, with the help of Harold van Heukelum and Rogier Wolfert.

## License
This repository is licensed under the [MIT license](https://choosealicense.com/licenses/mit/).