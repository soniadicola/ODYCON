# optimization.py
#comment to test

import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from preferendus import GeneticAlgorithm

class Optimization:
    def __init__(self,
                 paths_actual_network,
                 opt_mitigation_df,
                 target_duration,
                 delay,
                 penalty,
                 incentive,
                 penalty_nuisance,
                 incentive_nuisance,
                 w1,
                 w2,
                 w3,
                 x_points_1,
                 p_points_1,
                 x_points_2,
                 p_points_2,
                 x_points_3,
                 p_points_3
    ):
        
        self.paths_actual_network = paths_actual_network
        self.opt_mitigation_df = opt_mitigation_df
        self.target_duration = target_duration
        self.delay = delay
        self.penalty = penalty
        self.incentive = incentive
        self.penalty_nuisance = penalty_nuisance
        self.incentive_nuisance = incentive_nuisance
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.x_points_1 = x_points_1
        self.p_points_1 = p_points_1
        self.x_points_2 = x_points_2
        self.p_points_2 = p_points_2
        self.x_points_3 = x_points_3
        self.p_points_3 = p_points_3


    # Objective functions
    def objective_time(self, variables):
        """
        Objective function that describes time as a function of the project
        duration after implementing mitigation measures.

        Args:
            variables (array): Design variable values per member of the population.

        Returns:
            time (1D-array): Calculated project duration for the members of the population.
            paths_df (pandas.DataFrame): DataFrame containing path information of mitigated network.
        """

        # Get the number of members of population
        num_members = variables.shape[0]

        # Create paths_df for each member of population
        paths_df = self.paths_actual_network.copy()
        paths_df = pd.concat([paths_df] * num_members, ignore_index=True)

        # Repeat each member of population for all paths
        variables_repeated = np.repeat(variables, len(self.paths_actual_network), axis=0)

        # Create a mask of all mitigation measures on all paths of all members of population
        mask = np.array([
            [mit in path for path in paths_df['path']]
            for mit in self.opt_mitigation_df['mit_act_relation']
        ])

        # Calculate the mitigation for all paths and all members
        mitigation_effect = (
                self.opt_mitigation_df['mit_capacity_actual'].values[:, None]
                * variables_repeated.T
                * mask
        )

        # Subtract the mitigation effect from the duration
        paths_df['duration'] -= mitigation_effect.sum(axis=0)

        # Reshape durations to (num_members, num_paths)
        durations = paths_df['duration'].values.reshape(num_members, -1)

        # Get the maximum duration for each member
        time = durations.max(axis=1)

        return time, paths_df


    def objective_cost(self, variables, time):
        """
        Objective function that describes the project mitigation cost, as a function of
        the implemented mitigation measures.

        Args:
            variables (array): Design variable values per member of the population.
            time (array): Calculated project duration per members of the population.

        Returns:
            total_cost (1D-array): Calculated cost for the members of the population.
        """

        # Extract all mitigation costs from opt_mitigation_df
        mitigation_costs = self.opt_mitigation_df["mit_cost_actual"]

        # Calculate the cost for each member of the population
        cost = np.dot(variables, mitigation_costs)

        # Penalty and reward
        # Target duration
        t_target = np.ones(len(variables)) * self.target_duration

        # Calculate delta
        delta = time - t_target

        # if delta is positive: penalty for late completion
        dneg = np.where(delta > 0, delta, 0)

        # if delta is negative: reward for early completion
        dpos = np.where(delta < 0, t_target - time, 0)

        # Calculate the total cost including penalties and rewards
        total_cost = cost + dneg * self.penalty - dpos * self.incentive

        return total_cost
    

    def objective_nuisance(self, variables, time):
        """
        Objective function that describes the traffic user nuisance, as a function of
        the implemented mitigation measures.

        Args:
            variables (array): Design variable values per member of the population.
            time (array): Calculated project duration per members of the population.

        Returns:
            total_nuisance (1D-array): Calculated expeirence level for the members of the population.
        """

        # Extract all mitigation costs from opt_mitigation_df
        mitigation_nuisance = self.opt_mitigation_df["mit_nuisance_actual"].values

        # Calculate the cost for each member of the population
        nuisance = np.dot(variables, mitigation_nuisance)
        
        # Calculate the maximum possible traffic nuisance
        max_possible_nuisance = np.sum(mitigation_nuisance)
        
        # Normalize the traffic nuisance to the 0 to 10 scale
        normalized_nuisance = (nuisance / max_possible_nuisance) * 10
        
        # Target duration
        t_target = np.ones(len(variables)) * self.target_duration

        # Calculate delta
        delta = time - t_target

        # if delta is positive: penalty for late completion
        dneg = np.where(delta > 0, delta, 0)

        # if delta is negative: reward for early completion
        dpos = np.where(delta < 0, t_target - time, 0)

        # Calculate the total nuisance including penalties and rewards
        total_nuisance = normalized_nuisance + dneg * self.penalty_nuisance - dpos * self.incentive_nuisance

        # Ensure total nuisance is within the 0 to 10 range
        total_nuisance = np.clip(total_nuisance, 0, 10)

        return total_nuisance
    

    def objective(self, variables):
        """
        Objective function that is fed to the GA.

        Args:
            variables (array): Design variable values per member of the population.

        Returns:
            1D-array: Aggregated preference scores for the members of the population.
        """

        # calculate objectives
        time, _ = self.objective_time(variables)
        cost = self.objective_cost(variables, time)
        nuisance = self.objective_nuisance(variables, time)

        # calculate preference scores based on objective values
        p_cost = pchip_interpolate(self.x_points_1, self.p_points_1, cost)
        p_time = pchip_interpolate(self.x_points_2, self.p_points_2, time)
        p_nuisance = pchip_interpolate(self.x_points_3, self.p_points_3, nuisance)

        # initiate solver
        return [self.w1, self.w2, self.w3], [p_cost, p_time, p_nuisance]

    # Optimization function
    def optimization_function(self, info=False):
        """
        Function that initiates the GA.
        """
        
        # Define boundary for all mitigation measures
        bounds = [[0, 1] for _ in range(len(self.opt_mitigation_df))]

        # Define constraints
        cons = []

        # make dictionary with parameter settings for the GA
        options = {
            'n_pop': len(self.opt_mitigation_df) * 10,
            'max_stall': 10,
            'aggregation': 'IMAP',
            'mutation_rate_order': 4,
            'var_type_mixed': ['int' for _ in range(len(self.opt_mitigation_df))],
            'lsd_aggregation': "fast"
        }

        # list to save the results from every run to
        save_array_IMAP = list()

        # initiate and run the GA
        ga = GeneticAlgorithm(objective=self.objective, constraints=cons, bounds=bounds, options=options)
        _, design_variables_IMAP, _ = ga.run(verbose=False)
        save_array_IMAP.append(design_variables_IMAP)

        # make numpy array of results, to allow for array splicing
        variable = np.array(save_array_IMAP)

        return variable