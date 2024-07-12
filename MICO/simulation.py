"""
Python code of the multi-objective mitigation control support tool (MiCo).

Copyright (c) Lukas Teuber, 2024.
"""

from matplotlib import pyplot as plt
from scipy.interpolate import pchip_interpolate
import numpy as np
from preference_functions import PreferenceCurve
import time as clock
import warnings
from MICO.import_project import ProjectDataImporter
from MICO.optimization import Optimization
from MICO.interpolation import linear_interpolation
from MICO.plotting import plot_frequency
from MICO.plotting import plot_frequency_combinations
from MICO.plotting import plot_cdf
from MICO.plotting import plot_pdf
from MICO.network import Network
from MICO.network import ProbabilisticNetwork
from MICO.network import critical_from_path

warnings.filterwarnings("ignore")

class MiCo:
    def __init__(self, options):
        """
        Multi-objective mitigation control support tool (MiCo) for project scheduling optimization.

        Attributes:
            filename (str): The name of the project file.
            runs (int): Number of simulations.
            target_duration (float): Target duration of the project.
            min_duration (float): Optimistic duration of the project.
            penalty (float): Daily penalty towards objective cost.
            incentive (float): Daily incentive towards objective cost.
            w1 (float): Weight of objective cost.
            w2 (float): Weight of objective time.
            info_setting (bool): Whether to display preference curves during simulation (default: False).

        Methods:
            run_simulation(): Runs the simulation process.
        """
        self.filename = options.get('filename')
        self.runs = options.get('runs')
        self.target_duration = options.get('target_duration')
        self.min_duration = options.get('min_duration')
        self.penalty = options.get('penalty')
        self.incentive = options.get('incentive')
        self.penalty_nuisance = options.get('penalty_nuisance')
        self.incentive_nuisance = options.get('incentive_nuisance')
        self.w1 = options.get('w1')
        self.w2 = options.get('w2')
        self.w3 = options.get('w3')
        self.info_setting = options.get('info_setting', False)

    def run_simulation(self):
        # --- Parse Data
        project = ProjectDataImporter(self.filename)

        activities = project.activities_df
        mitigations = project.mitigation_df
        risks = project.risk_df
        correlations = project.correlation_df


        # --- Deterministic network
        deterministic_network = Network(activities, "act_duration_ml")

        # Generate all paths
        paths_deterministic_network = deterministic_network.get_paths()

        # Get critical path
        critical_path_info = deterministic_network.get_critical_path(paths_deterministic_network)
        critical_path_duration = critical_path_info["critical_path_duration"]

        if self.target_duration is None:
            self.target_duration = critical_path_duration

        # Plot Gantt chart
        deterministic_network.plot_network(project_start_date='2022-01-20')

        # --- Random draw
        network = ProbabilisticNetwork(activities, mitigations, risks, correlations)

        (shared_activity_duration,
        act_duration_no_corr,
        mit_capacity_actual,
        risk_actual)= network.rand_draw(self.runs)


        # --- MC simulation
        # Initiate timer
        start_time = clock.time()

        # Initiate solution storage
        mitigation_list = list()  # critical mitigation measures
        activity_list_original = list()  # critical activities (original)
        activity_list_tentative = list()  # critical activities (tentative)

        crit_path_original = list()  # critical project path ID (original)
        crit_path_tentative = list()  # critical project path ID (tentative)

        opt_mit_time = list()  # duration tentative (optimal mitigation measures)
        perm_mit_time = list()  # duration permanent (all mitigation measures)
        no_mit_time = list()  # duration original (no measures)

        opt_mit_cost = list()  # cost tentative (optimal mitigation measures)
        perm_mit_cost = list()  # cost permanent (all mitigation measures)

        opt_mit_nuisance = list()  # traffic nuisance (optimal mitigation measures)
        perm_mit_nuisance = list()  # traffic nuisance (all mitigation measures)

        # set count for runs without delay
        runs_without_delay = 0

        for i in range(self.runs):

            print(f"MC progress: {(((i + 1) / self.runs) * 100):.1f}%")

            # --- Probabilisitic network
            # Evaluate the duration of activities considering duration, corelation and risks occurence
            opt_activities_df = activities
            opt_activities_df["duration_total"] = shared_activity_duration[:,i] + act_duration_no_corr[:,i] + risk_actual[:,i]

            # Calculate the criteria-cost value based on the actual drawn mitigation capacity for a
            # given iteration, the cost of a mitigation measure corresponds to the mitigation capacity
            opt_mitigations_df = mitigations
            opt_mitigations_df["mit_capacity_actual"] = mit_capacity_actual[:,i]
            opt_mitigations_df["mit_cost_actual"] = linear_interpolation(
                opt_mitigations_df["mit_capacity_actual"],
                opt_mitigations_df["mit_capacity_opt"],
                opt_mitigations_df["mit_capacity_pes"],
                opt_mitigations_df["mit_cost_min"],
                opt_mitigations_df["mit_cost_max"]
            )

            # Calculate the criteria-traffic nuisance
            opt_mitigations_df["mit_nuisance_actual"] = linear_interpolation(
                opt_mitigations_df["mit_capacity_actual"],
                opt_mitigations_df["mit_capacity_opt"],
                opt_mitigations_df["mit_capacity_pes"],
                opt_mitigations_df["mit_nuisance_min"],
                opt_mitigations_df["mit_nuisance_max"]
            )

            opt_mitigations_df = opt_mitigations_df.loc[:,
                                                        ["mit_ID",
                                                        "mit_capacity_actual",
                                                        "mit_cost_actual",
                                                        "mit_nuisance_actual",
                                                        "mit_act_relation"]
                                                        ]

            # Generate network
            actual_network = Network(opt_activities_df, "duration_total")

            # Generate all paths
            paths_actual_network = actual_network.get_paths()

            # Generate critical path (before optimization)
            actual_critical_path = actual_network.get_critical_path(paths_actual_network)
            duration = actual_critical_path["critical_path_duration"]
            delay = duration - self.target_duration

            # if there is a delay, start the optimization
            if delay > 0:

                # Define the minimum preference of cost to be the sum of all possible mitigation measures costs
                p_cost_min = opt_mitigations_df["mit_cost_actual"].sum()
                p_cost_min = opt_mitigations_df["mit_cost_actual"].sum() + delay * self.penalty
                p_nuisance_min = 10

                # Set preference curves
                time_p = PreferenceCurve(self.min_duration, self.target_duration, delay)
                cost_p = PreferenceCurve(0.01, 0.01, p_cost_min)
                nuisance_p = PreferenceCurve(0.01, 0.01, p_nuisance_min)
                time_p.beta_PERT()
                cost_p.linear()
                nuisance_p.linear()
                x_points_1, p_points_1 = [list(cost_p.x_values), list(cost_p.y_values)]  # cost
                x_points_2, p_points_2 = [list(time_p.x_values), list(time_p.y_values)]  # time
                x_points_3, p_points_3 = [list(nuisance_p.x_values), list(nuisance_p.y_values)]  # nuisance

                # Run optimization
                opt = Optimization(paths_actual_network,
                                opt_mitigations_df,
                                self.target_duration,
                                delay,
                                self.penalty,
                                self.incentive,
                                self.penalty_nuisance,
                                self.incentive_nuisance,
                                self.w1,
                                self.w2,
                                self.w3,
                                x_points_1,
                                p_points_1,
                                x_points_2,
                                p_points_2,
                                x_points_3,
                                p_points_3)
                
                variable = opt.optimization_function()

                # calculate individual criteria outcome for the results of the GA
                time_res, paths_tentative_network = opt.objective_time(variable)
                cost_res = opt.objective_cost(variable, time_res)
                nuisance_res = opt.objective_nuisance(variable, time_res)

                # Gets critical path after optimization
                tentative_critical_path = critical_from_path(paths_tentative_network)

                # Display iteration results
                if self.info_setting is True:
                    # arrays for plotting continuous preference curves
                    c1 = np.linspace(min(x_points_1), max(x_points_1))
                    c2 = np.linspace(min(x_points_2), max(x_points_2))
                    c3 = np.linspace(min(x_points_3), max(x_points_3))

                    # calculate the preference functions
                    p1 = pchip_interpolate(x_points_1, p_points_1, c1)
                    p2 = pchip_interpolate(x_points_2, p_points_2, c2)
                    p3 = pchip_interpolate(x_points_3, p_points_3, c3)

                    # Calculate results
                    p1_res = pchip_interpolate(x_points_1, p_points_1, cost_res)
                    p2_res = pchip_interpolate(x_points_2, p_points_2, time_res)
                    p3_res = pchip_interpolate(x_points_3, p_points_3, nuisance_res)

                    # Create figure with specified size
                    fig = plt.figure(figsize=(17, 17), dpi=300)
                    gs = fig.add_gridspec(3, 2, wspace=0.1, hspace=0.2)

                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.plot(c1, p1, color='k', zorder=3, label='Preference curve')
                    ax1.scatter(cost_res, p1_res, label='Optimal solution', marker='D', color='g')
                    ax1.set_xlim((min(x_points_1) - 100, max(x_points_1) + 50))
                    ax1.set_ylim((0, 105))
                    ax1.set_title('Cost')
                    ax1.set_xlabel('Euro')
                    ax1.set_ylabel('Preference score')
                    ax1.legend()
                    ax1.grid()

                    ax2 = fig.add_subplot(gs[1, 0])
                    ax2.plot(c3, p3, color='k', zorder=3, label='Preference curve')
                    ax2.scatter(nuisance_res, p3_res, label='Optimal solution', marker='D', color='g')
                    ax2.set_xlim((min(x_points_3) , max(x_points_3)))
                    ax2.set_ylim((0, 105))
                    ax2.set_title('User traffic nuisance')
                    ax2.set_xlabel('User traffic nuisance')
                    ax2.set_ylabel('Preference score')
                    ax2.legend()
                    ax2.grid()

                    ax3 = fig.add_subplot(gs[2, 0])
                    ax3.plot(c2, p2, color='k', zorder=3, label='Preference curve')
                    ax3.scatter(time_res, p2_res, label='Optimal solution', marker='D', color='g')
                    ax3.axvline(self.target_duration)
                    ax3.axvline(self.target_duration + delay, color="#EE6677")
                    ax3.set_xlim((min(x_points_2) - 100, max(x_points_2) + 50))
                    ax3.set_ylim((0, 105))
                    ax3.set_title('Time')
                    ax3.set_xlabel('Days')
                    ax3.legend()
                    ax3.grid()

                    ax4 = fig.add_subplot(gs[0, 1])
                    ax4.axis('off')
                    table_4_data = [paths_actual_network.columns.to_list()] + paths_actual_network.values.tolist()
                    table_4 = ax4.table(cellText=table_4_data, cellLoc = 'center', loc='center')
                    table_4.auto_set_column_width(col=list(range(len(table_4_data[0]))))

                    ax5 = fig.add_subplot(gs[1, 1])
                    ax5.axis('off')
                    table_5_data = [paths_tentative_network.columns.to_list()] + paths_tentative_network.values.tolist()
                    table_5 = ax5.table(cellText=table_5_data, cellLoc = 'center', loc='center')
                    table_5.auto_set_column_width(col=list(range(len(table_5_data[0]))))

                    text_str = (f'Target duration: {self.target_duration} days.\n'
                                f'Project duration: {self.target_duration + delay} days.\n'
                                f'Delay: {delay} days.\n'
                                f'Measures used: {variable}\n'
                                f'Result criteria - time: {time_res[0]} days\n'
                                f'Result criteria - cost: {cost_res[0]} euros')
                    
                    ax6 = fig.add_subplot(gs[2, 1])
                    ax6.axis('off')
                    ax6.text(0, 1, text_str, va='top', ha='left', fontsize=12, transform=ax6.transAxes)

                    plt.savefig(f'results/preference_curves.png')

                # Store results of mitigation measures
                mitigation_list.append(((list(opt_mitigations_df["mit_ID"]) * variable).ravel()).tolist())

                # Critical activities (original - before measures)
                activity_list_original.append(list(actual_critical_path["critical_ID_list"]))

                # Critical activities (tentative - after measures)
                activity_list_tentative.append(list(tentative_critical_path["critical_ID_list"]))

                # critical project path ID (original - before measures)
                crit_path_original.append(actual_critical_path["critical_path_ID"])

                # critical project path ID (tentative - after measures)
                crit_path_tentative.append(tentative_critical_path["critical_path_ID"])

                # duration tentative (optimal mitigation measures)
                opt_mit_time.append(time_res[0])

                # duration permanent (all mitigation measures)
                perm_time_res, _ = opt.objective_time(variables=np.ones((1, len(opt_mitigations_df)), dtype=int))
                perm_mit_time.append(perm_time_res[0])

                # duration original (no measures)
                total_time_res = (duration)
                no_mit_time.append(total_time_res)

                # cost tentative (optimal mitigation measures)
                opt_mit_cost.append(cost_res[0])

                # cost permanent (all mitigation measures)
                perm_cost_res = opt.objective_cost(variables=np.ones((1, len(opt_mitigations_df)), dtype=int), 
                                                time=perm_time_res)
                perm_mit_cost.append(perm_cost_res[0])

                # traffic nuisance
                opt_mit_nuisance.append(nuisance_res[0])
                perm_mit_nuisance = opt.objective_nuisance(variables=np.ones((1, len(opt_mitigations_df)), dtype=int),
                                                    time=perm_time_res)

            # If there is no delay add to the count
            else:
                runs_without_delay += 1

        # End of MC
        print(
            f"---------- MONTE CARLO IS FINISHED AFTER {self.runs} ITERATIONS AND "
            f"{round((clock.time() - start_time) / 60)} MINUTES,"
            f"{runs_without_delay} RUN(S) WITHOUT DELAY-----------"
        )


        
        # --- Generate Plots
        # Plot frequency of occurrence of mitigation measures
        plot_frequency(
            elements=mitigations["mit_ID"],
            n_runs=self.runs,
            data1=mitigation_list,
            name="Mitigation measure ID"
        )

        # Plot frequency of occurence of combinations of measures
        plot_frequency_combinations(
            n_runs=self.runs,
            data=mitigation_list,
            name="Mitigation measure combinations"
        )

        # Plot criticality index of activities
        plot_frequency(
            elements=activities["act_ID"],
            n_runs=self.runs,
            data1=activity_list_original,
            data2=activity_list_tentative,
            data1_label='Original (before mitigation)',
            data2_label='Tentative (after optimization)',
            name="Activity ID"
        )

        # Plot criticality of project paths
        plot_frequency(
            elements=paths_deterministic_network["path_id"],
            n_runs=self.runs,
            data1=crit_path_original,
            data2=crit_path_tentative,
            data1_label='Original (before mitigation)',
            data2_label='Tentative (after optimization)',
            name="Path ID"
        )

        # Plot PDF criteria time
        plot_pdf(
            perm_mit_time,
            opt_mit_time,
            no_mit_time,
            plot_name="PDF - time",
            xaxis="Duration [days]",
        )

        # Plot CDF criteria time
        plot_cdf(
            opt_mit_time,
            perm_mit_time,
            no_mit_time,
            plot_name="CDF - time",
            xaxis="Time [d]",
            vline=self.target_duration,
        )

        # Plot PDF criteria cost
        plot_pdf(
            perm_mit_cost,
            opt_mit_cost,
            plot_name="PDF - cost",
            xaxis="Cost [euro]",
        )

        # Plot CDF criteria cost
        plot_cdf(
            opt_mit_cost,
            perm_mit_cost,
            plot_name="CDF - cost",
            xaxis="Cost [euro]",
            vline=None,
        )

        # Plot PDF criteria nuisance
        plot_pdf(
            perm_mit_nuisance,
            opt_mit_nuisance,
            plot_name="PDF - traffic nuisance",
            xaxis="Nuisance [-]",
        )

        # Plot CDF criteria nuisance
        plot_cdf(
            opt_mit_nuisance,
            perm_mit_nuisance,
            plot_name="CDF - traffic nuisance",
            xaxis="Nuisance [-]",
            vline=None,
        )