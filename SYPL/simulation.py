# simulation.py

import numpy as np
from scipy.interpolate import pchip_interpolate
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from collections import Counter
from SYPL.optimization import OptimizationClass
from preference_functions import PreferenceCurve


class Simulation:
    def __init__(self, runs, w1, w2, w3, w4, info_setting=False):
        self.runs = runs
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.info_setting = info_setting

        min_variance = 50
        target_duration = 90
        max_variance = 20

        time_p = PreferenceCurve(min_variance, target_duration, max_variance)
        time_p.beta_PERT()
        self.x_points_1, self.p_points_1 = [list(time_p.x_values), list(time_p.y_values)]  # time

        self.x_points_2, self.p_points_2 = [[9_500_000, 11_000_000, 17_000_000], [100, 20, 0]]
        self.x_points_3, self.p_points_3 = [[0, 0.6, 1], [100, 50, 0]]
        self.x_points_4, self.p_points_4 = [[3_200, 5_000, 10_200], [100, 40, 0]]

        # Arrays for plotting continuous preference curves
        self.c1 = np.linspace(min(self.x_points_1), max(self.x_points_1))
        self.c2 = np.linspace(min(self.x_points_2), max(self.x_points_2))
        self.c3 = np.linspace(0, 1)
        self.c4 = np.linspace(min(self.x_points_4), max(self.x_points_4))

        # Calculate the preference scores
        self.p1 = pchip_interpolate(self.x_points_1, self.p_points_1, self.c1)
        self.p2 = pchip_interpolate(self.x_points_2, self.p_points_2, self.c2)
        self.p3 = pchip_interpolate(self.x_points_3, self.p_points_3, self.c3)
        self.p4 = pchip_interpolate(self.x_points_4, self.p_points_4, self.c4)

    def plot_preference_curves(self):
        fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

        ax1.plot(self.c1, self.p1, label='Preference Function')
        ax1.set_ylim((0, 100))
        ax1.set_title('Project Duration')
        ax1.set_xlabel('Time [days]')
        ax1.set_ylabel('Preference function outcome')
        ax1.grid()

        ax2.plot(self.c2, self.p2)
        ax2.set_ylim((0, 100))
        ax2.set_title('Installation Costs')
        ax2.set_xlabel('Costs [€]')
        ax2.grid()

        ax3.plot(self.c3, self.p3)
        ax3.set_ylim((0, 100))
        ax3.set_title('Fleet Utilization')
        ax3.set_xlabel('Number of vessels [-]')
        ax3.grid()

        ax4.plot(self.c4 * 1e-3, self.p4)
        ax4.set_ylim((0, 100))
        ax4.set_title(r'$CO_2$ emissions')
        ax4.set_xlabel(r'$CO_2$ emission [$10^3$ tonnes]')
        ax4.grid()

        plt.savefig('results/preference_curves.png')

    def run_simulation(self):
        start_time = time.time()

        mean_duration = []
        variance_duration = []
        std_duration = []

        variables = list()
        c1_res_list = list()
        c2_res_list = list()
        c3_res_list = list()
        c4_res_list = list()

        base_seed = 2

        for i in range(self.runs):
            print(f"MC progress: {(((i + 1) / self.runs) * 100):.1f}%")

            iteration_seed = base_seed + i
            
            opt_class = OptimizationClass(
                iteration_seed=iteration_seed,
                w1=self.w1,
                w2=self.w2,
                w3=self.w3,
                w4=self.w4,
                x_points_1=self.x_points_1,
                p_points_1=self.p_points_1,
                x_points_2=self.x_points_2,
                p_points_2=self.p_points_2,
                x_points_3=self.x_points_3,
                p_points_3=self.p_points_3,
                x_points_4=self.x_points_4,
                p_points_4=self.p_points_4,
            )
            variable = opt_class.optimization_function()

            # Calculate individual criteria outcome
            c1_res, t_res_1, t_res_2, t_res_3 = opt_class.objective_time(variable[:, 0], variable[:, 1], variable[:, 2])
            c2_res = opt_class.objective_costs(variable[:, 3], variable[:, 4], t_res_1, t_res_2, t_res_3)
            c3_res = opt_class.objective_fleet_utilization(variable[:, 0], variable[:, 1], variable[:, 2])
            c4_res = opt_class.objective_co2(variable[:, 0], variable[:, 1], variable[:, 2], t_res_1, t_res_2, t_res_3)
            
            # Store results
            variables.append(variable)
            c1_res_list.append(c1_res)
            c2_res_list.append(c2_res)
            c3_res_list.append(c3_res)
            c4_res_list.append(c4_res)

            # Display iteration results if info_setting is True
            if self.info_setting:
                self.display_iteration_results(c1_res, c2_res, c3_res, c4_res, t_res_1, t_res_2, t_res_3)

            # Update metrics after each iteration
            mean_duration.append(np.mean(c1_res_list))
            variance_duration.append(np.var(c1_res_list))
            std_duration.append(np.std(c1_res_list))

        # End of MC
        print(f"---------- MONTE CARLO IS FINISHED AFTER {self.runs} ITERATIONS AND {round((time.time() - start_time) / 60)} MINUTES -----------")

        # Plot results
        self.plot_results(c1_res_list, c2_res_list, c3_res_list, c4_res_list, variables, mean_duration, variance_duration, std_duration)

    def display_iteration_results(self, c1_res, c2_res, c3_res, c4_res, t_res_1, t_res_2, t_res_3):
        print(f"time of individual vessels: {t_res_1, t_res_2, t_res_3}")
        print(f'cost: {c2_res}')
        print(f"utilization: {c3_res}")
        print(f"Co2: {c4_res}")

        p1_res = pchip_interpolate(self.x_points_1, self.p_points_1, c1_res)
        p2_res = pchip_interpolate(self.x_points_2, self.p_points_2, c2_res)
        p3_res = pchip_interpolate(self.x_points_3, self.p_points_3, c3_res)
        p4_res = pchip_interpolate(self.x_points_4, self.p_points_4, c4_res)

        fig2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

        ax1.plot(self.c1, self.p1, label='Preference Function')
        for i in range(len(c1_res)):
            ax1.scatter(c1_res[i], p1_res[i])
        ax1.set_ylim((0, 100))
        ax1.set_title('Project Duration')
        ax1.set_xlabel('Time [days]')
        ax1.set_ylabel('Preference function outcome')
        ax1.grid()
        fig2.legend()

        ax2.plot(self.c2, self.p2)
        for i in range(len(c2_res)):
            ax2.scatter(c2_res[i], p2_res[i])
        ax2.set_ylim((0, 100))
        ax2.set_title('Installation Costs')
        ax2.set_xlabel('Costs [€]')
        ax2.set_ylabel('Preference function outcome')
        ax2.grid()

        ax3.plot(self.c3, self.p3)
        for i in range(len(c3_res)):
            ax3.scatter(c3_res[i], p3_res[i])
        ax3.set_ylim((0, 100))
        ax3.set_title('Fleet Utilization')
        ax3.set_xlabel('Number of vessels [-]')
        ax3.set_ylabel('Preference function outcome')
        ax3.grid()

        ax4.plot(self.c4 * 1e-3, self.p4)
        for i in range(len(c4_res)):
            ax4.scatter(c4_res[i] * 1e-3, p4_res[i])
        ax4.set_ylim((0, 100))
        ax4.set_title(r'$CO_2$ emissions')
        ax4.set_xlabel(r'$CO_2$ emission [$10^3$ tonnes]')
        ax4.set_ylabel('Preference function outcome')
        ax4.grid()

        plt.savefig('results/preference_curves_result.png')

    def plot_results(self, c1_res_list, c2_res_list, c3_res_list, c4_res_list, variables, mean_duration, variance_duration, std_duration):
        flattened_c1_res_list = np.array(c1_res_list).flatten()
        self.plot_pdf(flattened_c1_res_list, name='Project Duration', xaxis='Time [days]')

        flattened_c2_res_list = np.array(c2_res_list).flatten()
        self.plot_pdf(flattened_c2_res_list, name='Installation costs', xaxis='Costs [€]')

        flattened_c3_res_list = np.array(c3_res_list).flatten()
        self.plot_pdf(flattened_c3_res_list, name='Fleet utilization', xaxis='Number of vessels [-]')

        flattened_c4_res_list = np.array(c4_res_list).flatten()
        self.plot_pdf(flattened_c4_res_list, name='$CO_2$ emissions', xaxis='$CO_2$ emission [tonnes]')

        self.plot_cdf(flattened_c1_res_list, name='Project Duration', xaxis='Time [days]')
        self.plot_cdf(flattened_c2_res_list, name='Installation costs', xaxis='Costs [€]')
        self.plot_cdf(flattened_c3_res_list, name='Fleet utilization', xaxis='Number of vessels [-]')
        self.plot_cdf(flattened_c4_res_list, name='$CO_2$ emissions', xaxis='$CO_2$ emission [tonnes]')

        vessel_counts = np.vstack(variables)[:, :3].astype(int)
        self.plot_vessel_frequency(vessel_counts)

        anchor_data = np.vstack(variables)[:, 3:]
        self.plot_pdf(anchor_data[:, 0], name='Anchor Diameter PDF', xaxis='Diameter')
        self.plot_pdf(anchor_data[:, 1], name='Anchor Length PDF', xaxis='Length')

        self.plot_convergence(mean_duration, variance_duration, std_duration)

    def plot_pdf(self, data, name='PDF Plot', figsize=(15, 10), xaxis='xaxis'):
        mu1, std1 = norm.fit(data)
        x1 = np.linspace(min(data), max(data), 100)
        pdf1 = norm.pdf(x1, mu1, std1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x1, pdf1, color="#EE6677", linewidth=2)
        ax.hist(data, bins=50, density=True, color="#EE6677", alpha=0.4)

        ax.set_title(name, fontsize=20)
        ax.set_xlabel(xaxis, fontsize=20)
        ax.set_ylabel('Probability Distribution Function', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True)
        plt.savefig(f'results/pdf_{name}.png')

    def plot_cdf(self, data, name="CDF Plot", figsize=(15, 10), xaxis="values"):
        cdf1 = np.arange(1, len(data) + 1) / len(data)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(np.sort(data), cdf1, color="#EE6677", linewidth=2, label="Permanent (all measures)")

        ax.set_title(name, fontsize=20)
        ax.set_xlabel(xaxis, fontsize=20)
        ax.set_ylabel("Cumulative Probability", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.grid(True)
        plt.savefig(f'results/cdf_{name}.png')

    def plot_vessel_frequency(self, data):
        vessel_counts = data

        vessel_names = ["Small OCV", "Large OCV", "Barge"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey='row')
        axes = axes.flatten()

        for i, vessel_name in enumerate(vessel_names):
            vessel_type_counts = Counter(vessel_counts[:, i])
            sorted_keys = sorted(vessel_type_counts.keys())
            sorted_percentages = [vessel_type_counts[key] / self.runs * 100 for key in sorted_keys]

            axes[i].bar(sorted_keys, sorted_percentages)
            axes[i].set_title(f'{vessel_name} Frequency (%)')
            axes[i].set_xlabel('Number of Vessels')
            axes[i].set_ylabel('Frequency (%)')
            axes[i].set_xticks(sorted_keys)

        combinations = Counter([tuple(row) for row in vessel_counts])
        sorted_combinations = sorted(combinations.items(), key=lambda x: x[1], reverse=True)

        sorted_combination_percentages = [count / self.runs * 100 for _, count in sorted_combinations]
        labels = [f'{combo[0]}, {combo[1]}, {combo[2]}' for combo, _ in sorted_combinations]

        axes[3].bar(range(len(sorted_combinations)), sorted_combination_percentages)
        axes[3].set_title('Combination Frequency (%)')
        axes[3].set_xlabel('Combinations')
        axes[3].set_ylabel('Frequency (%)')
        axes[3].set_xticks(range(len(sorted_combinations)))
        axes[3].set_xticklabels(labels, rotation=90)

        plt.tight_layout()
        plt.savefig('results/Vessel_frequency_2x2.png')

    def plot_convergence(self, mean_duration, variance_duration, std_duration):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.plot(mean_duration, label='Mean Project Duration')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Duration')
        plt.title('Convergence of Mean Duration')

        plt.subplot(1, 3, 2)
        plt.plot(variance_duration, label='Variance of Project Duration')
        plt.xlabel('Iteration')
        plt.ylabel('Variance')
        plt.title('Convergence of Variance in Duration')

        plt.subplot(1, 3, 3)
        plt.plot(std_duration, label='Standard deviation of Project Duration')
        plt.xlabel('Iteration')
        plt.ylabel('Standard deviation')
        plt.title('Convergence of Standard deviation of duration')

        plt.tight_layout()
        plt.savefig('results/convergence.png')