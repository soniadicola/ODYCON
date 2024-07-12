# plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import Counter
from itertools import combinations

def plot_frequency(
        elements,
        n_runs,
        data1,
        data2=None,
        data1_label=None,
        data2_label=None,
        name="Frequencies",
        figsize=(20, 10),
):
    # Check the type of the first element in data1
    if isinstance(data1[0], list):
        # Flatten the list of data1
        elements1 = [num for comb in data1 for num in comb if num != 0.0]
    else:
        elements1 = [num for num in data1 if num != 0.0]

    # Count the frequency of each element in data1
    frequencies1 = Counter(elements1)

    available_elements = set(elements)

    # Add missing elements with a frequency of 0 for data1
    for measure in available_elements:
        if measure not in frequencies1:
            frequencies1[measure] = 0

 
    data1 = [(element, value / n_runs * 100) for element, value in frequencies1.items()]

    data1.sort(key=lambda x: x[0])

    # Separate labels and values for data1
    labels = [str(element) for element, _ in data1]
    values1 = [value for _, value in data1]

    # Create the bar graph
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.35
    x = np.arange(len(labels))
    ax.bar(x - width/2, values1, width, label=data1_label, color='grey', edgecolor='black')

    if data2 is not None:
        if isinstance(data2[0], list):
            # Flatten the list of data2
            elements2 = [num for comb in data2 for num in comb if num != 0.0]
        else:
            # Use data2 directly
            elements2 = [num for num in data2 if num != 0.0]

        # Count the frequency of each element in data2
        frequencies2 = Counter(elements2)

        # Add missing elements with a frequency of 0 for data2
        for measure in available_elements:
            if measure not in frequencies2:
                frequencies2[measure] = 0

        # Prepare data for the bar graph for data2
        data2 = [(element, value / n_runs * 100) for element, value in frequencies2.items()]
        data2.sort(key=lambda y: y[0])
        values2 = [value for _, value in data2]
        ax.bar(x + width/2, values2, width, label=data2_label, color='lightgrey', edgecolor='black')

    ax.yaxis.grid(True)
    ax.set_title(f"Criticality index of {name}", fontsize=20)
    ax.set_xlabel(name, fontsize=16)
    ax.set_ylabel("Percentage of occurrence (%)", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=20)
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'results/frequency_{name}.png')
    

def plot_frequency_combinations(
    n_runs,
    data,
    name='Frequency',
    top_n=10,
    figsize=(10, 8)
):

    # Count occurrences of each combination
    combination_counts = Counter()
    for sublist in data:
        measures = [measure for measure in sublist if measure != 0]
        for combo in combinations(measures, 2):
            combination_counts[combo] += 1

    # Select the top N combinations
    top_combinations = combination_counts.most_common(top_n)

    # Prepare data for plotting
    labels = [f"{combo[0]} & {combo[1]}" for combo, _ in top_combinations]
    values = [count for _, count in top_combinations]
    
    # Convert counts to percentages
    percentages = [(count / n_runs) * 100 for count in values]
    
    # Create the bar graph
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, percentages, color='grey', edgecolor='black')
    ax.yaxis.grid(True)
    ax.set_title(f"Criticality index of {name}", fontsize=20)
    ax.set_xlabel("Measure Combinations", fontsize=15)
    ax.set_ylabel("Frequency (%)", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'results/frequency_combinations_{name}.png')


def plot_cdf(
        opt_list,
        perm_list,
        original_list=None,
        plot_name="CDF Plot",
        figsize=(15, 10),
        xaxis="values",
        vline=None
):
    # Calculate the CDF for each list
    cdf1 = np.arange(1, len(opt_list) + 1) / len(opt_list)
    cdf2 = np.arange(1, len(perm_list) + 1) / len(perm_list)

    # Create the figure and add subplots for each CDF
    fig, ax = plt.subplots(figsize=figsize)

    # Plot original planning line
    if vline is not None:
        ax.axvline(x=vline, color="k", linewidth=3, label="Original Most-Likely duration")
    
    # Plot CDFs
    ax.plot(np.sort(perm_list), cdf2, color="#EE6677", linewidth=2, label="Permanent (all measures)")

    if original_list is not None:
        cdf3 = np.cumsum(np.sort(original_list)) / np.sum(original_list)
        ax.plot(np.sort(original_list), cdf3, color="#4477AA", linewidth=2, label="Original (no mitigation)")

    ax.plot(np.sort(opt_list), cdf1, color="#228833", linewidth=2, label="Tentative (optimized measures)")

    ax.set_title(plot_name, fontsize=20)
    ax.set_xlabel(xaxis, fontsize=20)
    ax.set_ylabel("Cumulative Probability", fontsize=20)
    ax.legend(loc="lower right", prop={'size': 20})
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True)
    plt.savefig(f'results/cdf_{plot_name}.png')


def plot_pdf(opt_list, perm_list, original_list=None, plot_name='PDF Plot', figsize=(15, 10), xaxis='Cost [$]'):

    mu1, std1 = norm.fit(opt_list)
    x1 = np.linspace(min(opt_list), max(opt_list), 100)
    pdf1 = norm.pdf(x1, mu1, std1)

    mu2, std2 = norm.fit(perm_list)
    x2 = np.linspace(min(perm_list), max(perm_list), 100)
    pdf2 = norm.pdf(x2, mu2, std2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x1, pdf1, color="#EE6677", linewidth=2)
    ax.hist(opt_list, bins=50, density=True, color="#EE6677", alpha=0.4, label="Permanent (all measures)")

    if original_list is not None:
        mu3, std3 = norm.fit(original_list)
        x3 = np.linspace(min(original_list), max(original_list), 100)
        pdf3 = norm.pdf(x3, mu3, std3)
        ax.plot(x3, pdf3, color="#4477AA", linewidth=2)
        ax.hist(original_list, bins=50, density=True, alpha=0.4,  color="#4477AA", label="Original")
    
    ax.plot(x2, pdf2, color="#228833", linewidth=2)
    ax.hist(perm_list, bins=50, density=True, alpha=0.4,  color="#228833", label="Tentative (optimized measures)")

    ax.set_title(plot_name, fontsize=20)
    ax.set_xlabel(xaxis, fontsize=20)
    ax.set_ylabel('Probability Distribution Function', fontsize=20)
    ax.legend(loc='upper left', prop={'size': 20})
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    plt.savefig(f'results/pdf_{plot_name}.png')

def plot_cumulative_histograms(data, intervals, plot_name='PDF Evolution Plot', figsize=(15, 10), xaxis='Cost [$]'):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, len(intervals)))  # Generate a color map for different intervals

    # Iterate over intervals
    for i, interval in enumerate(intervals):
        subset = data[:interval]
        mu, std = norm.fit(subset)
        x = np.linspace(min(subset), max(subset), 100)
        pdf = norm.pdf(x, mu, std)

        # Plot PDF
        ax.plot(x, pdf, color=colors[i], linewidth=2, label=f'Interval {interval}')
        # Plot histogram for the same interval
        ax.hist(subset, bins=50, density=True, color=colors[i], alpha=0.4)

    ax.set_title(plot_name, fontsize=20)
    ax.set_xlabel(xaxis, fontsize=20)
    ax.set_ylabel('Probability Distribution Function', fontsize=20)
    ax.legend(loc='upper left', prop={'size': 15})
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True)
    plt.savefig(f'results/hist_convergence_{plot_name}.png')
