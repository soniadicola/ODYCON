# network.py

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

class Network:
    def __init__(self, activities, duration_column):
        """
        Initialize the ActivityNetwork with activities and the specified duration column.

        Args:
            activities (pandas.DataFrame): DataFrame containing activity information.
            duration_column (str): Name of the column specifying the duration of activities.
        """
        self.activities = activities
        self.duration_column = duration_column
        self.G = self.compile_network()

    def compile_network(self):
        """
        Create a network graph representing activities and their dependencies.

        Returns:
            nx.DiGraph: Directed graph representing activities and dependencies.
        """

        # Create a graph
        G = nx.DiGraph()

        # Add nodes for each activity
        for _, row in self.activities.iterrows():
            activity_id = row["act_ID"]
            description = row["act_description"]
            duration = row[self.duration_column]
            G.add_node(activity_id, description=description, duration=duration)

        # Add edges representing relation between activities
        for _, row in self.activities.iterrows():
            activity_id = row["act_ID"]
            predecessors = row["act_dependency"]
            duration = row[self.duration_column]
            for predecessor in predecessors:
                G.add_edge(predecessor, activity_id, duration=duration)

        return G

    def get_paths(self):
        """
        Generate all paths in a directed acyclic graph (DAG) starting from the first
        activity and ending with an activity that has no successor.

        Returns:
            pandas.DataFrame: DataFrame contains path ID, path itself,
            and duration of all paths in network.
        """

        # Initiate paths and nodes
        roots = (v for v, d in self.G.in_degree() if d == 0)
        leaves = (v for v, d in self.G.out_degree() if d == 0)
        all_paths = []
        path_id = 1

        for root in roots:
            for leaf in leaves:
                paths = nx.all_simple_paths(self.G, root, leaf)
                for path in paths:
                    # Calculate duration of the path
                    duration = sum(self.G.nodes[node]["duration"] for node in path)
                    # Add the path ID, path, and its duration as a tuple to all_paths
                    all_paths.append((path_id, path, duration))
                    path_id += 1

        # Convert all_paths to a DataFrame
        paths_df = pd.DataFrame(all_paths, columns=['path_id', 'path', 'duration'])

        return paths_df

    def get_critical_path(self, paths_df):
        """
        Identify the critical path in a directed acyclic graph (DAG) based on a DataFrame of all paths and their durations.

        Args:
            paths_df (pandas.DataFrame): A DataFrame where each row contains a path ID, the path itself, and its duration.

        Returns:
            dict: A dictionary containing information about the critical path, including the list of activity IDs,
            descriptions, and the total duration.
        """

        # Sort paths_df based on total duration
        paths_df = paths_df.sort_values(by='duration', ascending=False)

        # Get the longest path (critical path)
        longest_path_id, longest_path, longest_path_duration = paths_df.iloc[0]

        # Extract activity names from the longest path
        activity_names = [self.G.nodes[node]["description"] for node in longest_path]

        # Create and return the result dictionary
        critical_path = {
            "critical_path_ID": longest_path_id,
            "critical_ID_list": longest_path,
            "critical_description_list": activity_names,
            "critical_path_duration": longest_path_duration
        }

        return critical_path
    
    def plot_network(self, project_start_date='2022-01-01'):
        """
        Plot the network graph and generate a Gantt chart.

        Args:
            project_start_date (str): Start date of the project in "YYYY-MM-DD" format.
        """
        project_start_date = pd.to_datetime(project_start_date)

        # Calculate start and end times
        start_times = {node: 0 for node in self.G.nodes}
        for node in nx.topological_sort(self.G):
            for predecessor in self.G.predecessors(node):
                start_times[node] = max(start_times[node], start_times[predecessor] + self.G.nodes[predecessor]['duration'])
        finish_times = {node: start_times[node] + self.G.nodes[node]['duration'] for node in self.G.nodes}

        # Create a dataframe for the Gantt chart
        gantt_df = pd.DataFrame(
            [
                dict(
                    Task=self.G.nodes[node]['description'],
                    Start=project_start_date + pd.DateOffset(days=start_times[node]),
                    Finish=project_start_date + pd.DateOffset(days=finish_times[node]),
                    Critical=node in self.get_critical_path(self.get_paths())['critical_ID_list'],
                )
            for node in self.G.nodes
            ]
        )

        # Create the Gantt chart
        fig = px.timeline(
            gantt_df,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Critical", 
            color_discrete_map={True: 'red', False: 'darkgrey'},
            category_orders={"Task": gantt_df['Task'].tolist()}
        )

        # Create a scatter plot for tasks with zero duration
        zero_duration_tasks = gantt_df[gantt_df['Start'] == gantt_df['Finish']] 
        fig.add_trace(go.Scatter(
            x=zero_duration_tasks['Start'], 
            y=zero_duration_tasks['Task'], 
            mode='markers',
            marker=dict(color=zero_duration_tasks['Critical'].map({True: 'red', False: 'darkgrey'})),
            showlegend=False
            )
        )

        fig.update_layout(
            height=800,
            width=1000,
            yaxis=dict(tickfont=dict(size=10))
        )

        fig.write_image(f'results/gantt_chart.png')
    

def critical_from_path(paths_df):
    """
    Identify the critical path in a directed acyclic graph (DAG) based on a DataFrame of all paths and their durations.
    
    Args:
        paths_df (DataFrame): A DataFrame where each row contains a path ID, the path itself, and its duration.
    
    Returns:
        dict: A dictionary containing information about the critical path, including the list of activity IDs,
        and the total duration.
    """

    # Sort paths_df based on total duration
    paths_df = paths_df.sort_values(by='duration', ascending=False)

    # Get the longest path (critical path)
    longest_path_id, longest_path, longest_path_duration = paths_df.iloc[0]

    # Create and return the result dictionary
    critical_path = {
        "critical_path_ID": longest_path_id,
        "critical_ID_list": longest_path,
        "critical_path_duration": longest_path_duration
    }
    
    return critical_path

class ProbabilisticNetwork:
    def __init__(self, activities, mitigation, risk, correlation):
        self.activities = activities
        self.mitigation = mitigation
        self.risk = risk
        self.correlation = correlation

    def rand_pert(self, property, n_samples):
        """
        Generate random numbers based on PERT distribution for pandas Series.
        In statistics, the PERT distributions are a family of continuous probability distributions
        defined by the minimum, most likely, and maximum values, mean and standard deviation that a variable can take.

        Parameters:
        property (pandas.Series): Individual series containing the [min, max, mean, sd] or [min, most likely, max] 
        n_samples (int): Number of random samples to generate.

        Returns:
        pandas.Series: New Series with random numbers sampled from the PERT distribution.
        """
        try:
            property = np.array(property, dtype=float)
        except ValueError as e:
            raise ValueError("Input property contains non-numeric values, which cannot be processed.") from e

        if len(property) ==3:
            a, m, b = property

            mean = ((a + 4*m + b)/6)
            sd = ((b-a)/6)

        elif len(property) == 4:
            a, b, mean, sd = property

            mean = np.where((mean >= b) | (mean <= a), (a+b)/2, mean)

            sd = np.where((sd >= mean-a) | (sd >= b-mean), np.minimum((mean-a)/2, (b-mean)/2), sd)  

        # Shift distribution (since beta distribution is defined for 0 < a,b)
        min_val = min(a.min(), b.min(), mean.min())
        if min_val < 0:
            a_adj = a - min_val
            b_adj = b - min_val
            mean_adj = mean - min_val
        else:
            a_adj = a
            b_adj = b
            mean_adj = mean
        
        a_adj = np.tile(a_adj, (n_samples, 1)).T
        b_adj = np.tile(b_adj, (n_samples, 1)).T
        mean_adj = np.tile(mean_adj, (n_samples, 1)).T
        sd = np.tile(sd, (n_samples, 1)).T

        alpha = np.full((len(a_adj), n_samples), np.nan)
        beta = np.full((len(b_adj), n_samples), np.nan)
        sample = np.full((len(b_adj), n_samples), np.nan)

        valid = (b_adj - a_adj != 0) & (sd > 0)

        alpha[valid] = (
            ((mean_adj[valid] - a_adj[valid]) / (b_adj[valid] - a_adj[valid])) *
            ((mean_adj[valid] - a_adj[valid]) * (b_adj[valid] - mean_adj[valid]) / sd[valid]**2 - 1)
        )

        valid_beta = valid & ~np.isnan(alpha)

        beta[valid_beta] = (
            alpha[valid_beta] * (b_adj[valid_beta] - mean_adj[valid_beta]) / (mean_adj[valid_beta] - a_adj[valid_beta])
        )

        # Calculate sample
        sample = (
            np.random.beta(alpha, beta, size=(len(alpha), n_samples)) 
            * (b_adj - a_adj) + a_adj
        )

        # Shift distribution back to original position
        if min_val < 0:
            # mask is true where sample has non-NaN values
            mask = ~np.isnan(sample)
            sample[mask] = sample[mask] + min_val
        
        # For cases where a = b, set sample to a
        mask = np.isnan(sample)
        a = np.tile(a, (n_samples, 1)).T
        sample[mask] = a[mask]
        
        return np.round(sample)

    def rand_draw(self, n_samples):
        """
        Creates set of random samples and calculates activity correlation correlation.

        Args:
            activities (pandas.DataFrame): DataFrame containing activity information.
            mitigation (pandas.DataFrame): DataFrame containing mitigation measure information.
            risk (pandas.DataFrame): DataFrame containing risk event information.
            correlation (pandas.DataFrame): DataFrame containing activity correlation information.
            n_samples (float): Number of random samples to be generated.

        Returns:
            shared_activity_duration (np.array): Random samples for shared activity duration.
            act_duration_no_corr (np.array): Random samples for non correlated activity duration.
            mit_capacity_actual (np.array): Random samples for mitigation capacity.
            risk_actual (np.array): Random draw of risk occurence and impact.
        """

        # Draw random duration for activities (total duration)
        act_duration_actual = self.rand_pert(
            [
                self.activities["act_duration_opt"],
                self.activities["act_duration_ml"],
                self.activities["act_duration_pes"]
            ],
            n_samples=n_samples
        )

        # Check for shared activity correlation
        if self.correlation.empty == True:
            print('Data does NOT include shared correlation.')
        else:
            print('Data includes shared activity correlation.')

            # Draw random number for shared uncertainty factor
            suf_duration_act = self.rand_pert(
                [
                    self.correlation["suf_duration_opt"],
                    self.correlation["suf_duration_ml"],
                    self.correlation["suf_duration_pes"],        
                ],
                n_samples=n_samples
            )

        suf_aggregated_draws = {}

        # Iterate through each row in the correlation DataFrame
        for index, row in self.correlation.iterrows():
            # Get the IDs of the activities affected by this uncertainty factor
            act_ids = [int(i) for i in row['suf_act_relations']]
            
            # Get the random draws for the current row
            current_draws = suf_duration_act[index]
            
            # Aggregate the draws for the affected activities
            for act_id in act_ids:
                if act_id not in suf_aggregated_draws:
                    suf_aggregated_draws[act_id] = np.zeros_like(current_draws)
                suf_aggregated_draws[act_id] += current_draws

        shared_activity_duration = np.zeros((len(self.activities), n_samples))

        for index, act_id in enumerate(self.activities['act_ID']):
            if act_id in suf_aggregated_draws:
                shared_activity_duration[index, :] = suf_aggregated_draws[act_id]


        # Initialize new columns in activities
        self.activities['act_duration_corr_min'] = 0
        self.activities['act_duration_corr_ml'] = 0
        self.activities['act_duration_corr_max'] = 0

        # Activities shared durations (only the shared part; the part of the duration that is causing correlation)
        for _, row in self.correlation.iterrows():
            # Get the IDs of the activities affected by the shared uncertainty factor
            act_ids = [int(i) for i in row['suf_act_relations']]

            for act_id in act_ids:
                # Add the shared uncertainty duration to the corresponding activity's correlated duration
                self.activities.loc[self.activities['act_ID'] == act_id, 
                            [
                                'act_duration_corr_min', 
                                'act_duration_corr_ml', 
                                'act_duration_corr_max'
                            ]] += row[[
                                'suf_duration_opt', 
                                'suf_duration_ml', 
                                'suf_duration_pes'
                            ]].values

        # Draw random duration for activities (correlation)
        act_duration_corr = self.rand_pert(
            [
                self.activities["act_duration_corr_min"],
                self.activities["act_duration_corr_ml"],
                self.activities["act_duration_corr_max"]
            ],
            n_samples=n_samples
        )

        # Activities duration without correlation with other activities (removing the shared durations)
        self.activities['act_duration_no_corr_min'] = self.activities["act_duration_opt"] - self.activities['act_duration_corr_min']
        self.activities['act_duration_no_corr_ml'] = self.activities["act_duration_ml"] - self.activities['act_duration_corr_ml']
        self.activities['act_duration_no_corr_max'] = self.activities["act_duration_pes"] - self.activities['act_duration_corr_max']

        # Calculate correlation statistics of the actual duration and correlated duration
        self.activities['tot_mean'] = np.mean(act_duration_actual, axis=1)
        self.activities['tot_var'] = np.var(act_duration_actual, axis=1)
        self.activities['corr_mean'] = np.mean(act_duration_corr, axis=1)
        self.activities['corr_var'] = np.var(act_duration_corr, axis=1)

        # Calculate no-correlation statistics
        self.activities['no_corr_mean'] = self.activities['tot_mean'] - self.activities['corr_mean']
        self.activities['no_corr_var'] = self.activities['tot_var'] - self.activities['corr_var']
        self.activities['no_corr_std'] = self.activities['no_corr_var'].abs() ** 0.5

        # Draw random duration for activities (no correlation)
        act_duration_no_corr = self.rand_pert(
            [
                self.activities['act_duration_no_corr_min'],
                self.activities["act_duration_no_corr_max"],
                self.activities['no_corr_mean'],
                self.activities['no_corr_std']
            ],
            n_samples=n_samples
        )

        # Draw random number for mitigation capacity (time)
        mit_capacity_actual = self.rand_pert(
            [
                self.mitigation["mit_capacity_opt"],
                self.mitigation["mit_capacity_ml"],
                self.mitigation["mit_capacity_pes"]
            ],
            n_samples=n_samples
        )

        # Map the risk to activities based on risk_act_relation
        mapped_df = pd.merge(self.activities, self.risk, how='left', left_on='act_ID', right_on='risk_act_relation')

        # Draw random number for risk event
        risk_duration_actual = self.rand_pert(
            [
                mapped_df["risk_duration_opt"],
                mapped_df["risk_duration_ml"],
                mapped_df["risk_duration_pes"]
            ],
            n_samples=n_samples
        )
        risk_duration_actual = np.nan_to_num(risk_duration_actual)
        
        # Draw if a risk occurs or not (bernoulli)
        probability = np.tile(mapped_df["risk_probability"], (n_samples, 1)).T
        probability = np.nan_to_num(probability)
        risk_occurrence = np.random.binomial(n=1, p=probability, size=(len(probability), n_samples)).astype(bool)
        risk_actual = np.where(risk_occurrence, risk_duration_actual, 0)

        return shared_activity_duration, act_duration_no_corr, mit_capacity_actual, risk_actual