# import_project.py
import pandas as pd
import os


class ProjectDataImporter:
    def __init__(self, filename):
        """
        Initializes the ProjectDataImporter object and imports project data from the specified Excel file.

        Args:
            filename (str): The path to the Excel file.
        """
        self.filename = filename
        self.activities_df = None
        self.mitigation_df = None
        self.risk_df = None
        self.correlation_df = None
        self.import_project()

    @staticmethod
    def convert_dependencies(dependencies):
        """
        Convert dependencies to a list of integers. This function takes a value which can be NaN,
        an integer, a float, or a string of comma-separated numbers.
        It returns a list of integers representing the dependencies.

        Args:
            dependencies (nan, int, float, or str): The dependencies to be converted.

        Returns:
            list of int: The converted dependencies.

        Raises:
            ValueError: If dependencies is not of type nan, int, float, or str.
        """
        if pd.isna(dependencies):
            return []
        elif isinstance(dependencies, (int, float)):
            return [int(dependencies)]
        elif isinstance(dependencies, str):
            dependencies = dependencies.split(',')
            return [int(float(dep)) for dep in dependencies]
        else:
            raise ValueError(f"Unexpected type {type(dependencies)} in dependencies")

    def import_project(self):
        """
        Import project data from an Excel file and populate the object's dataframes.
        """
        # Check if file exists
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"{self.filename} does not exist")

        # Check file extension
        _, extension = os.path.splitext(self.filename)
        if extension != '.xlsx':
            raise ValueError(f"{self.filename} is an unsupported filetype")

        # Import project datafile if all checks are ok
        # Read Excel file
        data_act = pd.read_excel(self.filename, skiprows=2, sheet_name="Activities")
        data_mit = pd.read_excel(self.filename, skiprows=2, sheet_name="Mitigations")
        data_risk = pd.read_excel(self.filename, skiprows=2, sheet_name="Risks")
        data_corr = pd.read_excel(self.filename, skiprows=2, sheet_name="Correlation")

        # Create activities dataframe
        self.activities_df = pd.DataFrame(
            {
                "act_ID": data_act["Activity ID"],
                "act_description": data_act["Activity description"],
                "act_duration_opt": data_act["Optimistic"],
                "act_duration_ml": data_act["Most-Likely"],
                "act_duration_pes": data_act["Pessimistic"],
                "act_dependency": data_act["Predecessor activity"],
            }
        )

        # Convert 'act_dependency' to list of integers
        self.activities_df['act_dependency'] = self.activities_df['act_dependency'].apply(self.convert_dependencies)

        # Create mitigation dataframe
        self.mitigation_df = pd.DataFrame(
            {
                "mit_ID": data_mit["Mitigation ID"],
                "mit_description": data_mit["Mitigation measure"],
                "mit_capacity_opt": data_mit["Minimum time"],
                "mit_capacity_ml": data_mit["Most likely time"],
                "mit_capacity_pes": data_mit["Maximum time"],
                "mit_act_relation": data_mit["Relations"],
                "mit_dependency_factor": data_mit["dependency factor (eta)"],
                "mit_cost_min": data_mit["Minimum cost"],
                "mit_cost_ml": data_mit["Most likely cost"],
                "mit_cost_max": data_mit["Maximum cost"],
                "mit_nuisance_min": data_mit["Minimum nuisance"],
                "mit_nuisance_ml": data_mit["Most likely nuisance"],
                "mit_nuisance_max": data_mit["Maximum nuisance"],

            }
        )

        # Create risk dataframe
        self.risk_df = pd.DataFrame(
            {
                "risk_ID": data_risk["Risk event ID"],
                "risk_description": data_risk["Risk event description"],
                "risk_duration_opt": data_risk["Minimum"],
                "risk_duration_ml": data_risk["Most likely"],
                "risk_duration_pes": data_risk["Maximum"],
                "risk_act_relation": data_risk["Affected activities"],
                "risk_probability": data_risk["Risk probability"],
            }
        )

        # Create shared uncertainty factor dataframe
        self.correlation_df = pd.DataFrame(
            {
                "suf_ID": data_corr["SUF ID"],
                "suf_description": data_corr["SUF description"],
                "suf_duration_opt": data_corr["Minimum"],
                "suf_duration_ml": data_corr["Most likely"],
                "suf_duration_pes": data_corr["Maximum"],
                "suf_act_relations": data_corr["Relations"],
            }
        )

        # Convert 'suf_act_relations' to list of integers
        self.correlation_df['suf_act_relations'] = self.correlation_df['suf_act_relations'].apply(self.convert_dependencies)
