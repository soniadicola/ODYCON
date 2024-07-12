# preference_functions.py

import numpy as np
from scipy.stats import beta, norm

class PreferenceCurve:
    """
    A class used to generate different types of preference curves based on the specified parameters.

    Attributes
    ----------
    lower_bound : float
        The lower bound of the x values.
    mode : float
        The mode (peak) of the preference curve.
    upper_bound : float
        The upper bound of the x values.
    x_values : np.ndarray
        The generated x values for the curve.
    y_values : np.ndarray
        The calculated y values for the curve.

    Methods
    -------
    linear():
        Generates a linear preference curve.
    beta_PERT():
        Generates a Beta PERT preference curve.
    parabolic():
        Generates a parabolic preference curve.
    normal_distribution():
        Generates a normal distribution preference curve.
    """

    def __init__(self, lower_bound, mode, upper_bound):
        """
        Constructs the necessary attributes for the PreferenceCurve object.

        Parameters
        ----------
        lower_bound : float
            The lower bound of the x values.
        mode : float
            The mode (peak) of the preference curve.
        upper_bound : float
            The upper bound of the x values.
        """
        self.lower_bound = lower_bound
        self.mode = mode
        self.upper_bound = upper_bound
        self.x_values = np.array([])
        self.y_values = np.array([])

    def linear(self):
        """
        Generates a linear preference curve based on the mode value.
        """
        x = np.linspace(self.lower_bound, self.upper_bound, 1000)
        self.x_values = x
        if self.mode == self.lower_bound:
            m = -100 / np.max(x)
            b = 100
            y = m * x + b
        elif self.mode == self.upper_bound:
            m = 100 / np.max(x)
            y = m * x
        else:
            y = np.where(
                x <= self.mode,
                100 * (x - self.lower_bound) / (self.mode - self.lower_bound),
                100 * (self.upper_bound - x) / (self.upper_bound - self.mode),
            )
        self.y_values = y

    def beta_PERT(self):
        """
        Generates a Beta PERT preference curve.
        """
        x = np.linspace(self.mode - self.lower_bound, self.mode + self.upper_bound, 1000)
        self.x_values = x
        peak = self.mode
        min_val = self.mode - self.lower_bound
        max_val = self.mode + self.upper_bound

        alpha = 1 + 4 * (peak - min_val) / (max_val - min_val)
        beta_b = 1 + 4 * (max_val - peak) / (max_val - min_val)

        y = beta.pdf((x - min_val) / (max_val - min_val), alpha, beta_b)
        self.y_values = (y * (100 / np.max(y))).astype(int)

    def parabolic(self):
        """
        Generates a parabolic preference curve based on the mode value.
        """
        x = np.linspace(self.lower_bound, self.upper_bound, 1000)
        self.x_values = x

        if self.lower_bound == self.mode:
            self.y_values = (
                -100 * ((x - self.mode) ** 2) / (self.upper_bound - self.mode) ** 2
                + 100
            )
            self.y_values[x < self.mode] = 0
        elif self.upper_bound == self.mode:
            self.y_values = (
                -100 * ((x - self.mode) ** 2) / (self.mode - self.lower_bound) ** 2
                + 100
            )
            self.y_values[x > self.mode] = 0
        else:
            self.y_values = (
                -100 * ((x - self.mode) ** 2) / (self.mode - self.lower_bound) ** 2
                + 100
            )

    def normal_distribution(self):
        """
        Generates a normal distribution preference curve based on the mode value.
        """
        x = np.linspace(self.lower_bound, self.upper_bound, 1000)
        self.x_values = x
        std_dev = (self.upper_bound - self.mode) / 3
        y = norm.pdf(x, self.mode, std_dev)
        self.y_values = y / np.max(y) * 100
