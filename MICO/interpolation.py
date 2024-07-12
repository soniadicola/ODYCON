# interpolation.py

import numpy as np


def linear_interpolation(x, x1, x2, y1, y2):
    """
    The function computes the slope based on the difference between the y-values and the difference between the x-values.
    If the difference in x-values is zero, indicating division by zero, it sets the slope to zero to avoid NaN values.
    Then, it calculates the y-value using the calculated slope and the input x-value.
    
    Args:
        x (pandas.Series): The x-values for which we want to calculate the corresponding y-values.
        x1 (pandas.Series): The first x-values.
        x2 (pandas.Series): The second x-values.
        y1 (pandas.Series): The y-values corresponding to the first x-values.
        y2 (pandas.Series): The y-values corresponding to the second x-values.
    
    Returns:
        pandas.Series: The calculated y-values.
    """

    m = np.where(x1 != x2, (y2 - y1) / (x2 - x1), 0)
    y = y1 + m * (x - x1)
    
    return round(y, 1)