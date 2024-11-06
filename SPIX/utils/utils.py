import numpy as np
def min_max(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))