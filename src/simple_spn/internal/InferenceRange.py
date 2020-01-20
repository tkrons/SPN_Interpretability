'''
Created on 18.06.2019

@author: Moritz
'''


import numpy as np
from scipy import stats





def categorical_likelihood_range(node, data, dtype=np.float64, **kwargs):
    """
    Returns the probability for the given ranges.
    
    ranges is multi-dimensional array:
    - First index specifies the instance
    - Second index specifies the feature
    
    Each entry of range contains a Range-object or None (e.g. for categorical-node NominalRange exists).
    If the entry is None, then 1 will be returned.
    """

    # Assert that the given node is only build on one instance
    assert len(node.scope) == 1, node.scope

    # Initialize the return variable log_probs with zeros
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    # Only select the ranges for the specific feature
    ranges = data[:, node.scope[0]]

    # For each instance
    for i, rang in enumerate(ranges):

        # Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            continue

        # Skip if no values for the range are provided
        if rang.is_impossible():
            probs[i] = 0

        # Compute the sum of the probability of all possible values
        probs[i] = sum([node.p[possible_val] for possible_val in rang.get_ranges()])

    return probs



def gaussian_likelihood_range(node, data, dtype=np.float64, **kwargs):
    """
    Returns the probability for the given ranges.
    
    ranges is multi-dimensional array:
    - First index specifies the instance
    - Second index specifies the feature
    
    Each entry of range contains a Range-object or None (e.g. for categorical-node NominalRange exists).
    If the entry is None, then 1 will be returned.
    """

    # Assert that the given node is only build on one instance
    assert len(node.scope) == 1, node.scope

    # Initialize the return variable log_probs with zeros
    densities = np.zeros((data.shape[0], 1), dtype=dtype)

    # Only select the ranges for the specific feature
    data = data[:, node.scope[0]]

    # For each instance
    for i, rang in enumerate(data):

        # Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            densities[i] = 1.
            continue

        # Skip if no values for the range are provided
        if rang.is_impossible():
            densities[i] = 0.
            
        # Assert to not mix probabilities and densities
        if len(rang.ranges) > 1:
            assert len(set([len(vals) for vals in rang.ranges])) == 1
        
        for interval in rang.ranges:
            if len(interval) == 1:
                x_val = interval[0]
                densities[i] += stats.norm.pdf(x_val, node.mean, node.stdev)
            elif len(interval) == 2:
                x_val1 = interval[0]
                x_val2 = interval[1]
                p1 = stats.norm.cdf(x_val1, node.mean, node.stdev)
                p2 = stats.norm.cdf(x_val2, node.mean, node.stdev)
                densities[i] += p2 - p1
    
    return np.array(densities, dtype=dtype)



def piecewise_likelihood_range(node, data, dtype=np.float64, **kwargs):
    """
    Returns the probability for the given ranges.
    
    ranges is multi-dimensional array:
    - First index specifies the instance
    - Second index specifies the feature
    
    Each entry of range contains a Range-object or None (e.g. for piecewise-node NumericRange exists).
    If the entry is None, then 1 will be returned.
    """

    # Assert context is not None and assert that the given node is only build on one instance
    assert len(node.scope) == 1, node.scope

    # Initialize the return variable log_probs with zeros
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    # Only select the ranges for the specific feature
    ranges = data[:, node.scope[0]]

    for i, rang in enumerate(ranges):

        # Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            continue

        # Skip if no values for the range are provided
        if rang.is_impossible():
            probs[i] = 0

        # Compute the sum of the probability of all possible values
        probs[i] = sum([__compute_probability_for_range(node, interval) for interval in rang.get_ranges()])

    return probs



def identity_likelihood_range(node, data, dtype=np.float64, **kwargs):
    """
    Returns the probability for the given ranges.
    
    ranges is multi-dimensional array:
    - First index specifies the instance
    - Second index specifies the feature
    
    Each entry of range contains a Range-object or None (e.g. for identity NumericRange exists).
    If the entry is None, then 1 will be returned.
    """
    assert len(node.scope) == 1, node.scope

    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    ranges = data[:, node.scope[0]]

    for i, rang in enumerate(ranges):

        # Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            probs[i] = 1
            continue

        # Skip if no values for the range are provided
        if rang.is_impossible():
            continue

        for interval in rang.get_ranges():

            if len(interval) == 1:
                lower = np.searchsorted(node.vals, interval[0], side="left")
                higher = np.searchsorted(node.vals, interval[0], side="right")
            else:
                lower = np.searchsorted(node.vals, interval[0], side="left")
                higher = np.searchsorted(node.vals, interval[1], side="right")

            probs[i] += (higher - lower) / len(node.vals)

    return probs




def __compute_probability_for_range(node, interval):
    '''
    Computes the probability for PWL gievn an interval
    '''
    
    if len(interval) == 1:
        #Returns a density
        return np.interp(x=interval[0], xp=node.x_range, fp=node.y_range)

    lower = interval[0]
    higher = interval[1]

    x_range = np.array(node.x_range)
    y_range = np.array(node.y_range)

    if lower <= x_range[0] and higher >= x_range[-1]:
        return 1.0

    lower_prob = np.interp(lower, xp=x_range, fp=y_range)
    higher_prob = np.interp(higher, xp=x_range, fp=y_range)

    indicies = np.where((lower < x_range) & (x_range < higher))

    x_tmp = [lower] + list(x_range[indicies]) + [higher]
    y_tmp = [lower_prob] + list(y_range[indicies]) + [higher_prob]
    
    #Returns a probability
    return np.trapz(y_tmp, x_tmp)


