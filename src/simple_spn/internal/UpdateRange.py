'''
Created on 18.06.2019

@author: Moritz
'''

import numpy as np

def categorical_update_range(node, rang, **kwargs):
    """
    Updates the probability distribution of a Categorical leave
    """

    # Assert that the given node is only build on one instance
    assert len(node.scope) == 1, node.scope

    if rang is None or rang.is_impossible():
        return 0
    
    new_p = np.zeros(len(node.p))
    for possible_val in rang.get_ranges():
        new_p[possible_val] = node.p[possible_val]
    
    new_p = new_p / np.sum(new_p)
    node.p = new_p
