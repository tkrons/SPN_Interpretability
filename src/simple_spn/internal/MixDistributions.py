'''
Created on 27.06.2019

@author: Moritz
'''

import numpy as np
from spn.structure.leaves.parametric.Parametric import Categorical

def mix_categorical(weighted_nodes):
    assert sum([weight for (weight, node) in weighted_nodes]) == 1
    
    p = np.zeros(len(weighted_nodes[0][1].p))
    scope = weighted_nodes[0][1].scope
    
    for (weight, node) in weighted_nodes:
        assert isinstance(node, Categorical)
        for i in range(len(p)):
            p[i] += weight * node.p[i]
        
    return Categorical(p=p, scope=scope)
    
    
    