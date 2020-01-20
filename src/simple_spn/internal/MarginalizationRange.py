'''
Created on 11.10.2019

@author: Moritz
'''

import numpy as np
from copy import deepcopy
from spn.structure.Base import assign_ids
from spn.algorithms.Validity import is_valid
from spn.algorithms.TransformStructure import Prune
from spn.structure.Base import Sum, Product, Leaf
from spn.structure.leaves.parametric.Parametric import Categorical


def marg_rang(spn, rang, node_likelihood, keep=None):
    
    if keep is None:
        keep = set([i for i, val in enumerate(rang) if val is None]) 
    
    def spn_for_evidence_recursive(node):
        
        if isinstance(node, Leaf):
            
            if rang[node.scope[0]] is not None:
                t_node = type(node)
                if t_node in node_likelihood:
                    ranges = np.array([rang])
                    prob =  node_likelihood[t_node](node, ranges, node_likelihood=node_likelihood)[0][0]
                    return prob, None
                else:
                    raise Exception('No log-likelihood method specified for node type: ' + str(type(node)))
            else:
                if node.scope[0] in keep:
                    return 1., deepcopy(node)
                else:
                    return 1., None


        newNode = node.__class__()
        newNode.scope = sorted(list(keep.intersection(set(node.scope))))
        
        if isinstance(node, Sum):
            new_weights = []
            new_childs = []
        
            for i, c in enumerate(node.children):
                prob, new_child  = spn_for_evidence_recursive(c)
                new_prob = prob * node.weights[i]
                if new_child is not None and new_prob > 0:
                    new_weights.append(new_prob)
                    new_childs.append(new_child)
            
            new_weights = np.array(new_weights)
            newNode.weights = new_weights / np.sum(new_weights)
            newNode.children = new_childs
            return np.sum(new_weights), newNode
        
        
        elif isinstance(node, Product):
            new_childs = []
            
            new_prob = 1.
            for i, c in enumerate(node.children):
                prob, new_child = spn_for_evidence_recursive(c)
                new_prob *= prob
                if new_child is not None:
                    new_childs.append(new_child)
                
            newNode.children = new_childs
            return new_prob, newNode
    
    prob, newNode = spn_for_evidence_recursive(spn)
    assign_ids(newNode)
    #newSPN = Prune(newNode)    ####ERROR???????????
    newSPN = newNode
    valid, err = is_valid(newSPN)
    assert valid, err

    return prob, newSPN



'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''



from simple_spn.internal.UpdateRange import categorical_update_range
distribution_update_ranges = {Categorical     : categorical_update_range}
def marg_rang_special(spn, rang, node_likelihood, distribution_update_ranges=distribution_update_ranges):
    
    def spn_for_evidence_recursive(node):
        
        if isinstance(node, Leaf):
            if len(node.scope) > 1:
                raise Exception("Leaf Node with |scope| > 1")
            
            if rang[node.scope[0]] is not None:
                t_node = type(node)
                if t_node in node_likelihood:
                    ranges = np.array([rang])
                    prob =  node_likelihood[t_node](node, ranges, node_likelihood=node_likelihood)[0][0]
                    if prob == 0:
                        newNode = deepcopy(node)
                    else:
                        newNode = deepcopy(node)
                        distribution_update_ranges[t_node](newNode, rang[node.scope[0]])
                else:
                    raise Exception('No log-likelihood method specified for node type: ' + str(type(node)))
            else:
                prob = 1
                newNode = deepcopy(node)
                
            return prob, newNode
            

        newNode = node.__class__()
        newNode.scope = node.scope

        if isinstance(node, Sum):
            new_weights = []
            new_childs = []
        
            for i, c in enumerate(node.children):
                prob, new_child  = spn_for_evidence_recursive(c)
                new_prob = prob * node.weights[i]
                if new_prob > 0:
                    new_weights.append(new_prob)
                    new_childs.append(new_child)
            
            new_weights = np.array(new_weights)
            newNode.weights = new_weights / np.sum(new_weights)
            newNode.children = new_childs
            return np.sum(new_weights), newNode
        
        
        elif isinstance(node, Product):
            new_childs = []
            
            new_prob = 1.
            for i, c in enumerate(node.children):
                prob, new_child = spn_for_evidence_recursive(c)
                new_prob *= prob
                new_childs.append(new_child)
                
            newNode.children = new_childs
            return new_prob, newNode

    prob, newNode = spn_for_evidence_recursive(spn)
    assign_ids(newNode)
    newSPN = Prune(newNode)
    valid, err = is_valid(newSPN)
    assert valid, err

    return prob, newSPN