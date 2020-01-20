'''
Created on 08.07.2019

@author: Moritz
'''

import numpy as np

from spn.structure.Base import get_nodes_by_type, Leaf, Sum, Product
from spn.algorithms.Inference import likelihood, sum_likelihood, prod_likelihood

from spn.experiments.AQP.leaves.identity.Expectation import identity_expectation as ie


def Expectation(spn, feature_id, ranges, node_expectation, node_likelihood):
    
    def leaf_expectation(node, data, dtype=np.float64, **kwargs):
        if node.scope[0] == feature_id:
            t_node = type(node)
            if t_node in node_expectation:
                exps = np.zeros((data.shape[0], 1), dtype=dtype)
                exps[:] = node_expectation[t_node](node)
                return exps
            else:
                raise Exception("Node type unknown for expectation: " + str(t_node))
        else:
            t_node = type(node) 
            if t_node in node_likelihood:
                return node_likelihood[t_node](node, ranges, node_likelihood=node_likelihood)
    
    node_expectations = {type(leaf): leaf_expectation for leaf in get_nodes_by_type(spn, Leaf)}
    node_expectations.update({Sum: sum_likelihood, Product: prod_likelihood})
    
    expectation = likelihood(spn, ranges, node_likelihood=node_expectations)
    expectation = expectation / likelihood(spn, ranges, node_likelihood=node_likelihood)
    
    return expectation


def old_Expectation(spn, feature_scope, evidence_scope, evidence, node_expectation, node_likelihood):

    """Compute the Expectation:

        E[X_feature_scope | X_evidence_scope] given the spn and the evidence data

    Keyword arguments:
    spn -- the spn to compute the probabilities from
    feature_scope -- set() of integers, the scope of the features to get the expectation from
    evidence_scope -- set() of integers, the scope of the evidence features
    evidence -- numpy 2d array of the evidence data
    """

    if evidence_scope is None:
        evidence_scope = set()

    assert not (len(evidence_scope) > 0 and evidence is None)

    assert len(feature_scope.intersection(evidence_scope)) == 0
    
    #Marginalize beforehand to improve efficiency
    # marg_spn = marginalize(spn, keep=feature_scope | evidence_scope)

    def leaf_expectation(node, data, dtype=np.float64, **kwargs):
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:
                exps = np.zeros((data.shape[0], 1), dtype=dtype)
                exps[:] = node_expectation[t_node](node)
                return exps
            else:
                raise Exception("Node type unknown for expectation: " + str(t_node))
        
        if node.scope[0] in evidence_scope: 
            t_node = type(node) 
            if t_node in node_likelihood:
                return node_likelihood[t_node](node, evidence, node_likelihood=node_likelihood)
        else:
            return 1
        

    node_expectations = {type(leaf): leaf_expectation for leaf in get_nodes_by_type(spn, Leaf)}
    node_expectations.update({Sum: sum_likelihood, Product: prod_likelihood})

    if evidence is None:
        # fake_evidence is not used
        fake_evidence = np.zeros((1, len(spn.scope))).reshape(1, -1)
        expectation = likelihood(spn, fake_evidence, node_likelihood=node_expectations)
        return expectation

    # if we have evidence, we want to compute the conditional expectation
    expectation = likelihood(spn, evidence, node_likelihood=node_expectations)
    expectation = expectation / likelihood(spn, evidence, node_likelihood=node_likelihood)

    return expectation



def identity_expectation(node):
    return ie(node)


def piecewise_expectation(node):
    exp = 0
    for i in range(len(node.x_range) - 1):
        y0 = node.y_range[i]
        y1 = node.y_range[i + 1]
        x0 = node.x_range[i]
        x1 = node.x_range[i + 1]

        # compute the line of the top of the trapezoid
        m = (y0 - y1) / (x0 - x1)
        b = (x0 * y1 - x1 * y0) / (x0 - x1)

        # integral from w to z, of x * (mx+b) dx
        w = x0
        z = x1
        integral = (1 / 6) * (-3 * b * (w ** 2) + 3 * b * (z ** 2) - 2 * m * (w ** 3) + 2 * m * (z ** 3))
        exp += integral

    return exp


def gaussian_expectation(node):
    return node.mean


def categorical_expectation(node):
    return np.argmax(node.p)

