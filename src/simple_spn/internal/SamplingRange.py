"""
Created on May 23, 2018

@author: Moritz
"""

import numpy as np


from spn.structure.Base import Product, Sum, Leaf, get_nodes_by_type
from spn.algorithms.Inference import _node_likelihood
import logging

logger = logging.getLogger(__name__)


def _validate_ids(node):
    all_nodes = get_nodes_by_type(node)

    ids = set()
    for n in all_nodes:
        ids.add(n.id)

    assert len(ids) == len(all_nodes), "not all nodes have ID's"

    assert min(ids) == 0 and max(ids) == len(ids) - 1, "ID's are not in order"


def _reset_node_counters(node):
    all_nodes = get_nodes_by_type(node)
    max_id = 0
    for n in all_nodes:
        # reset sum node counts
        if isinstance(n, Sum):
            n.edge_counts = np.zeros(len(n.children), dtype=np.int64)
        # sets nr_nodes to the max id
        max_id = max(max_id, n.id)
        n.row_ids = []
    return max_id


def _set_weights_for_evidence(node, ranges, dtype=np.float64, node_likelihood=_node_likelihood):

    if isinstance(node, Product):

        prob = 1.0
        for c in node.children:
            prob *= _set_weights_for_evidence(c, ranges, node_likelihood=node_likelihood)

        return prob

    elif isinstance(node, Sum):
        evidence_weights = np.zeros(len(node.children))

        for i, c in enumerate(node.children):
            prob = _set_weights_for_evidence(c, ranges, node_likelihood=node_likelihood)
            evidence_weights[i] = prob * node.weights[i]

        if np.sum(evidence_weights) == 0:
            return 0

        node.evidence_weights = evidence_weights / np.sum(evidence_weights)
        return np.sum(evidence_weights)

    elif isinstance(node, Leaf):
        t_node = type(node)
        if t_node in node_likelihood:
            ranges = np.array([ranges])
            return node_likelihood[t_node](node, ranges, dtype=dtype, node_likelihood=node_likelihood)
        else:
            raise Exception("No log-likelihood method specified for node type: " + str(type(node)))


def sample_instances(node, D, n_samples, rand_gen, ranges=None, dtype=np.float64, node_sample=None, node_likelihood=_node_likelihood):

    instance_ids = np.arange(n_samples)
    X = np.zeros((n_samples, D), dtype=dtype)

    _max_id = _reset_node_counters(node)
    result = _set_weights_for_evidence(node, ranges, node_likelihood=node_likelihood)

    if result == 0:
        return np.zeros((0, D), dtype=dtype)

    def _sample_instances(node, row_ids):
        if len(row_ids) == 0:
            return
        node.row_ids = row_ids

        if isinstance(node, Product):
            for c in node.children:
                _sample_instances(c, row_ids)
            return

        if isinstance(node, Sum):

            rand_child_branches = rand_gen.choice(
                np.arange(len(node.evidence_weights)), p=node.evidence_weights, size=len(row_ids)
            )

            for i, c in enumerate(node.children):
                new_row_ids = row_ids[rand_child_branches == i]
                node.edge_counts[i] = len(new_row_ids)
                _sample_instances(c, new_row_ids)

        if isinstance(node, Leaf):

            t_node = type(node)
            if t_node in node_sample:
                X[row_ids, node.scope] = node_sample[t_node](node, len(row_ids), rand_gen, ranges)
            else:
                raise Exception("No sample method specified for node type: " + str(type(node)))

            return

    _sample_instances(node, instance_ids)

    return X



'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''

from scipy.stats import norm
from spn.experiments.AQP.Ranges import NumericRange
from spn.structure.leaves.parametric.Parametric import Gaussian


def sample_gaussian_node(node, n_samples, rand_gen, ranges=None):
    assert isinstance(node, Gaussian)
    assert n_samples > 0


    if ranges is None or ranges[node.scope[0]] is None:
        return rand_gen.normal(node.mean, node.stdev, n_samples)
    else:
        # Generate bins for the specified range
        rang = ranges[node.scope[0]]
        assert isinstance(rang, NumericRange)

        # Iterate over the specified ranges
        probs = []
        prob_ranges = []
        intervals = rang.get_ranges()
        for interval in intervals:
            low_cdf = norm.cdf(interval[0], loc=node.mean, scale=node.stdev)
            high_cdf = norm.cdf(interval[1], loc=node.mean, scale=node.stdev)
            prob_ranges.append([low_cdf, high_cdf])
            probs.append(high_cdf-low_cdf)
        
        probs = probs/np.sum(probs)
        samples_from_parts = rand_gen.choice(np.arange(len(probs)), p=probs, size=n_samples)
        unique, counts = np.unique(samples_from_parts, return_counts=True)
        
        p_samples = []
        for i in range(len(unique)):
            prob_rang = prob_ranges[unique[i]]
            n_vals = counts[i]
            p_samples += list((prob_rang[1] - prob_rang[0]) * rand_gen.random_sample(size=n_vals) + prob_rang[0])
            
        return norm.ppf(p_samples, loc=node.mean, scale=node.stdev)

            


if __name__ == '__main__':
    
    g = Gaussian(mean=10, stdev=2, scope=[0])
    samples = sample_gaussian_node(g, 5, np.random.RandomState(1), ranges=None)
    print(samples)
    
    ranges = np.array([NumericRange([[0,10]])])
    samples = sample_gaussian_node(g, 100, np.random.RandomState(1), ranges=ranges)
    print(samples)


















