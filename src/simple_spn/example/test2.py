'''
Created on 17.06.2019

@author: Moritz
'''

import numpy as np

from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

from spn.algorithms.Inference import likelihood
from spn.structure.Base import get_nodes_by_type, Leaf

def generate_gender_age_data(num_instances, rand_seed):    
    '''
    Name:
    "gender-age"
    
    Features:
    1st column : gender : {male,female}
    2nd column : student: {yes,no}
    3rd column : age    : continuous (Gaussian distribution)
    
    Correlations:
    P(gender=male) = 50%
    P(gender=male) = 50%
    P(student=yes|gender=m) = 30%
    P(student=yes|gender=f) = 80%
    P(age) = N(mu=20, sigma=3)
    '''
    
    np.random.seed(rand_seed)

    data = [] 
    for _ in range(num_instances):
        inst = []
        if np.random.random() < 0.5:
            inst.append(0)
            if np.random.random() < 0.3:
                inst.append(1)
            else:
                inst.append(0)
            
        else:
            inst.append(1)
            if np.random.random() < 0.8:
                inst.append(1)
            else:
                inst.append(0)     
            
        inst.append(int(np.random.normal(20, 3)))
        data.append(inst)
    
    return np.array(data), ["discrete", "discrete", "continuous"]
    
    
    
def generate_spn_parameters(rdc_threshold=0.3, cols="rdc", rows="rdc", min_instances_slice=50, ohe=False, prior_weight=0.00, identity_numeric=False):
    
    return {"rdc_threshold" : rdc_threshold,
            "cols" : cols,
            "rows" : rows,
            "min_instances_slice" : min_instances_slice,
            "ohe" : ohe,
            "prior_weight" : prior_weight,
            "identity_numeric" : identity_numeric}
    
    

def build_spn(numpy_data, feature_types, spn_params, rand_gen):
    
    from spn.algorithms.StructureLearning import get_next_operation, learn_structure
    from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py

    from spn.structure.leaves.parametric.Parametric import Categorical
    from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
    from spn.experiments.AQP.leaves.identity.IdentityNumeric import create_identity_leaf


    #cast may not be necessary
    numpy_data = np.array(numpy_data, np.float64)
    
    #Generate meta_type array
    meta_types = []
    for feature_type in feature_types:
        if feature_type == "discrete":
            meta_types.append(MetaType.DISCRETE)
        elif feature_type == "continuous":
            meta_types.append(MetaType.REAL)
        else:
            raise Exception("Unknown feature type for SPN: " + feature_type)
    
    #Create information about the domains
    domains = []
    for col in range(numpy_data.shape[1]):
        feature_type = feature_types[col]
        if feature_type == 'continuous':
            domains.append([np.min(numpy_data[:, col]), np.max(numpy_data[:, col])])
        elif feature_type in {'discrete', 'categorical'}:
            domains.append(np.unique(numpy_data[:, col]))
    
    #Create context
    ds_context = Context(meta_types=meta_types, domains=domains)
        
    #Fixed parameters
    rdc_threshold = spn_params["rdc_threshold"]
    cols = spn_params["cols"]
    rows = spn_params["rows"]
    min_instances_slice = spn_params["min_instances_slice"]
    ohe = spn_params["ohe"]
    prior_weight = spn_params["prior_weight"]
    identity_numeric = spn_params["identity_numeric"]
    
    #Method to create leaves in the SPN
    def create_leaf(data, ds_context, scope):
        idx = scope[0]
        meta_type = ds_context.meta_types[idx]
        
        if meta_type == MetaType.REAL:
            if identity_numeric:
                return create_identity_leaf(data, scope)
        
            if prior_weight == 0.:
                return create_piecewise_leaf(data, ds_context, scope, prior_weight=None)
            else:
                return create_piecewise_leaf(data, ds_context, scope, prior_weight=prior_weight)
            

        elif meta_type == MetaType.DISCRETE:
            
            unique, counts = np.unique(data[:,0], return_counts=True)
            
            sorted_counts = np.zeros(len(ds_context.domains[idx]), dtype=np.float64)
            for i, x in enumerate(unique):
                sorted_counts[int(x)] = counts[i] 
            
            p = sorted_counts / data.shape[0]
            
            #Do regularization
            if prior_weight > 0.:
                p += prior_weight
            p = p/np.sum(p)
            
            return Categorical(p, scope)

        else:
            raise Exception("Mehtod learn_mspn_for_aqp(...) cannot create leaf for " + str(meta_type))
    
    #Set method to create leaves
    leaves = create_leaf
    
    #Set methods to cluster and to do the independence test
    if cols == "rdc":
        #split_cols = get_split_cols_RDC(rdc_threshold, ohe=ohe, linear=True)
        split_cols = get_split_cols_RDC_py(rdc_threshold, ohe=ohe, k=10, s=1 / 6,
                                           non_linearity=np.sin, n_jobs=1,
                                           rand_gen=rand_gen)
        
    if rows == "rdc":
        #split_rows = get_split_rows_RDC(ohe=ohe)
        split_rows = get_split_rows_RDC_py(n_clusters=2, ohe=ohe, k=10, s=1 / 6,
                                           non_linearity=np.sin, n_jobs=1,
                                           rand_gen=rand_gen)
        
    #This choses which operation is performed
    nextop = get_next_operation(min_instances_slice)
    
    #Learn the SPN
    root_node = learn_structure(numpy_data, ds_context, split_rows, split_cols, leaves, nextop)
    
    return root_node


'''
Adapted code
'''

def piecewise_likelihood_range(node, data, dtype=np.float64, **kwargs):
    """
    Returns the probability for the given ranges.
    
    ranges is multi-dimensional array:
    - First index specifies the instance
    - Second index specifies the feature
    
    Each entry of range contains a Range-object or None (e.g. for piecewise-node NumericRange exists).
    If the entry is None, then the log-likelihood probability of 0 will be returned.
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
        probs[i] = sum([_compute_probability_for_range(node, interval) for interval in rang.get_ranges()])

    return probs

def _compute_probability_for_range(node, interval):

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


def identity_likelihood_range(node, data, dtype=np.float64, **kwargs):
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


def categorical_likelihood_range(node, data, dtype=np.float64, **kwargs):
    """
    Returns the probability for the given ranges.
    
    ranges is multi-dimensional array:
    - First index specifies the instance
    - Second index specifies the feature
    
    Each entry of range contains a Range-object or None (e.g. for categorical-node NominalRange exists).
    If the entry is None, then the log-likelihood probability of 0 will be returned.
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


def Expectation(spn, feature_scope, evidence_scope, evidence, node_expectation, node_likelihood):

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


if __name__ == '__main__':
    
    data, feature_types = generate_gender_age_data(1000, 0)
    
    #Set identity_numeric=False, if you want to use PWLs
    spn_params = generate_spn_parameters(identity_numeric=True)
    
    root_node = build_spn(data, feature_types, spn_params, np.random.RandomState(1))

    
    from spn.io.Graphics import plot_spn
    plot_spn(root_node, "spn.pdf")
    
    #Import nodes
    from spn.experiments.AQP.leaves.identity.IdentityNumeric import IdentityNumeric
    from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
    from spn.structure.leaves.parametric.Parametric import Categorical
    from spn.structure.Base import Sum, Product
    
    #Import conditions
    from spn.experiments.AQP.Ranges import NominalRange, NumericRange
    
    #Import inference
    from spn.algorithms import Inference
    from spn.algorithms.Inference import sum_likelihood, prod_likelihood
    
    
    inference_support_ranges = {PiecewiseLinear : piecewise_likelihood_range, 
                                Categorical     : categorical_likelihood_range,
                                IdentityNumeric : identity_likelihood_range,
                                Sum             : sum_likelihood,
                                Product         : prod_likelihood}
    
    #Use None instead of np.nan
    ranges = np.array([[None, None, None],                                                          #Without any conditions
                       [NominalRange([0]), None, None],                                             #Only male
                       [NominalRange([0]), NominalRange([1]), None],                                #Only male and student
                       [NominalRange([0]), NominalRange([1]), NumericRange([[21,100]])],            #Only male and student and older than 21
                       [NominalRange([0]), NominalRange([1]), NumericRange([[10,15], [25,100]])]]   #Only male and student and age between 10 and 17 or 21 and 100
    )                  
    probabilities = Inference.likelihood(root_node, ranges, dtype=np.float64, node_likelihood=inference_support_ranges)
    
    print("Probabilities:")
    print(probabilities)
    print()
    
    
    
    #Sampling for given ranges
    from spn.algorithms import SamplingRange
    from spn.structure.leaves.piecewise.SamplingRange import sample_piecewise_node
    from spn.structure.leaves.parametric.SamplingRange import sample_categorical_node
    from spn.experiments.AQP.leaves.identity.SamplingRange import sample_identity_node
    
    node_sample_support = {PiecewiseLinear : sample_piecewise_node,
                           Categorical     : sample_categorical_node,
                           IdentityNumeric : sample_identity_node}
    
    n_instances = 10
    rang = np.array([NominalRange([0]), NominalRange([1]), NumericRange([[21,100]])])
    samples = SamplingRange.sample_instances(root_node, 3, n_instances, np.random.RandomState(123), ranges=rang, node_sample=node_sample_support, node_likelihood=inference_support_ranges)
    
    print("Samples:")
    print(samples)
    print()
    
    
    
    n_instances = 10
    rang = np.array([NominalRange([0]), NominalRange([0,1]), None])
    samples = SamplingRange.sample_instances(root_node, 3, n_instances, np.random.RandomState(123), ranges=rang, node_sample=node_sample_support, node_likelihood=inference_support_ranges)
    
    print("Samples:")
    print(samples)
    print()
    
    
    n_instances = 10
    rang = np.array([None, NominalRange([1]), NumericRange([[0,17]])])
    samples = SamplingRange.sample_instances(root_node, 3, n_instances, np.random.RandomState(123), ranges=rang, node_sample=node_sample_support, node_likelihood=inference_support_ranges)
    
    print("Samples:")
    print(samples)
    print()
    
    
    #Expectation
    from spn.experiments.AQP.leaves.identity.Expectation import identity_expectation
    
    node_expectation_support = {PiecewiseLinear : piecewise_expectation,
                                IdentityNumeric : identity_expectation}
    
    evidence = np.array([[None, None, None],                                                    
                         [NominalRange([0]), None, None],                                             
                         [NominalRange([0]), NominalRange([1]), None]])
    expect = Expectation(root_node, feature_scope=set([2]), evidence_scope=set([0,1]), evidence=evidence, node_expectation=node_expectation_support, node_likelihood=inference_support_ranges)
    print("Expectations:")
    print(expect)
    print()
    
    
    #Marginalize
    from spn.algorithms import Marginalization
    marg_spn =  Marginalization.marginalize(root_node, keep=set([2]))
    plot_spn(marg_spn, "marg_spn.pdf")
    
    
    #Statistics
    from spn.structure.Base import get_number_of_edges, Node
    num_nodes = len(get_nodes_by_type(root_node, Node))
    num_leafs = len(get_nodes_by_type(root_node, Leaf))
    num_sums = len(get_nodes_by_type(root_node, Sum))
    num_products = len(get_nodes_by_type(root_node, Product))
    edges = get_number_of_edges(root_node)
    print(num_nodes)
    print(num_leafs)
    print(num_sums)
    print(num_products)
    print(edges)
    
    