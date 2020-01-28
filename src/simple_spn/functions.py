'''
Created on 08.07.2019

@author: Moritz
'''

import warnings
import numpy as np

from spn.structure.Base import Node, Sum, Product, Leaf
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
from spn.experiments.AQP.leaves.identity.IdentityNumeric import IdentityNumeric



'''
#############################
### Compute probabilities ###
#############################
'''

#Import inference
from spn.algorithms import Inference
from spn.algorithms.Inference import sum_likelihood, prod_likelihood
from simple_spn.internal.InferenceRange import piecewise_likelihood_range, categorical_likelihood_range, identity_likelihood_range, gaussian_likelihood_range

inference_support_ranges = {PiecewiseLinear : piecewise_likelihood_range, 
                            Categorical     : categorical_likelihood_range,
                            IdentityNumeric : identity_likelihood_range,
                            Gaussian        : gaussian_likelihood_range,
                            Sum             : sum_likelihood,
                            Product         : prod_likelihood}



def prob(spn, rang):
    return probs(spn, [rang])[0]

def probs(spn, ranges):
    ranges = np.array(ranges)
    return Inference.likelihood(spn, ranges, dtype=np.float64, node_likelihood=inference_support_ranges).reshape(len(ranges))

def prob_spflow(spn, inst):
    return probs_spflow(spn, np.array([inst]))[0]

def probs_spflow(spn, data):
    return Inference.likelihood(spn, data, dtype=np.float64).reshape(len(data))  



'''
############################
### Compute expectations ###
############################
'''

from spn.algorithms.stats import Expectations
from simple_spn.internal.Expectation import Expectation, piecewise_expectation, identity_expectation, gaussian_expectation, categorical_expectation

node_expectation_support = {Categorical     : categorical_expectation,
                            PiecewiseLinear : piecewise_expectation,
                            IdentityNumeric : identity_expectation,
                            Gaussian        : gaussian_expectation}



def expect(spn, feature_id, rang):
    if rang is None : rang = [None] * (np.max(spn.scope)+1)
    assert(rang[feature_id] is None)
    return expects(spn, feature_id, np.array([rang]))[0]

def expects(spn, feature_id, ranges):
    return Expectation(spn, feature_id, ranges, node_expectation=node_expectation_support, node_likelihood=inference_support_ranges).reshape(len(ranges))

def expect_spnflow(spn, feature_scope, inst):
    return expects_spnflow(spn, feature_scope, np.array([inst]))[0]

def expects_spnflow(spn, feature_scope, data):
    return Expectations.Expectation(spn, feature_scope, data)


'''
######################
### Classification ###
######################
'''

from spn.algorithms import MPE
from spn.experiments.AQP.Ranges import NominalRange, NumericRange



def classify(spn, target_id, rang=None, value_dict=None):
    if rang is None : rang = [None] * (np.max(spn.scope)+1)
    return classifies(spn, target_id, np.array([rang]), value_dict)[0]

def classifies(spn, target_id, ranges, value_dict=None):
    if value_dict is None : value_dict = generate_adhoc_value_dict(spn)
    if ranges is None : ranges = np.array([[None] * (np.max(spn.scope)+1)])
    assert(not any(ranges[:,target_id]))
    ps = []
    for v in range(len(value_dict[target_id][2])):
        ranges[:,target_id] = NominalRange([v])
        ps.append(probs(spn, ranges))
    return np.argmax(ps, axis=0)


def classify_dataset(spn, target_id, df, transform=False, value_dict=None, epsilon=0.01):
    if value_dict is None : value_dict = generate_adhoc_value_dict(spn)
    sorted_scope = sorted(spn.scope)
    
    if transform:
        inv_val_dict = { v[1]: { v2: k2 for k2, v2 in v[2].items() } for _, v in value_dict.items() if v[0] == "discrete"}
        for col_name, map_dict in inv_val_dict.items():
            df[col_name] = df[col_name].map(map_dict)
    
    values = np.array(df.values)
    ranges = np.full(shape=(len(values), np.max(spn.scope)+1), fill_value=None)
    for i, col in enumerate(values.T):
        f_id = sorted_scope[i]
        if f_id == target_id: continue
        
        if value_dict[f_id][0] == "discrete":
            for j, v in enumerate(col) : ranges[j, f_id] = NominalRange([v])
        elif value_dict[f_id][0] == "numeric":
            bound = epsilon*(value_dict[f_id][2][1]-value_dict[f_id][2][0])
            for j, v in enumerate(col) : ranges[j, f_id] = NumericRange([[v-bound, v+bound]])
        else:
            raise Exception("Unknown attribute-type: " + str(value_dict[f_id][0]))

    return classifies(spn, target_id, ranges, value_dict)


def mpe_spflow(spn, target_id, input_data):
    data = np.array(input_data)
    data[:,target_id] = np.nan
    return MPE.mpe(spn, data)[:,target_id]
    


'''
###################
### Marginalize ###
###################
'''

from spn.algorithms import Marginalization
from simple_spn.internal import MarginalizationRange


def marg(spn, keep):
    return Marginalization.marginalize(spn, keep=keep)

def marg_rang(spn, rang, node_likelihood=inference_support_ranges, keep=None):
    return MarginalizationRange.marg_rang(spn, rang, node_likelihood, keep=keep)



'''
###################
### Sampleing ###
###################
'''

from simple_spn.internal import SamplingRange
from simple_spn.internal.SamplingRange import sample_gaussian_node
from spn.structure.leaves.piecewise.SamplingRange import sample_piecewise_node
from spn.structure.leaves.parametric.SamplingRange import sample_categorical_node
from spn.experiments.AQP.leaves.identity.SamplingRange import sample_identity_node
    
node_sample_support = {PiecewiseLinear : sample_piecewise_node,
                       Categorical     : sample_categorical_node,
                       IdentityNumeric : sample_identity_node,
                       Gaussian        : sample_gaussian_node}


def sampling(spn, n_samples, random_seed=123):
    rang = [None] * (np.max(spn.scope)+1)
    return SamplingRange.sample_instances(spn, len(spn.scope), n_samples, np.random.RandomState(random_seed), ranges=rang, node_sample=node_sample_support, node_likelihood=inference_support_ranges)
 

def sampling_rang(spn, rang, n_samples, random_seed=123):
    return SamplingRange.sample_instances(spn, len(spn.scope), n_samples, np.random.RandomState(random_seed), ranges=rang, node_sample=node_sample_support, node_likelihood=inference_support_ranges)
    



'''
########################
### Interpretability ###
########################
'''

import itertools


def evaluate_discrete_leaf(leaf, f_vals):
    f_id = leaf.scope[0]
    ranges = np.array([f_id*[None] + [NominalRange([x])] for x in f_vals])
    return probs(leaf, ranges)



def evaluate_numeric_density_leaf(leaf, x_vals):
    f_id = leaf.scope[0]
    ranges = np.array([f_id*[None] + [NumericRange([[x]])] for x in x_vals])
    return probs(leaf, ranges)



def get_overall_population(spn, value_dict=None, numeric_prec=50):
    
    if value_dict is None: value_dict = generate_adhoc_value_dict(spn)
    sub_pops = get_sub_populations(spn, sort=False)   
    
    result_dict = {}
    for i, f_id in enumerate(sorted(spn.scope)):
        
        result_dict[f_id] = {}
        result_dict[f_id]["feature_name"] = value_dict[f_id][1]
        
        if value_dict[f_id][0] == "discrete":
            result_dict[f_id]["feature_type"] = "discrete"
            
            val_pairs = sorted(value_dict[f_id][2].items(), key=lambda x: x[0])
            x_vals = [x[0] for x in val_pairs]
            
            y = []
            weights = []
            for [p, dists] in sub_pops:
                y_vals = evaluate_discrete_leaf(dists[i], f_vals=x_vals)
                y.append(y_vals)
                weights.append(p)
            
            y = np.array(y)
            y_means = []
            y_vars = []
            for i in range(len(x_vals)):
                y_mean, y_var = _compute_weighted_mean_and_variance(weights, y[:,i])
                y_means.append(y_mean)
                y_vars.append(y_var)
            result_dict[f_id]["x_labels"] = [x[1] for x in val_pairs]
            result_dict[f_id]["y_means"] =  np.array(y_means)
            result_dict[f_id]["y_vars"] =  np.array(y_vars)
            
        elif value_dict[f_id][0] == "numeric":
            result_dict[f_id]["feature_type"] = "numeric"
            
            x_vals = np.linspace(value_dict[f_id][2][0], value_dict[f_id][2][1], numeric_prec, endpoint=True)
            
            y = []
            weights = []
            for [p, dists] in sub_pops:
                y_vals = evaluate_numeric_density_leaf(dists[i], x_vals)
                y.append(y_vals)
                weights.append(p)
            
            y = np.array(y)
            y_means = []
            y_vars = []
            for i in range(len(x_vals)):
                y_mean, y_var = _compute_weighted_mean_and_variance(weights, y[:,i])
                y_means.append(y_mean)
                y_vars.append(y_var)
            result_dict[f_id]["x_vals"] =  x_vals
            result_dict[f_id]["y_means"] =  np.array(y_means)
            result_dict[f_id]["y_vars"] =  np.array(y_vars)

        else:
            raise Exception("Unknown attribute-type: " + str(value_dict[f_id][0]))
    
    return result_dict



def _compute_weighted_mean_and_variance(weights, vals):
    tot_weight = np.sum(weights)
    non_zero_weights = np.sum([1. for w in weights if w > 0.])
    mean = sum([weights[i]*val for i, val in enumerate(vals)]) / tot_weight
    variance = sum([weights[i]*(val-mean)*(val-mean) for i, val in enumerate(vals)])/(((non_zero_weights-1.)*tot_weight)/non_zero_weights)
    return mean, variance




def get_sub_populations(spn, sort=True, top=None):
    sub_pops = _get_sub_populations_recursive(spn, rang=None)
    sub_pops = [[prob, dists] for [prob, dists] in sub_pops if prob > 0]
    sub_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in sub_pops]
    if sort:
        sub_pops = sorted(sub_pops, key=lambda x: x[0], reverse=True)
    if top is not None:
        sub_pops = sub_pops[:top]
    return sub_pops



def _get_sub_populations_recursive(spn, rang=None):
    
    if isinstance(spn, Leaf):
        if rang is None:
            return [[1, [spn]]]
        else: 
            return [[prob(spn, rang), [spn]]]
        
    elif isinstance(spn, Sum):
        collected_subs = []
        for i, child in enumerate(spn.children):
            weight = spn.weights[i]
            retrieved_subs = _get_sub_populations_recursive(child, rang=rang)
            for [p, dists] in retrieved_subs:
                collected_subs.append([weight*p, dists])
        return collected_subs
    
    elif isinstance(spn, Product):
        results = []
        for child in spn.children:
            results.append(_get_sub_populations_recursive(child, rang=rang))
        collected_subs = []
        for combo in list(itertools.product(*results)):
            new_prob = 1
            new_dists = []
            for [p, dists] in combo:
                new_prob *= p
                new_dists += dists
            collected_subs.append([new_prob, new_dists])
        return collected_subs
    
    else:
        raise Exception("Invalide node: " + str(spn))




'''
#############################
### information of leaves ###
#############################
'''


def get_nodes_with_weight(spn, feature_id):

    if feature_id in spn.scope:
        if isinstance(spn, Leaf):
            return [(1.0, spn)]
        elif isinstance(spn, Sum):
            weighted_nodes = []
            for i, child in enumerate(spn.children):
                weight = spn.weights[i]
                for (r_weight, r_node) in get_nodes_with_weight(child, feature_id):
                    weighted_nodes.append((weight*r_weight, r_node))
            return weighted_nodes
            
        elif isinstance(spn, Product):
            weighted_nodes = []
            for i, child in enumerate(spn.children):
                if feature_id in child.scope:
                    weighted_nodes += get_nodes_with_weight(child, feature_id)      
            return weighted_nodes
        else:
            raise Exception("Invalide node: " + str(spn))



def get_leaf_type_and_id(spn, feature_id):
    if isinstance(spn, Leaf) and spn.scope[0] == feature_id:
        return [spn.scope[0], type(spn)]
    elif isinstance(spn, Sum):
        return get_leaf_type_and_id(spn.children[0], feature_id)
    elif isinstance(spn, Product):
        for child in spn.children:
            if feature_id in child.scope:
                return get_leaf_type_and_id(child, feature_id)
    else:
        raise Exception("Invalide node: " + str(spn))
    


def get_parametric_types(spn):
    return [get_leaf_type_and_id(spn, i)[1] for i in sorted(spn.scope)]



def get_parametric_types_and_feature_ids(spn):
    return [get_leaf_type_and_id(spn, i) for i in sorted(spn.scope)]



'''
###############################################
### Value-dictionary and dataset processing ###
###############################################
'''


def transform_dataset(df, feature_types=None):
    """
    todo bug with boolean enncoding!
    :param df: any df
    :param feature_types: list of column types of df: ['discrete', 'numeric']
    len(feature_types) == len(df.columns)
    :return: transformed_df, dict to transform back, parametric types
    """
    if feature_types is None: feature_types = get_feature_types_from_dataset(df)
    
    value_dict = {}
    trans_df = df.copy()
    for i, col_name in enumerate(df.columns):
        
        if feature_types[i] == "discrete":
            if df[col_name].dtype in [np.dtype(np.int64).type, np.bool]:
                unq_vals = sorted(df[col_name].unique())
            else:
                unq_vals = df[col_name].unique()
                
            v_dict = {i : v for i,v in enumerate(unq_vals)}
            inv_v_dict = {v : i for i,v in v_dict.items()}
            trans_df[col_name] = trans_df[col_name].map(inv_v_dict)
            value_dict[i] = ["discrete", col_name, v_dict]
            
        elif feature_types[i] == "numeric":
            value_dict[i] = ["numeric", col_name, [df[col_name].min(), df[col_name].max()]]
        else:
            raise Exception("Unknown attribute-type: " + str(feature_types[i]))  
    
    return trans_df, value_dict, get_standard_parametric_types(feature_types)



def get_feature_types_from_dataset(df):
    feature_types = []
    for col_name in df.columns:
        if df[col_name].dtype == np.dtype(np.float64).type:
            feature_types.append("numeric")
        elif df[col_name].dtype == np.dtype(np.int64).type:
            n_unique_vals = len(df[col_name].unique())
            if n_unique_vals < 30:
                feature_types.append("discrete")
            elif n_unique_vals < 100:
                warnings.warn("High number of discrete values " + str(n_unique_vals) + "for dataset column: " + col_name)
                feature_types.append("discrete")
            else:
                feature_types.append("numeric")
        elif df[col_name].dtype == np.dtype(np.object).type:
            feature_types.append("discrete")
        elif df[col_name].dtype == np.dtype(np.bool).type:
            feature_types.append("discrete")
        else:
            raise Exception("Unknown dtype: " + str(df[col_name].dtype))  
    return feature_types



def get_standard_parametric_types(feature_types):
    parametric_types = []
    for feature_type in feature_types:
        if feature_type == "discrete": parametric_types.append(Categorical)
        elif feature_type == "numeric": parametric_types.append(Gaussian)
        else: raise Exception("Unknown attribute-type: " + str(feature_type))  
    return parametric_types



def generate_adhoc_value_dict_from_data(data, parametric_types):
    val_dict = {}
    for i, p_type in enumerate(parametric_types):
        if p_type == Categorical:
            val_dict[i] = ["discrete",  "Attr_"+str(i), {i : v for i,v in enumerate(sorted(np.unique(data[:,i])))}]
        elif p_type==Gaussian or p_type==PiecewiseLinear or p_type==IdentityNumeric:
            val_dict[i] = ["numeric",  "Attr_"+str(i), [np.min(data[:,i]), np.max(data[:,i])]]
        else:
            raise Exception("Cannot process parametric-type: " + str(p_type))
    return val_dict
    


def generate_adhoc_value_dict(spn):
    val_dict = {}
    for leaf in get_nodes_by_type(spn, Leaf):
        assert(len(leaf.scope) == 1)
        feature_id = leaf.scope[0]
        
        if feature_id in val_dict:
            if val_dict[feature_id][0] == "numeric":
                v_min, v_max = _get_min_max_numeric_from_leaf(leaf)            
                if v_min < val_dict[feature_id][2][0] : val_dict[feature_id][2][0] = v_min
                if v_max > val_dict[feature_id][2][1] : val_dict[feature_id][2][1] = v_max
        else:
            if isinstance(leaf, Categorical):
                val_dict[feature_id] = ["discrete", "Attr_"+str(feature_id), {i:str(i) for i in range(len(leaf.p))}]
            elif isinstance(leaf, Gaussian) or isinstance(leaf, PiecewiseLinear) or isinstance(leaf, IdentityNumeric):
                val_dict[feature_id] = ["numeric", "Attr_"+str(feature_id), _get_min_max_numeric_from_leaf(leaf)]
            else:
                raise Exception("Cannot process node-type: " + str(leaf))
    
    return val_dict
    
    
    
def _get_min_max_numeric_from_leaf(leaf):
    if isinstance(leaf, Gaussian):
        return [leaf.mean-3*leaf.stdev, leaf.mean+3*leaf.stdev]
    elif isinstance(leaf, PiecewiseLinear):
        return [leaf.x_range[0], leaf.x_range[-1]]
    elif isinstance(leaf, IdentityNumeric):
        return [leaf.vals[0], leaf.vals[-1]]
    else:
        raise Exception("Cannot process node-type: " + str(leaf))



'''
##################
### Statistics ###
##################
'''


from spn.structure.Base import get_nodes_by_type, get_number_of_edges


def print_statistics(spn):
    print("#nodes:    " + str(get_num_nodes(spn)))
    print("#sums:     " + str(get_num_sums(spn)))
    print("#products: " + str(get_num_products(spn)))
    print("#leaves:   " + str(get_num_leafs(spn)))
    print("#edges:    " + str(get_num_edges(spn)))


def get_num_nodes(spn):
    return len(get_nodes_by_type(spn, Node))

def get_num_leafs(spn):
    return len(get_nodes_by_type(spn, Leaf))

def get_num_sums(spn):
    return len(get_nodes_by_type(spn, Sum))

def get_num_products(spn):
    return len(get_nodes_by_type(spn, Product))

def get_num_edges(spn):
    return get_number_of_edges(spn)

def get_num_sub_populations(spn):
    if isinstance(spn, Leaf):
        return 1
    elif isinstance(spn, Sum):
        sub_populations = 0
        for child in spn.children:
            sub_populations += get_num_sub_populations(child) 
        return sub_populations
    elif isinstance(spn, Product):
        sub_populations = 1
        for child in spn.children:
            sub_populations *= get_num_sub_populations(child)
        return sub_populations
    else:
        raise Exception("Invalide node: " + str(spn))

def get_num_values_feature(spn, feature_id):
    if isinstance(spn, Leaf) and spn.scope[0] == feature_id:
        if isinstance(spn, Categorical):
            return len(spn.p)
        else:
            raise Exception("Unknown node: " + str(spn))
    elif isinstance(spn, Sum):
        return get_num_values_feature(spn.children[0], feature_id)
    elif isinstance(spn, Product):
        for child in spn.children:
            if feature_id in child.scope:
                return get_num_values_feature(child, feature_id)
    else:
        raise Exception("Invalide node: " + str(spn))





'''
################
### Plotting ###
################
'''


def plot_spn(spn, fname="plot.pdf", value_dict=None):
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout
    import matplotlib.pyplot as plt
    
    plt.clf()
    g, labels = __get_networkx_obj(spn)
    #pos = graphviz_layout(g, prog="neato")
    pos = graphviz_layout(g, prog='dot')
    plt.figure(figsize=(18, 12))
    ax = plt.gca()
    
    nx.draw(g, pos, with_labels=True, arrows=False, node_color="#DDDDDD", edge_color="#888888", node_size=1250, labels=labels, font_size=10)
    ax.collections[0].set_edgecolor("#333333")
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=nx.get_edge_attributes(g, "weight"), clip_on=False, alpha=0.6)
    if value_dict:
        text = '\n'.join([str(i)+': '+str(val[1]) for i, val in enumerate(value_dict.items())])
        #TODO fix annotation for das_beispiel
        plt.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.savefig(fname, bbox_inches="tight", pad_inches=0)



def plot_spn_to_svg(root_node, fname="plot.svg"):
    import networkx.drawing.nx_pydot as nxpd

    g, _ = __get_networkx_obj(root_node)

    pdG = nxpd.to_pydot(g)
    svg_string = pdG.create_svg()

    f = open(fname, "wb")
    f.write(svg_string)
    
    
def __get_networkx_obj(spn):
    import networkx as nx
    #TODO prettier graph? with node plots and right variable names
    all_nodes = get_nodes_by_type(spn)
    g = nx.Graph()

    labels = {}
    for n in all_nodes:

        if isinstance(n, Sum):
            label = "+\n{}".format(n.scope)
        elif isinstance(n, Product):
            label = "x"
        elif isinstance(n, Gaussian):
            label = "G" + str(n.scope[0])  + "\n(" + str(round(n.mean, 2)) + ", " + str(round(n.stdev, 2)) + ")"
        elif isinstance(n, Categorical):
            vals = [round(x,2) for x in n.p]
            label = "C" + str(n.scope[0]) + " (" + str(vals) + ")"
        else:
            label = "Unk" + str(n.scope[0])
        g.add_node(n.id)
        labels[n.id] = label

        if isinstance(n, Leaf):
            continue
        for i, c in enumerate(n.children):
            edge_label = ""
            if isinstance(n, Sum):
                edge_label = np.round(n.weights[i], 2)
            g.add_edge(c.id, n.id, weight=edge_label)

    return g, labels

def plot_subgroups_TSNE(spn):
    #idea: color each spn subgroup on a TSNE plot, kind of a benchmark? if too many subgroups, take higher levels
    # todo plot_subgroups TSNE
    return


