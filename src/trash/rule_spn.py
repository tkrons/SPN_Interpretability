'''
Created on 17.06.2019

@author: Moritz
'''

import pandas as pd
import numpy as np

from spn.structure.Base import Context

from util import io



def _create_data(num_instances, rand_seed):    
    '''
    Name:
    Correlations:
    P(gender=male) = 50%
    P(gender=male) = 50%
    P(student=yes|gender=m) = 30%
    P(student=yes|gender=f) = 80%
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
            
        data.append(inst)
    
    from spn.structure.leaves.parametric.Parametric import Categorical
    return np.array(data), [Categorical, Categorical]
    
    

def learn_parametric_spn(data, parametric_types):
    
    from spn.algorithms.LearningWrappers import learn_parametric
    ds_context = Context(parametric_types=parametric_types).add_domains(data)
    ds_context.add_domains(data)
    spn = learn_parametric(data, ds_context, min_instances_slice=100, threshold=0.01)
    return spn



def get_nodes_with_weight(node, feature_id):
    from spn.structure.Base import Sum, Product, Leaf
    
    if feature_id in node.scope:
        if isinstance(node, Leaf):
            return [(1.0, node)]
        elif isinstance(node, Sum):
            weighted_nodes = []
            for i, child in enumerate(node.children):
                weight = node.weights[i]
                for (r_weight, r_node) in get_nodes_with_weight(child, feature_id):
                    weighted_nodes.append((weight*r_weight, r_node))
            return weighted_nodes
            
        elif isinstance(node, Product):
            weighted_nodes = []
            for i, child in enumerate(node.children):
                if feature_id in child.scope:
                    weighted_nodes += get_nodes_with_weight(child, feature_id)      
            return weighted_nodes
        else:
            raise Exception("Invalide node: " + str(node))



def spn_for_evidence(spn, evidence_ranges, node_likelihood=None, distribution_update_ranges=None):
    from spn.structure.Base import Sum, Product, Leaf, assign_ids
    from spn.algorithms.TransformStructure import Prune
    from spn.algorithms.Validity import is_valid
    from copy import deepcopy
    
    def spn_for_evidence_recursive(node):
        
        if isinstance(node, Leaf):
            if len(node.scope) > 1:
                raise Exception("Leaf Node with |scope| > 1")
            
            if evidence_ranges[node.scope[0]] is not None:
                t_node = type(node)
                if t_node in node_likelihood:
                    ranges = np.array([evidence_ranges])
                    prob =  node_likelihood[t_node](node, ranges, node_likelihood=node_likelihood)[0][0]
                    #if prob == 0:
                    #    newNode = deepcopy(node)
                    #else:
                    #    newNode = deepcopy(node)
                    #    distribution_update_ranges[t_node](newNode, evidence_ranges[node.scope[0]])
                    return prob, None
                else:
                    raise Exception('No log-likelihood method specified for node type: ' + str(type(node)))
            else:
                prob = 1
                newNode = deepcopy(node)
                
            return prob, newNode
            

        newNode = node.__class__()
        #newNode.scope = node.scope

        if isinstance(node, Sum):
            new_weights = []
            new_childs = []
            
            overall_prob = 0
            for i, c in enumerate(node.children):
                prob, new_child  = spn_for_evidence_recursive(c)
                new_prob = prob * node.weights[i]
                overall_prob += new_prob
                
                if new_child is not None and new_prob > 0:
                    new_weights.append(new_prob)
                    new_childs.append(new_child)
            
            if len(new_childs) == 0:
                return overall_prob, None
            
            new_weights = np.array(new_weights)
            newNode.weights = new_weights / np.sum(new_weights)
            newNode.children = new_childs
            
            newNode.scope = []
            for c in newNode.children:
                newNode.scope = list(set(newNode.scope + c.scope))
            
            return np.sum(new_weights), newNode
        
        
        elif isinstance(node, Product):
            new_childs = []
            newNode.scope = []
            
            new_prob = 1.
            for i, c in enumerate(node.children):
                prob, new_child = spn_for_evidence_recursive(c)
                new_prob *= prob
                if new_child is not None:
                    new_childs.append(new_child)
                    newNode.scope = list(set(newNode.scope + c.scope))
            
            if len(new_childs) is None:
                return new_prob, None
            
            newNode.children = new_childs
            return new_prob, newNode

    prob, newNode = spn_for_evidence_recursive(spn)
    assign_ids(newNode)
    newNode = Prune(newNode)
    valid, err = is_valid(newNode)
    assert valid, err

    return prob, newNode




def reduce_spn():
    from spn.experiments.AQP.Ranges import NominalRange, NumericRange
    from spn.structure.Base import Sum, Product
    from spn.algorithms.Inference import sum_likelihood, prod_likelihood
    
    from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
    from spn.structure.leaves.parametric.InferenceRange import categorical_likelihood_range
    from simple_spn.internal.UpdateRange import categorical_update_range
    
    evidence = [NominalRange([0]), None, None, None]
    
    inference_support_ranges = {Gaussian        : None, 
                                Categorical     : categorical_likelihood_range,
                                Sum             : sum_likelihood,
                                Product         : prod_likelihood}
    
    distribution_update_ranges = {Gaussian        : None, 
                                Categorical     : categorical_update_range}
    
    
    #spn_util.plot_spn(spn, "old.pdf")
    
    prob, spn = spn_for_evidence(spn, evidence, node_likelihood=inference_support_ranges, distribution_update_ranges=distribution_update_ranges)
    print(prob)
    
    


def get_flat_spn(spn, target_id):
    
    from spn.structure.Base import Sum, Product, Leaf, assign_ids
    from spn.algorithms.TransformStructure import Prune
    from spn.algorithms.Validity import is_valid
    from copy import deepcopy
    
    
    flat_spn = Sum()
    flat_spn.scope=spn.scope
    
    def create_flat_spn_recursive(node, distribution_mix, prob=1.0, independent_nodes=[]):
        
        if isinstance(node, Sum):
            for i, c in enumerate(node.children):
                forwarded_weight = node.weights[i] * prob
                create_flat_spn_recursive(c, distribution_mix, forwarded_weight, independent_nodes.copy())
        
        elif isinstance(node, Product):
            
            stop = False
            next_node = None
            
            for c in node.children:
                if target_id in c.scope:
                    if len(c.scope) == 1:
                        stop = True
                        independent_nodes.append(deepcopy(c))
                    else:
                        next_node = c
                else:
                    for feature_id in c.scope:
                        weighted_nodes = get_nodes_with_weight(c, feature_id)
                        t_node = type(weighted_nodes[0][1])
                        mixed_node = distribution_mix[t_node](weighted_nodes)
                        independent_nodes.append(mixed_node)
            
            if stop:
                flat_spn.weights.append(prob)
                prod = Product(children=independent_nodes)
                prod.scope = spn.scope
                flat_spn.children.append(prod)
                
            else:
                create_flat_spn_recursive(next_node, distribution_mix, prob, independent_nodes)
                
        else:
            raise Exception("Can only iterate over Sum and Product nodes")
        
        
    from simple_spn.internal.MixDistributions import mix_categorical
    
    distribution_mix = {Categorical : mix_categorical}
    
    
    create_flat_spn_recursive(spn, distribution_mix)
    assign_ids(flat_spn)
    flat_spn = Prune(flat_spn)
    valid, err = is_valid(flat_spn)
    assert valid, err

    return flat_spn



def _same_conds(conds1, conds2):
    if len(conds1) != len(conds2):
        return False
    
    for i in range(len(conds1)):
        if conds1[i][0] == conds2[i][0] and conds1[i][1] == conds2[i][1]:
            continue
        else:
            return False
    
    return True
    

def get_frequent_items(node, min_support=0.5, candidates=[], cur_prob=1.0):
    

    if isinstance(node, Sum):
        
        freq_candidates = []
        for i, c in enumerate(node.children):
            weight = node.weights[i]
            
            new_candidates = []
            for (prob, conds) in candidates:
                new_prob = weight * prob
                if new_prob >= min_support:
                    new_candidates.append([new_prob, conds])
            
            if len(freq_candidates) == 0:
                freq_candidates = get_frequent_items(c, min_support, new_candidates, cur_prob*weight)
            else:
                tmp = get_frequent_items(c, min_support, new_candidates, cur_prob*weight)
                added_candidates = []
                for (sup_c, conds_c) in tmp:
                    found = False
                    for j, (sup_f, conds_f) in enumerate(freq_candidates):
                        if _same_conds(conds_f, conds_c):
                            found = True
                            
                            #print(sup_c)
                            #print(freq_candidates[j])
                            freq_candidates[j][0] = sup_c + sup_f
                            break
                    
                    if not found:
                        added_candidates.append([sup_c, conds_c])
                        
                freq_candidates += added_candidates
        
            
    elif isinstance(node, Product):
        
        new_candidates = []
        for c in node.children:
            if len(c.scope) == 1:
                
                added_candidates = []
                for (prob, conds) in [[cur_prob,[]]] + candidates + new_candidates:
                    if isinstance(c, Categorical):
                        for i, p in enumerate(c.p):
                            new_prob = prob * p
                            
                            if new_prob >= min_support and new_prob > 0:
                                new_conds = conds.copy() + [[c.scope[0], i]]
                                new_conds = sorted(new_conds, key=lambda x: (x[0], x[1]))
                                added_candidates.append([new_prob, new_conds])
                            
                    else:
                        raise Exception("Cannot process node: " + str(c))
                
                new_candidates += added_candidates
        
        freq_candidates = new_candidates.copy()
        for c in node.children:
            if len(c.scope) > 1:
                freq_candidates += get_frequent_items(c, min_support, candidates.copy() + new_candidates.copy(), cur_prob)
        
        
        
                
    
    else:
        raise Exception("Unknown node")
    
    print("********************************************")
    print(node)
    _print_items(candidates)       
    _print_items(freq_candidates)
    print()
    print()
    print()
    print()
    
    return freq_candidates
    

def _print_items(freq_items):
    feature_dict = {0: ("g", ("m  ", "w  ")), 1: ("c", ("no ", "yes")), 2: ("s", ("no ", "yes")), 3: ("w", ("no ", "yes"))}
    freq_sets = []
    for (sup, conds) in freq_items:
        
        str_conds=[]
        for cond in conds:
            str_conds.append(feature_dict[cond[0]][0] + "=" + feature_dict[cond[0]][1][cond[1]])
        freq_sets.append(["(" + ", ".join(str_conds) + ")", sup]) 
        
        
    #freq_sets = sorted(freq_sets, key=lambda x : x[1], reverse=True)
    rule_df = pd.DataFrame(freq_sets, columns=["frequent set", "s_support"])
    
    io.print_pretty_table(rule_df)

def extract_rules(spn, feature_id=1):
    
    from spn.experiments.AQP.Ranges import NominalRange
    from spn.algorithms import Inference
    from simple_spn.internal.InferenceRange import categorical_likelihood_range
    from spn.structure.Base import Sum, Product
    from spn.algorithms.Inference import sum_likelihood, prod_likelihood
    from spn.structure.leaves.parametric.Parametric import Categorical
    
    inference_support_ranges = {Categorical     : categorical_likelihood_range,
                                    Sum             : sum_likelihood,
                                    Product         : prod_likelihood}
    
    
    
    
    
    freq_items = get_frequent_items(spn, min_support=0.0)
    freq_items_filtered = freq_items#filter(lambda x : any(cond[0] == feature_id for cond in x[1]), freq_items)
    freq_items_sorted = sorted(freq_items_filtered, key=lambda x: x[0], reverse=True)
    
    #evidence = numpy.empty((3,3,)
    
    
    feature_dict = {0: ("g", ("m  ", "w  ")), 1: ("c", ("no ", "yes")), 2: ("s", ("no ", "yes")), 3: ("w", ("no ", "yes"))}
    freq_sets = []
    for (sup, conds) in freq_items_sorted:
        
        str_conds=[]
        ranges = [None] * len(spn.scope)
        for cond in conds:
            ranges[cond[0]] = NominalRange([cond[1]])
            str_conds.append(feature_dict[cond[0]][0] + "=" + feature_dict[cond[0]][1][cond[1]])
            
        ranges = np.array([ranges])
        sup_spn = Inference.likelihood(spn, data=ranges, dtype=np.float64, node_likelihood=inference_support_ranges)[:,0][0]
        

        freq_sets.append(["(" + ", ".join(str_conds) + ")", sup, sup_spn]) 
        
        
    rules = sorted(freq_sets, key=lambda x : x[2], reverse=True)
    rule_df = pd.DataFrame(rules, columns=["frequent set", "s_support", "g_support"])
    
    io.print_pretty_table(rule_df.head(400))
    
    
    exit()
    
    
    
    
    rules = []
    for (sup, conds) in freq_items_sorted:
        
        rule_body = []
        rule_head = []
        conf = np.nan
        
        ranges = [None] * len(spn.scope)
        
        
        
        
        for cond in conds:
            if cond[0] == feature_id:
                rule_head.append(feature_dict[cond[0]][0] + "=" + feature_dict[cond[0]][1][cond[1]])
            else:
                rule_body.append(feature_dict[cond[0]][0] + "=" + feature_dict[cond[0]][1][cond[1]])
            
            ranges[cond[0]] = NominalRange([cond[1]])
        
        
        #Optimization possible
        ranges = np.array([ranges])
        prob_with_feature = Inference.likelihood(spn, data=ranges, dtype=np.float64, node_likelihood=inference_support_ranges)[:,0][0]
        
        ranges[0][feature_id] = None
        prob_without_feature = Inference.likelihood(spn, data=ranges, dtype=np.float64, node_likelihood=inference_support_ranges)[:,0][0]
        
        spn_sup = prob_without_feature
        spn_conf = prob_with_feature / prob_without_feature
        
        
        rules.append([" AND ".join(rule_body) + "-->" + " AND ".join(rule_head), sup, conf, spn_sup, spn_conf, spn_sup*spn_conf])
    
    
    rules = sorted(rules, key=lambda x : x[5], reverse=True)
    
    
     
    rule_df = pd.DataFrame(rules, columns=["Rule", "c_Support", "c_Confidence", "spn_Support", "spn_Confidence", "score"])
    
    #rule_df.drop_duplicates(subset=["Rule"], keep = True, inplace = True) 
    
    io.print_pretty_table(rule_df.head(400))
    

    
    pass



    
  

if __name__ == '__main__':
    
   
    from spn.structure.Base import Sum, Product, Leaf
    from spn.structure.leaves.parametric.Parametric import Categorical 
    
    
    spn1 = Categorical(p=[0.0, 1.0], scope=[2]) * Categorical(p=[0.5, 0.5], scope=[3]) 
    spn2 = Categorical(p=[1.0, 0.0], scope=[2]) * Categorical(p=[0.1, 0.9], scope=[3]) 
    spn3 = 0.3 * spn1 + 0.7 * spn2
    spn4 = Categorical(p=[0.0, 1.0], scope=[1]) * spn3
    
    spn6 = Product([Categorical(p=[1.0, 0.0], scope=[1]), Categorical(p=[0.0, 1.0], scope=[2]), Categorical(p=[1.0, 0.0], scope=[3])])
    spn6.scope = [1,2,3]
    
    spn7 = 0.8 * spn4 + 0.2 * spn6
    spn = spn7 * Categorical(p=[0.2, 0.8], scope=[0])
    
    #spn_util.plot_spn(spn, "rule_spn.pdf")
    
    
    
    extract_rules(spn)
    
    #res = get_frequent_items(spn)
    #print(res)
    
    
    
    
    
    
    
    exit()
    
    #x = get_nodes_with_weight(spn, feature_id=1)
    #print(x)
    
    flat_spn = get_flat_spn(spn, 1)
    
    #spn_util.plot_spn(flat_spn, "flat_rule_spn.pdf")
    
    
    from spn.experiments.AQP.Ranges import NominalRange, NumericRange
    from spn.structure.Base import Sum, Product
    from spn.algorithms.Inference import sum_likelihood, prod_likelihood
    
    from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
    from spn.structure.leaves.parametric.InferenceRange import categorical_likelihood_range
    from simple_spn.internal.UpdateRange import categorical_update_range
    
    
    inference_support_ranges = {Gaussian        : None, 
                                Categorical     : categorical_likelihood_range,
                                Sum             : sum_likelihood,
                                Product         : prod_likelihood}
    distribution_update_ranges = {Gaussian        : None, 
                                  Categorical     : categorical_update_range}
    
    evidence = [None, NominalRange([1]), None, None]
    prob, pos_spn = spn_for_evidence(flat_spn, evidence, node_likelihood=inference_support_ranges, distribution_update_ranges=distribution_update_ranges)
    #spn_util.plot_spn(pos_spn, "positive_flat_rule_spn.pdf")
    
    
    evidence = [None, NominalRange([0]), None, None]
    prob, neg_spn = spn_for_evidence(flat_spn, evidence, node_likelihood=inference_support_ranges, distribution_update_ranges=distribution_update_ranges)
    #spn_util.plot_spn(neg_spn, "negative_flat_rule_spn.pdf")
    
    
    
    
    
    
    
    
    
    
    
    exit()
    
    
    
    import os
    from util import io
    np.random.seed(123)
    path = os.path.dirname(os.path.abspath(__file__)) + "/"
    
    if os.path.exists(path + "spn.pkl"):
        spn = io.load_pickle(path + "spn.pkl")
    else:
        data, parametric_types = _create_data(10000, 0)
        spn = learn_parametric_spn(data, parametric_types)
        io.dump_pickle(path, "spn.pkl", spn)

    print(spn)
    
    
    #spn_util.plot_spn(spn, "rule_spn.pdf")
    
    
    
    