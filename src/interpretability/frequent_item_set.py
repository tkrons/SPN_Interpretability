'''
Created on 15.10.2019

@author: Moritz
'''


import numpy as np

from util import io

from simple_spn import functions as fn
from spn.experiments.AQP.Ranges import NominalRange
from spn.structure.Base import Sum, Product
from spn.structure.leaves.parametric.Parametric import Categorical

'''
Assume only binary random variables at the moment


TODO

'''



def naive_approach(spn, min_support=0.1, value_dict=None):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    n_rv = np.max(spn.scope)+1
    
    ranges = np.full(shape=(n_rv, n_rv), fill_value=None)
    for i in range(len(ranges)):
        if len(value_dict[i][2]) == 2:
            ranges[i][i] = NominalRange([1])
    
    
    freq_sets = []
    new_freq_sets = []
    for i in range(len(spn.scope)):
        print("Iteration: " + str(i))
        
        if len(ranges) == 0: break
        probs = fn.probs(spn, ranges)
        
        new_freq_sets = []
        for i, prob in enumerate(probs):
            if prob >= min_support:
                ids = [i for i, cond in enumerate(ranges[i]) if cond is not None]
                new_freq_sets.append([prob, ids])
        
        freq_sets += new_freq_sets
        
        ranges = []
        for prob, ids in new_freq_sets:
            for i in range(n_rv):
                if i not in ids:
                    rang = np.array([None] * n_rv)
                    rang[ids] = NominalRange([1])
                    rang[i] = NominalRange([1])
                    ranges.append(rang)
        ranges = np.array(ranges)
        
    return freq_sets
    
    


def min_sub_population(node, min_support=0.5, candidates=[], cur_prob=1.0, binary_positive=True):
    
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
                freq_candidates = min_sub_population(c, min_support, new_candidates, cur_prob*weight, binary_positive=binary_positive)
            else:
                tmp = min_sub_population(c, min_support, new_candidates, cur_prob*weight, binary_positive=binary_positive)
                added_candidates = []
                for (sup_c, conds_c) in tmp:
                    found = False
                    for j, (sup_f, conds_f) in enumerate(freq_candidates):
                        if __same_conds(conds_f, conds_c):
                            found = True
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
                        
                        if binary_positive:
                            if len(c.p) == 2:
                                new_prob = prob * c.p[1]
                                
                                if new_prob >= min_support and new_prob > 0:
                                    new_conds = conds.copy() + [[c.scope[0], 1]]
                                    new_conds = sorted(new_conds, key=lambda x: (x[0], x[1]))
                                    added_candidates.append([new_prob, new_conds])
                        else:
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
                freq_candidates += min_sub_population(c, min_support, candidates.copy() + new_candidates.copy(), cur_prob, binary_positive=binary_positive)
        
    else:
        raise Exception("Unknown node")
    
    return freq_candidates


def __same_conds(conds1, conds2):
    if len(conds1) != len(conds2):
        return False
    for i in range(len(conds1)):
        if conds1[i][0] == conds2[i][0] and conds1[i][1] == conds2[i][1]:
            continue
        else:
            return False
    return True





if __name__ == '__main__':
    
    import time
    
    from data import real_data
    from simple_spn import learn_SPN
    
    
    ''' Data '''
    num_features = 20
    item_dataset, parametric_types = real_data.get_T10I4D(num_features, 100000)
    item_dataset[0][9] = 1
    dataset_name = "T10I4D_20"
    
    
    
    '''
    Create SPNs if necessary
    '''
    #rdc_thresholds = [0.1]
    #min_instances_slices = [0.01]
    #learn_SPN.create_parametric_spns(item_dataset, parametric_types, rdc_thresholds, min_instances_slices, folder=dataset_name)
    
    print("fdsafsdfasd")
    rdc_threshold = 0.1
    min_instances_slice = 0.01
    loc = "_spns"
    ident = "rdc=" + str(rdc_threshold) + "_mis=" + str(min_instances_slice)
    spn, const_time = io.load(ident, dataset_name, loc)
    
    print(spn)
    print(const_time)
    
    fn.print_statistics(spn)
    
    
    t0 = time.time()
    res = naive_approach(spn, min_support=0.0001)
    print(time.time()-t0)
    #print("Number item-sets naive approach: " + str(len(res)))
    
    
    
    t0 = time.time()
    res = min_sub_population(spn, min_support=0.0001)
    print(time.time()-t0)
    print("Number item-sets min sub population approach: " + str(len(res)))

    
    
    
    
    
    