import time
import numpy as np
import pandas as pd

import pyfpgrowth
from orangecontrib.associate import fpgrowth
from mlxtend.frequent_patterns import apriori






def apriori1(item_dataset, min_support=0.1, repetitions=10):
    
    df = pd.DataFrame(item_dataset)

    times = []
    for _ in range(repetitions):
        t0 = time.time()
        res = apriori(df, min_support=min_support)
        times.append(time.time() - t0)
    
    apr_item_sets = []
    for _, row in res.iterrows():
        conds = [[x,1]   for x in row["itemsets"]]
        apr_item_sets.append([row["support"], conds])
        
    exc_time = np.mean(times)
    
    return apr_item_sets, exc_time
      
      
      
      
            
def fpgrowth1(item_dataset, min_support=0.1, repetitions=10):
    
    #Transform data
    itemsets = []
    for row in item_dataset:
        vals = []
        for i, val in enumerate(row):
            if val == 1:
                vals.append(i)
        itemsets.append(vals)
    
    times = []
    for _ in range(repetitions):
        t0 = time.time()
        patterns = pyfpgrowth.find_frequent_patterns(itemsets, min_support*len(item_dataset))
        times.append(time.time() - t0)
    
    fp_item_sets = []
    for items, count in patterns.items():
        sup = count/len(item_dataset)
        conds = [[x,1]   for x in items]
        fp_item_sets.append([sup, conds])    
    exc_time = np.mean(times)
        
    return fp_item_sets, exc_time





def fpgrowth2(item_dataset, min_support=0.1, repetitions=10):
    
    #Transform data
    itemsets = []
    for row in item_dataset:
        vals = []
        for i, val in enumerate(row):
            if val == 1:
                vals.append(i)
        itemsets.append(vals)
    
    times = []
    for _ in range(repetitions):
        t0 = time.time()
        patterns = fpgrowth.frequent_itemsets(itemsets, min_support)
        times.append(time.time() - t0)
    
    fp_item_sets = []
    for (items, count) in patterns:
        sup = count/len(item_dataset)
        conds = [[x,1]   for x in items]
        fp_item_sets.append([sup, conds])         
    exc_time = np.mean(times)
    
    return fp_item_sets, exc_time
    


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''



from util import io
from spn.structure.Base import Sum, Product
from spn.structure.leaves.parametric.Parametric import Categorical


def spn1(spn, min_support=0.1, repetitions=10, binary_positive=True):

    times = []
    for _ in range(repetitions):
        t0 = time.time()
        item_sets = _get_frequent_items(spn, min_support, binary_positive=binary_positive)
        times.append(time.time() - t0)
    
    exc_time = np.mean(times)
    
    return item_sets, exc_time
    
    

def _get_frequent_items(node, min_support=0.5, candidates=[], cur_prob=1.0, binary_positive=True):
    
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
                freq_candidates = _get_frequent_items(c, min_support, new_candidates, cur_prob*weight, binary_positive=binary_positive)
            else:
                tmp = _get_frequent_items(c, min_support, new_candidates, cur_prob*weight, binary_positive=binary_positive)
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
                freq_candidates += _get_frequent_items(c, min_support, candidates.copy() + new_candidates.copy(), cur_prob, binary_positive=binary_positive)
        
    else:
        raise Exception("Unknown node")
    
    return freq_candidates
    
def __print_items(freq_candidates):
    freq_sets = []
    for (sup, conds) in freq_candidates:
        
        str_conds=[]
        for cond in conds:
            str_conds.append(cond[0] + "=" + cond[1])
        freq_sets.append(["(" + ", ".join(str_conds) + ")", sup]) 
        
    #freq_sets = sorted(freq_sets, key=lambda x : x[1], reverse=True)
    candidate_df = pd.DataFrame(freq_sets, columns=["frequent set", "s_support"])
    io.print_pretty_table(candidate_df)


def __same_conds(conds1, conds2):
    if len(conds1) != len(conds2):
        return False
    for i in range(len(conds1)):
        if conds1[i][0] == conds2[i][0] and conds1[i][1] == conds2[i][1]:
            continue
        else:
            return False
    return True









    
    