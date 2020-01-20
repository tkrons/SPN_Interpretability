'''
Created on 14.10.2019

@author: Moritz
'''

import numpy as np

from simple_spn import functions as fn
from spn.experiments.AQP.Ranges import NominalRange



'''
TODO include weights??????????
TODO inlcude variance????
'''


def feature_importance(spn, target_id, rang=None, value_dict=None, numeric_prec=50):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    if rang is not None : assert(rang[target_id] is None)
    if rang is not None: _, spn = fn.marg_rang(spn, rang)
    
    n_vals = len(value_dict[target_id][2])
    
    
    overall_pops = []
    for v in range(n_vals):
        tmp_rang = [None] * (np.max(spn.scope)+1)
        tmp_rang[target_id] = NominalRange([v])
        p, spn1 = fn.marg_rang(spn, tmp_rang)
        overall_pop = fn.get_overall_population(spn1, value_dict=value_dict, numeric_prec=numeric_prec)
        overall_pops.append([p, overall_pop])
    
    fis = []
    for f_id in spn1.scope:
        dists = [[p, overall_pop[f_id]] for p, overall_pop in overall_pops]
        fi = _compare_distributions(dists, value_dict[f_id])
        fis.append(fi)
    
    return fis  



def _compare_distributions(dists, value_info):
    
    
    if value_info[0] == "discrete":
        
        all_diffs = 0.0
        for i, dist1 in enumerate(dists):
            diffs = 0.0
            for j, dist2 in enumerate(dists):
                if i==j : continue
                diffs += np.abs(dist1[1]["y_means"] - dist2[1]["y_means"])
            
            diffs = diffs/(len(dists)-1)
            all_diffs += diffs
        
        all_diffs = all_diffs/(len(dists))
        res = np.sum(all_diffs)/len(all_diffs)
        return res

    elif value_info[0] == "numeric":
        
        all_diffs = 0.0
        for i, dist1 in enumerate(dists):
            
            diffs = 0.0
            x_vals = dist1[1]["x_vals"]
            x_pairs = zip(x_vals[:-1], x_vals[1:])
            y_vals1 = dist1[1]["y_means"]
            y_pairs1 = list(zip(y_vals1[:-1], y_vals1[1:]))
            for j, dist2 in enumerate(dists):
                if i==j:
                    continue
                y_vals2 = dist2[1]["y_means"]
                y_pairs2 = list(zip(y_vals2[:-1], y_vals2[1:]))
                
                dist_diff = 0.0
                for i, x in enumerate(x_pairs):
                    a1 = np.trapz(y_pairs1[i], x)
                    a2 = np.trapz(y_pairs2[i], x)
                    assert(a1 >= 0)
                    assert(a2 >= 0)
                    dist_diff += np.abs(a1 - a2)
                
                dist_diff /= 2
                
                diffs += dist_diff
            
            diffs = diffs/(len(dists)-1)
            all_diffs += diffs
        
        all_diffs = all_diffs/(len(dists))
        return all_diffs     
    
    else:
        raise Exception("Unknown attribute-type: " + str(value_info[0]))
    
    


def feature_importance_weighted(spn, target_id, rang=None, value_dict=None, numeric_prec=50):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    if rang is not None : assert(rang[target_id] is None)
    if rang is not None: _, spn = fn.marg_rang(spn, rang)
    
    n_vals = len(value_dict[target_id][2])
    
    
    overall_pops = []
    for v in range(n_vals):
        tmp_rang = [None] * (np.max(spn.scope)+1)
        tmp_rang[target_id] = NominalRange([v])
        p, spn1 = fn.marg_rang(spn, tmp_rang)
        overall_pop = fn.get_overall_population(spn1, value_dict=value_dict, numeric_prec=numeric_prec)
        overall_pops.append([p, overall_pop])
    
    fis = []
    for f_id in spn1.scope:
        dists = [[p, overall_pop[f_id]] for p, overall_pop in overall_pops]
        fi = _compare_distributions_weighted(dists, value_dict[f_id])
        fis.append(fi)
    
    return fis  
        
        


def _compare_distributions_weighted(dists, value_info):
    
    
    if value_info[0] == "discrete":
        
        all_diffs = 0.0
        ps = []
        for i, dist1 in enumerate(dists):
            diffs = 0.0
            for j, dist2 in enumerate(dists):
                if i==j : continue
                diffs += np.abs(dist1[1]["y_means"] - dist2[1]["y_means"])
            
            ps.append(dist1[0])
            diffs = dist1[0] * diffs/(len(dists)-1)
            all_diffs += diffs
        
        all_diffs = all_diffs/(np.sum(ps))
        res = np.sum(all_diffs)/len(all_diffs)
        return res

    elif value_info[0] == "numeric":
        
        all_diffs = 0.0
        ps = []
        for i, dist1 in enumerate(dists):
            
            diffs = 0.0
            x_vals = dist1[1]["x_vals"]
            x_pairs = zip(x_vals[:-1], x_vals[1:])
            y_vals1 = dist1[1]["y_means"]
            y_pairs1 = list(zip(y_vals1[:-1], y_vals1[1:]))
            for j, dist2 in enumerate(dists):
                if i==j:
                    continue
                y_vals2 = dist2[1]["y_means"]
                y_pairs2 = list(zip(y_vals2[:-1], y_vals2[1:]))
                
                dist_diff = 0.0
                for i, x in enumerate(x_pairs):
                    a1 = np.trapz(y_pairs1[i], x)
                    a2 = np.trapz(y_pairs2[i], x)
                    assert(a1 >= 0)
                    assert(a2 >= 0)
                    dist_diff += np.abs(a1 - a2)
                
                dist_diff /= 2
                
                diffs += dist_diff
            
            ps.append(dist1[0])
            diffs = dist1[0] *diffs/(len(dists)-1)
            all_diffs += diffs
        
        all_diffs = all_diffs/(np.sum(ps))
        return all_diffs     
    
    else:
        raise Exception("Unknown attribute-type: " + str(value_info[0]))








if __name__ == '__main__':
    
    from util import io
    from data import real_data
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.1)
    spn, _ = io.load(ident, "titanic", loc)
    value_dict = real_data.get_titanic_value_dict()
    spn = fn.marg(spn, keep=[0,1,2,3,4,5,6,7])
    #spn = example_spns.get_credit_spn()
    
    
    fis = feature_importance(spn, 1, value_dict=value_dict)
    print(fis)
    
    fis = feature_importance_weighted(spn, 1, value_dict=value_dict)
    print(fis)
    