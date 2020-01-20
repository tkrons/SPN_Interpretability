'''
Created on 14.10.2019

@author: Moritz
'''



import numpy as np
from simple_spn import functions as fn



def interestingness_matrix(spn, value_dict=None, numeric_prec=20):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    sub_pops = fn.get_sub_populations(spn)
    
    all_scores = []
    for i, f_id in enumerate(sorted(spn.scope)):
        dists = np.array([dists[i] for _, dists in sub_pops])
        scores = _compare_distributions(dists, value_dict[f_id], numeric_prec)
        all_scores.append(scores)
    
    all_scores = np.array(all_scores).T
    
    return sub_pops, all_scores
        
        
def sub_population_interestingness(spn, value_dict=None, numeric_prec=20):
    sub_pops, m = interestingness_matrix(spn, value_dict, numeric_prec)
    return sub_pops, np.sum(m, axis=1)/m.shape[1]


def feature_interestingness(spn, value_dict=None, numeric_prec=20):
    sub_pops, m = interestingness_matrix(spn, value_dict, numeric_prec)
    f_interests = []
    probs = [prob for prob, _ in sub_pops]
    for i in range(m.shape[1]):
        score, _ = fn._compute_weighted_mean_and_variance(probs, m[:,i])
        f_interests.append(score)
    return f_interests




def _compare_distributions(dists, value_info, numeric_prec=50):
    scores = []
    for i, dist in enumerate(dists):
        tmp_dists = np.delete(dists, i)
        score = _compare_distribution(dist, tmp_dists, value_info, numeric_prec)
        scores.append(score)
    return scores



def _compare_distribution(dist1, dists, value_info, numeric_prec=50):
    
    if value_info[0] == "discrete":
        
        f_vals = list(value_info[2])
        probs1 = fn.evaluate_discrete_leaf(dist1, f_vals)
        
        score = 0.0
        for dist2 in dists:
            probs2 = fn.evaluate_discrete_leaf(dist2, f_vals)
            score += np.sum(np.abs(probs1 - probs2))
        
        return score/(len(dists))
    
    elif value_info[0] == "numeric":
        
        x_vals = np.linspace(value_info[2][0], value_info[2][1], numeric_prec)
        x_pairs = zip(x_vals[:-1], x_vals[1:])
        
        dens1 = fn.evaluate_numeric_density_leaf(dist1, x_vals)
        dens1_pairs = list(zip(dens1[:-1], dens1[1:]))
        
        score = 0.0
        for dist2 in dists:
            
            dens2 = fn.evaluate_numeric_density_leaf(dist2, x_vals)
            dens2_pairs = list(zip(dens2[:-1], dens2[1:]))
            
            dist_score = 0.0
            for i, x in enumerate(x_pairs):
                a1 = np.trapz(dens1_pairs[i], x)
                a2 = np.trapz(dens2_pairs[i], x)
                assert(a1 >= 0)
                assert(a2 >= 0)
                dist_score += np.abs(a1 - a2)
            
            dist_score /= 2
            score += dist_score
            
        return score/(len(dists))
        
    else:
        raise Exception("Unknown attribute-type: " + str(value_info[0]))





if __name__ == '__main__':

    from util import io
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.1)
    spn, value_dict, _ = io.load(ident, "titanic", loc)
    spn = fn.marg(spn, keep=[0,1,2,3,4,5,6,7])
    

    #sub_pops, m = interestingness_matrix(spn, value_dict=value_dict)
    #sub_pops, res = sub_population_interestingness(spn, value_dict=value_dict)
    res = feature_interestingness(spn, value_dict=value_dict)
    print(res)
    
    
    
    