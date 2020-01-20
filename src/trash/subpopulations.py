'''
Created on 28.08.2019

@author: Moritz
'''

from spn.experiments.AQP.Ranges import NominalRange
from spn.structure.Base import  Sum, Product, Leaf
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

import os
import itertools
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from util import io
from simple_spn import learn_SPN
from simple_spn import functions as fn
from data import real_data


def _compute_weighted_mean_and_variance(vals):
    tot_prob = sum([prob for (prob, _) in vals])
    non_zero_weights = sum([1. for (prob, _) in vals if prob > 0.])
    mean = sum([prob*val for (prob, val) in vals]) / tot_prob
    variance = sum([prob*(val-mean)*(val-mean) for (prob, val) in vals])/(((non_zero_weights-1.)*tot_prob)/non_zero_weights)
    return mean, variance



def _get_sub_population_distributions(spn, rang=None):
    
    if isinstance(spn, Leaf):
        if rang is None:
            return [[1, [spn]]]
        else: 
            return [[fn.prob(spn, rang), [spn]]]
        
    elif isinstance(spn, Sum):
        collected_subs = []
        for i, child in enumerate(spn.children):
            weight = spn.weights[i]
            retrieved_subs = _get_sub_population_distributions(child, rang=rang)
            for [prob, dists] in retrieved_subs:
                collected_subs.append([weight*prob, dists])
        return collected_subs
    
    elif isinstance(spn, Product):
        results = []
        for child in spn.children:
            results.append(_get_sub_population_distributions(child, rang=rang))
        collected_subs = []
        for combo in list(itertools.product(*results)):
            new_prob = 1
            new_dists = []
            for [prob, dists] in combo:
                new_prob *= prob
                new_dists += dists
            collected_subs.append([new_prob, new_dists])
        return collected_subs
    
    else:
        raise Exception("Invalide node: " + str(spn))




'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''
   


def visualize_sub_population_distributions(spn, top=None, rang=None, value_dict=None, save_path=None):
    
    sub_pops = _get_sub_population_distributions(spn, rang=rang)
    sub_pops = [[prob, dists] for [prob, dists] in sub_pops if prob > 0]
    sub_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in sub_pops]
    sorted_sub_pops = sorted(sub_pops, key=lambda x: x[0], reverse=True)
    
    if top is not None:
        sorted_sub_pops = sorted_sub_pops[:top]
    
    ncols = len(spn.scope)
    nrows = len(sorted_sub_pops)
    figsize_x = ncols*3
    figsize_y = nrows*2
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)#, sharey=True)#, sharex=True)
    
    for i, [prob, dists] in enumerate(sorted_sub_pops):
        for j, dist in enumerate(dists):
            plot = axes[i][j]
            if isinstance(dist, Categorical):
                plot.bar(np.arange(len(dist.p)), dist.p)
                plot.set_ylim(0, 1)
                plot.set_xticks(np.arange(len(dist.p)))
                if value_dict is not None:
                    plot.set_xticklabels([value_dict[dist.scope[0]][2][x] for x in range(len(dist.p))])
                
                if rang is not None and rang[dist.scope[0]] is not None:
                    selected_vals = [dist.p[i] for i in rang[dist.scope[0]].get_ranges()]
                    plot.bar(rang[dist.scope[0]].get_ranges(), selected_vals, color="red")
            
            elif isinstance(dist, Gaussian):
                x_vals = np.linspace(dist.mean-3*dist.stdev, dist.mean+3*dist.stdev, 30, endpoint=True)
                y_vals = []
                for x in x_vals:
                    y_vals.append(stats.norm.pdf(x, dist.mean, dist.stdev))
                
                plot.plot(x_vals, y_vals)
                if value_dict is not None:
                    plot.set_xlim(value_dict[dist.scope[0]][2][0], value_dict[dist.scope[0]][2][1])
            else:
                raise Exception("Not implemented for other than categorical or gaussian")
                       
    pad_col = 5
    if value_dict is None:
        feature_names = ["Feature " + str(x) for x in sorted(spn.scope)]
    else:
        feature_names = [value_dict[x][1] for x in sorted(spn.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    pad_row = 5
    for ax, row in zip(axes[:,0], [round(x,6) for [x,_] in sorted_sub_pops]):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad_row, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
    plt.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



def visualize_sub_population_distributions_target(spn, target_id, top=None, rang=None, value_dict=None, save_path=None, precision_numeric=50): 
    assert(value_dict is not None)

    pt_fids = fn.get_parametric_types_and_feature_ids(spn)
    n_target_vals = len(value_dict[target_id][2])
    
    ncols = len(pt_fids)
    nrows = n_target_vals
    
    figsize_x = ncols*3
    figsize_y = n_target_vals*2
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)
    
    tot_prob_vals = []
    for v in range(n_target_vals):
        
        if rang is None:
            rang = np.array([None]*(max(spn.scope)+1))
        
        rang[target_id] = NominalRange([v])
        sub_pops = _get_sub_population_distributions(spn, rang=rang)
        sub_pops = [[prob, dists] for [prob, dists] in sub_pops if prob > 0]
        sub_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in sub_pops]
        sorted_sub_pops = sorted(sub_pops, key=lambda x: x[0], reverse=True)
        
        if top is not None:
            sorted_sub_pops = sorted_sub_pops[:top]
        
        tot_prob_vals.append(round(sum([prob for prob, dists in sorted_sub_pops]),6))
        
        for i, (f_id, param_type) in enumerate(pt_fids):
            
            #if f_id == target_id:
            #    continue
            
            plot = axes[v][i]
            if param_type is Categorical:
                n_vals = len(value_dict[f_id][2])
                bars = [[] for _ in range(n_vals)]
                for [prob, dists] in sorted_sub_pops:
                    dist = dists[i]
                    for k in range(n_vals):
                        bars[k].append([prob, dist.p[k]])
                
                means = []
                stds = []
                for bar_vals in bars:
                    m, va = _compute_weighted_mean_and_variance(bar_vals)
                    means.append(m)
                    stds.append(np.sqrt(va))
                
                plot.bar(np.arange(len(bars)), means, yerr=stds, align='center', ecolor='black')
                plot.set_ylim(0, 1)
                plot.set_xticks(np.arange(len(dist.p)))
                if value_dict is not None:
                    plot.set_xticklabels([value_dict[dist.scope[0]][2][x] for x in range(len(dist.p))])
                
            elif param_type is Gaussian:
                
                bars = [[] for _ in range(precision_numeric)]
                for [prob, dists] in sorted_sub_pops:
                    dist = dists[i]
                    x_vals = np.linspace(value_dict[f_id][2][0], value_dict[f_id][2][1], precision_numeric, endpoint=True)
                    for k, x_val in enumerate(x_vals):
                        bars[k].append([prob, stats.norm.pdf(x_val, dist.mean, dist.stdev)])

                means = []
                stds = []
                for bar_vals in bars:
                    m, va = _compute_weighted_mean_and_variance(bar_vals)
                    means.append(m)
                    stds.append(np.sqrt(va))
                means = np.array(means)
                stds = np.array(stds)
                
                plot.plot(x_vals, means)
                plot.fill_between(x_vals, means-stds, means+stds, alpha=0.5)

            else:
                raise Exception("Not implemented for other than categorical or gaussian")
                
    pad_col = 5
    if value_dict is None:
        feature_names = ["P(Feature " + str(x) + "|Feature " + str(target_id) + ")" for x in sorted(spn.scope)]
    else:
        feature_names = ["P(" + value_dict[x][1] + "|" + value_dict[target_id][1] + ")"  for x in sorted(spn.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
        
    pad_row = 5
    for i, prob in enumerate(tot_prob_vals):
        axes[i][0].annotate(str(round(prob*100,4)) + "%\n" + value_dict[target_id][1] + "=" + value_dict[target_id][2][i], xy=(0, 0.5), xytext=(-axes[i][0].yaxis.labelpad - pad_row, 0), xycoords=axes[i][0].yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
    plt.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



def visualize_sub_population_distributions_target_compact(spn, target_id, top=None, rang=None, value_dict=None, save_path=None, precision_numeric=50): 
    assert(value_dict is not None)


    pt_fids = fn.get_parametric_types_and_feature_ids(spn)
    n_target_vals = len(value_dict[target_id][2])
    
    ncols = len(pt_fids)
    nrows = 1
    
    figsize_x = ncols*4
    figsize_y = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)
    gap_size = 1/(n_target_vals + 1)
    
    
    tot_prob_vals = []
    for v in range(n_target_vals):
        
        if rang is None:
            rang = np.array([None]*(max(spn.scope)+1))
        
        rang[target_id] = NominalRange([v])
        sub_pops = _get_sub_population_distributions(spn, rang=rang)
        sub_pops = [[prob, dists] for [prob, dists] in sub_pops if prob > 0]
        sub_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in sub_pops]
        sorted_sub_pops = sorted(sub_pops, key=lambda x: x[0], reverse=True)
        
        if top is not None:
            sorted_sub_pops = sorted_sub_pops[:top]
        
        tot_prob_vals.append(round(sum([prob for prob, dists in sorted_sub_pops]),6))
        
        for i, (f_id, param_type) in enumerate(pt_fids):
            
            #if f_id == target_id:
            #    continue

            plot = axes[0][i]
            if param_type is Categorical:
                n_vals = len(value_dict[f_id][2])
                bars = [[] for _ in range(n_vals)]
                for [prob, dists] in sorted_sub_pops:
                    dist = dists[i]
                    for k in range(n_vals):
                        bars[k].append([prob, dist.p[k]])
                
                means = []
                stds = []
                for bar_vals in bars:
                    m, va = _compute_weighted_mean_and_variance(bar_vals)
                    means.append(m)
                    stds.append(np.sqrt(va))
                
                plot.bar(np.arange(len(bars))-0.5+0.5*gap_size*n_target_vals+v*gap_size, means, width=gap_size, yerr=stds, align='center', ecolor='black', label=value_dict[target_id][1] + "=" + value_dict[target_id][2][v])
                plot.set_ylim(0, 1)
                plot.set_xticks(np.arange(len(dist.p)))
                if value_dict is not None:
                    plot.set_xticklabels([value_dict[dist.scope[0]][2][x] for x in range(len(dist.p))])
                
            elif param_type is Gaussian:
                
                bars = [[] for _ in range(precision_numeric)]
                for [prob, dists] in sorted_sub_pops:
                    dist = dists[i]
                    x_vals = np.linspace(value_dict[f_id][2][0], value_dict[f_id][2][1], precision_numeric, endpoint=True)
                    for k, x_val in enumerate(x_vals):
                        bars[k].append([prob, stats.norm.pdf(x_val, dist.mean, dist.stdev)])

                means = []
                stds = []
                for bar_vals in bars:
                    m, va = _compute_weighted_mean_and_variance(bar_vals)
                    means.append(m)
                    stds.append(np.sqrt(va))
                means = np.array(means)
                stds = np.array(stds)
                
                plot.plot(x_vals, means, label=value_dict[target_id][1] + "=" + value_dict[target_id][2][v])
                plot.fill_between(x_vals, means-stds, means+stds, alpha=0.5)

            else:
                raise Exception("Not implemented for other than categorical or gaussian")
                
    pad_col = 5
    if value_dict is None:
        feature_names = ["P(Feature " + str(x) + "|Feature " + str(target_id) + ")" for x in sorted(spn.scope)]
    else:
        feature_names = ["P(" + value_dict[x][1] + "|" + value_dict[target_id][1] + ")"  for x in sorted(spn.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    
    
    pad_row = 5
    info = ""
    for i, prob in enumerate(tot_prob_vals):
        info += value_dict[target_id][1] + "=" + value_dict[target_id][2][i] + " " + str(round(prob*100,4)) + "%\n"
    axes[0][0].annotate(info, xy=(0, 0.5), xytext=(-axes[0][0].yaxis.labelpad - pad_row, 0), xycoords=axes[0][0].yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
    axes[0][0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    fig.subplots_adjust(left=0.15)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



def visualize_sub_population_distributions_combined(spn, top=None, rang=None, value_dict=None, save_path=None, precision_numeric=50): 
    assert(value_dict is not None)

    pt_fids = fn.get_parametric_types_and_feature_ids(spn)
    
    ncols = len(pt_fids)
    nrows = 1
    
    figsize_x = ncols*3
    figsize_y = nrows*2
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)
        
    if rang is None:
        rang = np.array([None]*(max(spn.scope)+1))
    
    sub_pops = _get_sub_population_distributions(spn, rang=rang)
    sub_pops = [[prob, dists] for [prob, dists] in sub_pops if prob > 0]
    sub_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in sub_pops]
    sorted_sub_pops = sorted(sub_pops, key=lambda x: x[0], reverse=True)
    
    if top is not None:
        sorted_sub_pops = sorted_sub_pops[:top]
    
    for i, (f_id, param_type) in enumerate(pt_fids):
        

        plot = axes[0][i]
        if param_type is Categorical:
            n_vals = len(value_dict[f_id][2])
            bars = [[] for _ in range(n_vals)]
            for [prob, dists] in sorted_sub_pops:
                dist = dists[i]
                for k in range(n_vals):
                    bars[k].append([prob, dist.p[k]])
            
            means = []
            stds = []
            for bar_vals in bars:
                m, va = _compute_weighted_mean_and_variance(bar_vals)
                means.append(m)
                stds.append(np.sqrt(va))
            
            plot.bar(np.arange(len(bars)), means, yerr=stds, align='center', ecolor='black')
            plot.set_ylim(0, 1)
            plot.set_xticks(np.arange(len(dist.p)))
            if value_dict is not None:
                plot.set_xticklabels([value_dict[dist.scope[0]][2][x] for x in range(len(dist.p))])
            
        elif param_type is Gaussian:
            
            bars = [[] for _ in range(precision_numeric)]
            for [prob, dists] in sorted_sub_pops:
                dist = dists[i]
                x_vals = np.linspace(value_dict[f_id][2][0], value_dict[f_id][2][1], precision_numeric, endpoint=True)
                for k, x_val in enumerate(x_vals):
                    bars[k].append([prob, stats.norm.pdf(x_val, dist.mean, dist.stdev)])

            means = []
            stds = []
            for bar_vals in bars:
                m, va = _compute_weighted_mean_and_variance(bar_vals)
                means.append(m)
                stds.append(np.sqrt(va))
            means = np.array(means)
            stds = np.array(stds)
            
            plot.plot(x_vals, means)
            plot.fill_between(x_vals, means-stds, means+stds, alpha=0.5)

        else:
            raise Exception("Not implemented for other than categorical or gaussian")
                
    pad_col = 5
    if value_dict is None:
        feature_names = ["P(Feature " + str(x) + ")" for x in sorted(spn.scope)]
    else:
        feature_names = ["P(" + value_dict[x][1] + ")"  for x in sorted(spn.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
        
    plt.tight_layout()
    fig.subplots_adjust()
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)




'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''
        
        

def visualize_sub_population_lines(spn, top=None, rang=None, value_dict=None, save_path=None, precision_numeric=50):
    
    sub_pops = _get_sub_population_distributions(spn, rang=rang)
    sub_pops = [[prob, dists] for [prob, dists] in sub_pops if prob > 0]
    sub_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in sub_pops]
    sorted_sub_pops = sorted(sub_pops, key=lambda x: x[0], reverse=True)
    
    if top is not None:
        sorted_sub_pops = sorted_sub_pops[:top]
     
    lines = []
    for [prob, dists] in sorted_sub_pops:
        line = []
        for dist in dists: 
            assert(len(dist.scope) == 1)
            feature_id = dist.scope[0]

            if isinstance(dist, Categorical):
                i = np.argmax(dist.p)
                y_val = np.linspace(0,1,len(value_dict[feature_id][2]))[i]
                line.append(y_val)
            elif isinstance(dist, Gaussian):
                mean = dist.mean
                mi = value_dict[feature_id][2][0]
                ma = value_dict[feature_id][2][1]
                y_val = (mean-mi)/(ma-mi)
                line.append(y_val)
            else:
                raise Exception("Not implemented for other than categorical or gaussian")
        lines.append([prob, line])
    
    ncols = 1
    nrows = 1
    figsize_x = 16
    figsize_y = 6
    _, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False, sharey=True, sharex=True)
    
    plot = axes[0][0]
    for [prob, line] in lines:
        x_vals = []
        y_vals = []
        for i in range(len(line)-1):
            y_val = line[i]
            next_y_val = line[i+1]
            for r in np.arange(0, 1, 1/precision_numeric):
                x_vals.append(i+r)
                y_vals.append(y_val + (next_y_val - y_val)*r + np.random.normal()*0.025)
                 
        plot.plot(x_vals, y_vals, alpha=prob*5, linewidth=prob*100)
    plot.set_xticks(np.arange(len(spn.scope)))
    if value_dict is not None:
        plot.set_xticklabels([value_dict[scope][1] for scope in spn.scope])
    
    for j, feature_id in enumerate(spn.scope):
        
        if value_dict[feature_id][0] == "discrete":
            for i, y_val in enumerate(np.linspace(0,1,len(value_dict[feature_id][2]))):
                val_name = value_dict[feature_id][2][i]
                plot.text(j, y_val, val_name)
        elif value_dict[feature_id][0] == "numeric":
            mi = value_dict[feature_id][2][0]
            ma = value_dict[feature_id][2][1]
            for i, y_val in enumerate(np.linspace(0,1,5)):
                val_name = round(y_val*(ma-mi) + mi, 4)
                plot.text(j, y_val, val_name)
        else:
            raise Exception("Not implemented for other than discrete or numeric")
            
            
    plt.tight_layout()
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



def visualize_sub_population_lines_target(spn, target_id, top=None, rang=None, value_dict=None, save_path=None, precision_numeric=50):
    
    all_lines = []
    n_target_vals = fn.get_num_values_feature(spn, target_id)
    for v in range(n_target_vals):
        
        if rang is None:
            rang = np.array([None]*(max(spn.scope)+1))
        
        rang[target_id] = NominalRange([v])
        sub_pops = _get_sub_population_distributions(spn, rang=rang)
        sub_pops = [[prob, dists] for [prob, dists] in sub_pops if prob > 0]
        sub_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in sub_pops]
        sorted_sub_pops = sorted(sub_pops, key=lambda x: x[0], reverse=True)
        
        if top is not None:
            sorted_sub_pops = sorted_sub_pops[:top]
         
        lines = []
        for [prob, dists] in sorted_sub_pops:
            line = []
            target_prob=0
            for dist in dists: 
                assert(len(dist.scope) == 1)
                feature_id = dist.scope[0]
            
                if isinstance(dist, Categorical):
                    if feature_id == target_id:
                        i = v
                        target_prob = dist.p[v]
                    else:
                        i = np.argmax(dist.p)
                        
                    y_val = np.linspace(0,1,len(value_dict[feature_id][2]))[i]
                    line.append(y_val)
                elif isinstance(dist, Gaussian):
                    assert(feature_id != target_id)
                    mean = dist.mean
                    mi = value_dict[feature_id][2][0]
                    ma = value_dict[feature_id][2][1]
                    y_val = (mean-mi)/(ma-mi)
                    line.append(y_val)
                else:
                    raise Exception("Not implemented for other than categorical or gaussian")

            lines.append([prob, target_prob, line])
        
        all_lines.append(lines)
       
    ncols = 1
    nrows = len(all_lines)
    figsize_x = 16
    figsize_y = 6 * len(all_lines)
    _, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False, sharey=True, sharex=True)
    
    for i, lines in enumerate(all_lines):
    
        plot = axes[i][0]
        for [prob, target_prob, line] in lines:
            x_vals = []
            y_vals = []
            for i in range(len(line)-1):
                y_val = line[i]
                next_y_val = line[i+1]
                for r in np.arange(0, 1, 1/precision_numeric):
                    x_vals.append(i+r)
                    y_vals.append(y_val + (next_y_val - y_val)*r + np.random.normal()*0.025)
                     
            plot.plot(x_vals, y_vals, alpha=target_prob, linewidth=prob*100)
        plot.set_xticks(np.arange(len(spn.scope)))
        if value_dict is not None:
            plot.set_xticklabels([value_dict[scope][1] for scope in spn.scope])
        
        for j, feature_id in enumerate(spn.scope):
            if value_dict[feature_id][0] == "discrete":
                for i, y_val in enumerate(np.linspace(0,1,len(value_dict[feature_id][2]))):
                    val_name = value_dict[feature_id][2][i]
                    plot.text(j, y_val, val_name)
            elif value_dict[feature_id][0] == "numeric":
                mi = value_dict[feature_id][2][0]
                ma = value_dict[feature_id][2][1]
                for i, y_val in enumerate(np.linspace(0,1,5)):
                    val_name = round(y_val*(ma-mi) + mi, 4)
                    plot.text(j, y_val, val_name)
            else:
                raise Exception("Not implemented for other than discrete or numeric")
        
    plt.tight_layout()
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''



def visualize_sub_population_intrestingness(spn, top=None, rang=None, value_dict=None, save_path=None):
    
    sub_pops = _get_sub_population_distributions(spn, rang=rang)
    sub_pops = [[prob, dists] for [prob, dists] in sub_pops if prob > 0]
    sub_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in sub_pops]
    sorted_sub_pops = sorted(sub_pops, key=lambda x: x[0], reverse=True)
    
    if top is not None:
        sorted_sub_pops = sorted_sub_pops[:top]
    
    ncols = len(spn.scope)
    nrows = len(sorted_sub_pops)
    figsize_x = ncols*3
    figsize_y = nrows*2
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)#, sharey=True)#, sharex=True)
    
    intrest_pops = []
    for i, [prob1, dists1] in enumerate(sorted_sub_pops):
        intrest_scores = [[] for _ in range(len(dists1))]
        
        for j, [prob2, dists2] in enumerate(sorted_sub_pops):
            if i == j:
                continue
            for i in range(len(dists1)):
                score = _compare_dists(dists1[i], dists2[i], value_dict)
                intrest_scores[i].append(score)
        
        intrest_scores = np.array(intrest_scores)
        normalized_intrest_score = np.sum(np.sum(intrest_scores, axis=1)/len(sub_pops))/len(dists1)
        intrest_pops.append([normalized_intrest_score, dists1])
        
    
    
    intrest_pops = [[prob, dists] for [prob, dists] in intrest_pops if prob > 0]
    intrest_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in intrest_pops]
    sorted_intrest_pops = sorted(intrest_pops, key=lambda x: x[0], reverse=True)
    
    ncols = len(spn.scope)
    nrows = len(sorted_intrest_pops)
    figsize_x = ncols*3
    figsize_y = nrows*2
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)#, sharey=True)#, sharex=True)
    
    for i, [prob, dists] in enumerate(sorted_intrest_pops):
        for j, dist in enumerate(dists):
            plot = axes[i][j]
            if isinstance(dist, Categorical):
                plot.bar(np.arange(len(dist.p)), dist.p)
                plot.set_ylim(0, 1)
                plot.set_xticks(np.arange(len(dist.p)))
                if value_dict is not None:
                    plot.set_xticklabels([value_dict[dist.scope[0]][2][x] for x in range(len(dist.p))])
                
                if rang is not None and rang[dist.scope[0]] is not None:
                    selected_vals = [dist.p[i] for i in rang[dist.scope[0]].get_ranges()]
                    plot.bar(rang[dist.scope[0]].get_ranges(), selected_vals, color="red")
            
            elif isinstance(dist, Gaussian):
                x_vals = np.linspace(dist.mean-3*dist.stdev, dist.mean+3*dist.stdev, 30, endpoint=True)
                y_vals = []
                for x in x_vals:
                    y_vals.append(stats.norm.pdf(x, dist.mean, dist.stdev))
                
                plot.plot(x_vals, y_vals)
                if value_dict is not None:
                    plot.set_xlim(value_dict[dist.scope[0]][2][0], value_dict[dist.scope[0]][2][1])
            else:
                raise Exception("Not implemented for other than categorical or gaussian")
                       
    pad_col = 5
    if value_dict is None:
        feature_names = ["Feature " + str(x) for x in sorted(spn.scope)]
    else:
        feature_names = [value_dict[x][1] for x in sorted(spn.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    pad_row = 5
    for ax, row in zip(axes[:,0], [round(x,6) for [x,_] in sorted_intrest_pops]):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad_row, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
    plt.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    
    
    
    
        
        
        



def _compare_dists(dist1, dist2, value_dict, precision_numeric=50):
    assert(dist1.scope[0] == dist2.scope[0])
    
    if isinstance(dist1, Categorical):
        return _compute_difference_nominal(dist1.p, dist2.p)#_compute_kullback_leibner_divergence(dist1.p, dist2.p)
    elif isinstance(dist1, Gaussian):
        
        x_vals = np.linspace(value_dict[dist1.scope[0]][2][0], value_dict[dist1.scope[0]][2][1], precision_numeric, endpoint=True)
        y_vals1 = [] 
        y_vals2 = []
        for x in x_vals:
            y_vals1.append(stats.norm.pdf(x, dist1.mean, dist1.stdev))
            y_vals2.append(stats.norm.pdf(x, dist2.mean, dist2.stdev))        
        return _compute_difference_numeric(y_vals1, y_vals2)

    else:
        raise Exception("Not implemented for other than categorical or gaussian")



def _compute_difference_nominal(p_vals, q_vals):
    return np.sum([ np.abs(p_vals[i]-q_vals[i]) for i in range(len(p_vals))])/len(p_vals)

def _compute_difference_numeric(p_vals, q_vals):
    return np.sum([ np.abs(p_vals[i]-q_vals[i])/(np.max([p_vals[i], q_vals[i]])) for i in range(len(p_vals))])/len(p_vals)

#def _compute_kullback_leibner_divergence(p_vals, q_vals):
#    return np.sum([p_vals[i] * np.log(p_vals[i]/q_vals[i]) for i in range(len(p_vals))])



'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def feature_importance(spn, target_id, value_dict, precision_numeric=50):
    
    pt_fids = fn.get_parametric_types_and_feature_ids(spn)
    n_target_vals = len(value_dict[target_id][2])
    rang = None
    
    tot_prob_vals = []
    fis = [[] for _ in range(len(spn.scope))]
    for v in range(n_target_vals):
        
        if rang is None:
            rang = np.array([None]*(max(spn.scope)+1))
        
        rang[target_id] = NominalRange([v])
        sub_pops = _get_sub_population_distributions(spn, rang=rang)
        sub_pops = [[prob, dists] for [prob, dists] in sub_pops if prob > 0]
        sub_pops = [[prob, sorted(dists, key=lambda x: x.scope[0])] for [prob, dists] in sub_pops]
        sorted_sub_pops = sorted(sub_pops, key=lambda x: x[0], reverse=True)
    
        tot_prob_vals.append(round(sum([prob for prob, dists in sorted_sub_pops]),6))
        
        for i, (f_id, param_type) in enumerate(pt_fids):
            
            if f_id == target_id:
                continue

            if param_type is Categorical:
                n_vals = len(value_dict[f_id][2])
                bars = [[] for _ in range(n_vals)]
                for [prob, dists] in sorted_sub_pops:
                    dist = dists[i]
                    for k in range(n_vals):
                        bars[k].append([prob, dist.p[k]])
                
                means = []
                stds = []
                for bar_vals in bars:
                    m, va = _compute_weighted_mean_and_variance(bar_vals)
                    means.append(m)
                    stds.append(np.sqrt(va))
                
                fis[i].append(means)
                
                
            elif param_type is Gaussian:
                
                bars = [[] for _ in range(precision_numeric)]
                for [prob, dists] in sorted_sub_pops:
                    dist = dists[i]
                    x_vals = np.linspace(value_dict[f_id][2][0], value_dict[f_id][2][1], precision_numeric, endpoint=True)
                    for k, x_val in enumerate(x_vals):
                        bars[k].append([prob, stats.norm.pdf(x_val, dist.mean, dist.stdev)])

                means = []
                stds = []
                for bar_vals in bars:
                    m, va = _compute_weighted_mean_and_variance(bar_vals)
                    means.append(m)
                    stds.append(np.sqrt(va))
                means = np.array(means)
                stds = np.array(stds)
                
                '''
                TODOOO
                '''
                raise Exception("Not implemented")
                


            else:
                raise Exception("Not implemented for other than categorical or gaussian")
    
    
    importance = [0 for _ in range(len(spn.scope))]
    remove_index = -1
    for i in range(len(fis)):
        if len(fis[i]) == 0:
            remove_index = i
            continue
        
        diffs = 0
        for j, dist1 in enumerate(fis[i]):
            for k, dist2 in enumerate(fis[i]):
                if j == k:
                    continue
                for l in range(len(dist1)):
                    diffs += np.abs(dist1[l]-dist2[l])           
        importance[i] = diffs
                

    new_importance = []
    for i in range(len(importance)):
        if i == remove_index:
            continue
        new_importance.append(importance[i])
            
    return new_importance
    




'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''
    
    
def demo_visualze_subpopulation_distributions():
    
    #Survived |   Pclass |   Sex |   Age |   SibSp |   Parch |     Fare |   Embarked 
    #data, parametric_types = real_data.get_titanic()
    #learn_SPN.create_parametric_spns(data, parametric_types, [0.3, 0.1, 0.05, 0.01], [0.1, 0.05, 0.01], folder="titanic")
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.1)
    spn, _ = io.load(ident, "titanic", loc)
    value_dict = real_data.get_titanic_value_dict()
    spn = fn.marg(spn, keep=[0,1,2,3])#,4,5,6,7])
    
    
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/dist_combined1.pdf"
    visualize_sub_population_distributions_combined(spn, top=None, rang=rang, value_dict=value_dict, save_path=save_path)
    
    exit()
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/dist_demo_target_compact.pdf"
    visualize_sub_population_distributions_target_compact(spn, 0, top=None, rang=rang, value_dict=value_dict, save_path=save_path)
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/dist_demo_target.pdf"
    visualize_sub_population_distributions_target(spn, 0, top=None, rang=rang, value_dict=value_dict, save_path=save_path)
    
   
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/dist_demo1.pdf"
    visualize_sub_population_distributions(spn, top=3, rang=rang, value_dict=value_dict, save_path=save_path)
    
    rang = [NominalRange([1])] + [None]*7
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/dist_demo2.pdf"
    visualize_sub_population_distributions(spn, top=10, rang=rang, value_dict=value_dict, save_path=save_path)
    
    rang = [NominalRange([1])] + [None] + [NominalRange([0])] + [None]*5
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/dist_demo3.pdf"
    visualize_sub_population_distributions(spn, top=10, rang=rang, value_dict=value_dict, save_path=save_path)



def demo_visualze_subpopulation_lines():
    
    #Survived |   Pclass |   Sex |   Age |   SibSp |   Parch |     Fare |   Embarked 
    #data, parametric_types = real_data.get_titanic()
    #learn_SPN.create_parametric_spns(data, parametric_types, [0.3], [0.01], folder="titanic")
    
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.1)
    spn, _ = io.load(ident, "titanic", loc)
    value_dict = real_data.get_titanic_value_dict()
    spn = fn.marg(spn, keep=[0,1,2,4,5,7])
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/line_demo1.pdf"
    visualize_sub_population_lines(spn, top=None, rang=rang, value_dict=value_dict, save_path=save_path)
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/line_demo2.pdf"
    visualize_sub_population_lines_target(spn, target_id=0, top=None, rang=rang, value_dict=value_dict, save_path=save_path)
    
    rang = [None]*2 + [NominalRange([0])] + [None]*5
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/line_demo3.pdf"
    visualize_sub_population_lines_target(spn, target_id=0, top=None, rang=rang, value_dict=value_dict, save_path=save_path)
    
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.1)
    spn, _ = io.load(ident, "titanic", loc)
    value_dict = real_data.get_titanic_value_dict()
    spn = fn.marg(spn, keep=[0,1])#,2,3,4,5,6,7])
    
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/line_demo4.pdf"
    visualize_sub_population_lines(spn, top=None, rang=rang, value_dict=value_dict, save_path=save_path)
    
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/line_demo5.pdf"
    visualize_sub_population_lines_target(spn, target_id=0, top=None, rang=rang, value_dict=value_dict, save_path=save_path)
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/line_demo6.pdf"
    visualize_sub_population_lines_target(spn, target_id=0, top=3, rang=rang, value_dict=value_dict, save_path=save_path)



def demo_visualze_subpopulation_intrestingness():
    
    #Survived |   Pclass |   Sex |   Age |   SibSp |   Parch |     Fare |   Embarked 
    #data, parametric_types = real_data.get_titanic()
    #learn_SPN.create_parametric_spns(data, parametric_types, [0.3, 0.1, 0.05, 0.01], [0.1, 0.05, 0.01], folder="titanic")
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.1)
    spn, _ = io.load(ident, "titanic", loc)
    value_dict = real_data.get_titanic_value_dict()
    spn = fn.marg(spn, keep=[0,1,2,4,5,7])
    
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/subpopulations/intrest_demo_1.pdf"
    visualize_sub_population_intrestingness(spn, top=None, rang=rang, value_dict=value_dict, save_path=save_path)



if __name__ == '__main__':
    
    #import logging
    #logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    
    #demo_visualze_subpopulation_distributions()
    #demo_visualze_subpopulation_lines()
    demo_visualze_subpopulation_intrestingness()
    
    
    
    
    