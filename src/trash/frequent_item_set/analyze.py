'''
Created on 26.08.2019

@author: Moritz
'''

from util import io
import numpy as np
import matplotlib.pyplot as plt

def __compare_itemsets(set_gt, set_pred):
    
    same_sets = []
    missing_sets = []
    for (s1, c1) in set_gt:
        
        found = False
        for (s2, c2) in set_pred:
            if __same_conds(c1, c2):
                same_sets += [[c1, s1, s2]]
                found = True
                break;
        if not found:
            missing_sets.append([s1, c1])
    
    different_sets = []   
    for (s2, c2) in set_pred:
        found = False
        for (s1, c1) in set_gt:
            if __same_conds(c1, c2):
                found = True
                break;
        if not found:
            different_sets.append([s2, c2])
        
    return same_sets, missing_sets, different_sets
    
    
    
    
def __same_conds(conds1, conds2):
    
    conds1 = sorted(conds1, key=lambda x: (x[0], x[1]))
    conds2 = sorted(conds2, key=lambda x: (x[0], x[1]))
    
    if len(conds1) != len(conds2):
        return False
    for i in range(len(conds1)):
        if conds1[i][0] == conds2[i][0] and conds1[i][1] == conds2[i][1]:
            continue
        else:
            return False
    return True   




def visualize_stats(dataset_name, min_supports, gt_method="apriori1", other_methods=["fpgrowth2",
                                                                                      "spn1_rdc=0.3_mis=0.1",
                                                                                      "spn1_rdc=0.3_mis=0.01",
                                                                                      "spn1_rdc=0.2_mis=0.1",
                                                                                      "spn1_rdc=0.2_mis=0.01",
                                                                                      "spn1_rdc=0.1_mis=0.1",
                                                                                      "spn1_rdc=0.1_mis=0.01"]):
    
    folder = "freq_sets/" + dataset_name
    vis_dict = {"item_sets" : {}, "n_items" : {}, "exc_times" : {}, "same_items" : {}, "same_errors" : {}, "missing_items" : {}, "missing_probs" : {}, "new_items" : {}, "new_probs" : {}}
    
    for method in [gt_method] + other_methods:    
        vis_dict["item_sets"][method] = []
        vis_dict["n_items"][method] = []
        vis_dict["exc_times"][method] = []
        vis_dict["same_items"][method] = []
        vis_dict["missing_items"][method] = []
        vis_dict["new_items"][method] = []
        
        vis_dict["same_errors"][method] = []
        vis_dict["new_probs"][method] = []
        vis_dict["missing_probs"][method] = []
        
        for i, min_sup in enumerate(min_supports):
            method_itemsets, method_exc_time = io.load(method + "_minSup=" + str(min_sup), folder, loc="_results")
            vis_dict["item_sets"][method].append(method_itemsets)
            vis_dict["n_items"][method].append(len(method_itemsets))
            vis_dict["exc_times"][method].append(method_exc_time)
        
            same_sets, missing_sets, different_sets = __compare_itemsets(vis_dict["item_sets"][gt_method][i], method_itemsets)
            vis_dict["same_items"][method].append(len(same_sets))
            vis_dict["missing_items"][method].append(len(missing_sets))
            vis_dict["new_items"][method].append(len(different_sets))
            
            vis_dict["same_errors"][method].append([s1-s2 for (_, s1, s2) in same_sets])
            vis_dict["new_probs"][method].append([sup for (sup, _) in different_sets])
            vis_dict["missing_probs"][method].append([sup for (sup, _) in missing_sets])
            
    
    plot_names = list(vis_dict.keys())
    plot_names.remove("item_sets") 
    
    ncols = 1
    nrows = len(plot_names)
    figsize_x = 12
    figsize_y = 20
    _, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False, sharex=False)
    
    r = np.arange(len(min_supports))
    n_methods = 1 + len(other_methods)
    barWidth = 1/(1 + n_methods)
        
        
    for j, plot_name in enumerate(plot_names):
        
        if plot_name in ["same_errors", "new_probs", "missing_probs"]:
            
            for i, (name, vals) in enumerate(vis_dict[plot_name].items()):
                #positions = np.arange(len(method_box_plots[i]))+ (i/(2 + len(method_box_plots))) - (len(method_box_plots)/2)/len(method_box_plots) + 1.5/len(method_box_plots)
                axes[j][0].boxplot(vals, positions=r+i*barWidth, widths=barWidth)
                
            axes[j][0].set_title(plot_name)
            axes[j][0].set_xlabel('min support')
            axes[j][0].set_xlim(-0.25, len(min_supports))
            axes[j][0].set_xticks(np.arange(0, len(min_supports)))
            axes[j][0].set_xticklabels([str(min_sup) for min_sup in min_supports])
            axes[j][0].legend()
        else:
        
            for i, (name, vals) in enumerate(vis_dict[plot_name].items()):
                axes[j][0].bar(r+i*barWidth, vals, width=barWidth,label=name)
            axes[j][0].set_title(plot_name)
            axes[j][0].set_xlabel('min support')
            
            if plot_name == "exc_times":
                axes[j][0].set_yscale('log')
            
            axes[j][0].set_xlim(-0.25, len(min_supports))
            axes[j][0].set_xticks(np.arange(0, len(min_supports)))
            axes[j][0].set_xticklabels([str(min_sup) for min_sup in min_supports])
            axes[j][0].legend()
    
    plt.tight_layout()
    
    plt.savefig("stats.pdf")
    
    plt.show()
    
    

if __name__ == '__main__':
    
    dataset_name = "T10I4D_100"
    min_supports = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    visualize_stats(dataset_name, min_supports)
    
    
    