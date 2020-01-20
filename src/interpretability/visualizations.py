'''
Created on 11.10.2019

@author: Moritz
'''
import numpy as np
import matplotlib.pyplot as plt

from simple_spn import functions as fn
from util import viz_helper
from spn.experiments.AQP.Ranges import NominalRange, NumericRange



def visualize_sub_populations(spn, value_dict=None, top=None, rang=None, numeric_prec=50, save_path=None):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    if rang is not None: spn = fn.marg_rang(spn, rang)
    sub_pops = fn.get_sub_populations(spn, sort=True, top=top)
    
    ncols = len(spn.scope)
    nrows = len(sub_pops)
    figsize_x = ncols*3
    figsize_y = nrows*2
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)
    
    for i, [_, dists] in enumerate(sub_pops):
        for j, dist in enumerate(dists):
            f_id = dist.scope[0]
            if value_dict[f_id][0] == "discrete":
                
                val_pairs = sorted(value_dict[f_id][2].items(), key=lambda x: x[0])
                y_vals = fn.evaluate_discrete_leaf(dist, f_vals=[x[0] for x in val_pairs])
                viz_helper.bar_plot(axes[i][j], y_vals, x_tick_labels= [x[1] for x in val_pairs], y_label="probability", ylim=[0,1])
                
            elif value_dict[f_id][0] == "numeric":
                
                x_vals = np.linspace(value_dict[f_id][2][0], value_dict[f_id][2][1], num=numeric_prec)
                y_vals = fn.evaluate_numeric_density_leaf(dist, x_vals)
                viz_helper.line_plot(axes[i][j],x_vals,  y_vals, y_label="density")
                
            else:
                raise Exception("Unknown attribute-type: " + str(value_dict[dist.scope[0]]))
    
    pad_col = 5
    if value_dict is None:
        feature_names = ["Feature " + str(x) for x in sorted(spn.scope)]
    else:
        feature_names = [value_dict[x][1] for x in sorted(spn.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    pad_row = 5
    for ax, row in zip(axes[:,0], [round(x,6) for [x,_] in sub_pops]):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad_row, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
    plt.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



def visualize_overall_distribution(spn, value_dict=None, rang=None, numeric_prec=50, save_path=None):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    if rang is not None: _, spn = fn.marg_rang(spn, rang)
    overall_population = fn.get_overall_population(spn, value_dict=value_dict, numeric_prec=numeric_prec)
    
    ncols = len(spn.scope)
    nrows = 1
    figsize_x = ncols*3
    figsize_y = nrows*3
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)
    
    for i, f_id in enumerate(sorted(list(overall_population))):
        dist = overall_population[f_id]
        
        if dist["feature_type"] == "discrete":
            viz_helper.bar_plot(axes[0][i], dist["y_means"], dist["x_labels"], y_err=np.sqrt(dist["y_vars"]), y_label="probability", ylim=[0,1])
        
        elif dist["feature_type"] == "numeric":
            viz_helper.line_plot(axes[0][i], dist["x_vals"], dist["y_means"], y_errs=np.sqrt(dist["y_vars"]),y_label="density")
        else:
            raise Exception("Unknown attribute-type: " + str(value_dict[dist.scope[0]]))
        
    pad_col = 5
    feature_names = [value_dict[x][1] for x in sorted(spn.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    
    

def visualize_target_based_overall_distribution_single(spn, target_id, value_dict=None, rang=None, numeric_prec=50, save_path=None):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    if rang is not None : assert(rang[target_id] is None)
    if rang is not None: _, spn = fn.marg_rang(spn, rang)
    
    n_vals = len(value_dict[target_id][2])
    
    ncols = len(spn.scope)-1
    nrows = n_vals
    figsize_x = ncols*3
    figsize_y = nrows*2
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)
    
    ps = []
    for v in range(n_vals):
        tmp_rang = [None] * (np.max(spn.scope)+1)
        tmp_rang[target_id] = NominalRange([v])
        
        p, spn1 = fn.marg_rang(spn, tmp_rang)
        ps.append(p)
        overall_population = fn.get_overall_population(spn1, value_dict=value_dict, numeric_prec=numeric_prec)
        
        for i, f_id in enumerate(sorted(spn1.scope)):
            dist = overall_population[f_id]
            
            if dist["feature_type"] == "discrete":
                viz_helper.bar_plot(axes[v][i], dist["y_means"], dist["x_labels"], y_err=np.sqrt(dist["y_vars"]), y_label="probability", ylim=[0,1])
            
            elif dist["feature_type"] == "numeric":
                viz_helper.line_plot(axes[v][i], dist["x_vals"], dist["y_means"], y_errs=np.sqrt(dist["y_vars"]), y_label="density")
            else:
                raise Exception("Unknown attribute-type: " + str(value_dict[dist.scope[0]]))


    pad_col = 5
    feature_names = [value_dict[x][1] for x in sorted(spn1.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    
    pad_row = 5
    for i, p in enumerate(ps):
        axes[i][0].annotate(str(round(p*100,4)) + "%\n" + value_dict[target_id][1] + "=" + value_dict[target_id][2][i], xy=(0, 0.5), xytext=(-axes[i][0].yaxis.labelpad - pad_row, 0), xycoords=axes[i][0].yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

    plt.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.9)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



        
def visualize_target_based_overall_distribution_compact(spn, target_id, value_dict=None, rang=None, numeric_prec=50, save_path=None):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    if rang is not None : assert(rang[target_id] is None)
    if rang is not None: _, spn = fn.marg_rang(spn, rang)
    
    n_vals = len(value_dict[target_id][2])
    
    ncols = len(spn.scope)-1
    nrows = 1
    figsize_x = ncols*3
    figsize_y = nrows*3
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)
    
    ps = []
    plot_data = {f_id : [] for f_id in spn.scope if f_id != target_id}
    for v in range(n_vals):
        tmp_rang = [None] * (np.max(spn.scope)+1)
        tmp_rang[target_id] = NominalRange([v])
        
        p, spn1 = fn.marg_rang(spn, tmp_rang)
        ps.append(p)
        overall_population = fn.get_overall_population(spn1, value_dict=value_dict, numeric_prec=numeric_prec)
        for f_id in spn1.scope:
            plot_data[f_id].append(overall_population[f_id])
            
    for i, f_id in enumerate(plot_data):
        
        if value_dict[f_id][0] == "discrete":
            y_means = []
            y_errs = []
            legend_labels = []
            for j, dist in enumerate(plot_data[f_id]):
                y_means.append(dist["y_means"])
                y_errs.append(dist["y_vars"])
                legend_labels.append(str(value_dict[target_id][1]) + "=" + str(value_dict[target_id][2][j]))
            viz_helper.multiple_bar_plot(axes[0][i], y_means, dist["x_labels"], y_errs=np.sqrt(y_errs), legend_labels=legend_labels, y_label="probability", ylim=[0,1])

        elif value_dict[f_id][0] == "numeric":
            for j, dist in enumerate(plot_data[f_id]):
                viz_helper.line_plot(axes[0][i], dist["x_vals"], dist["y_means"], y_errs=np.sqrt(dist["y_vars"]), label=str(value_dict[target_id][1]) + "=" + str(value_dict[target_id][2][j]), y_label="density")
        else:
            raise Exception("Unknown attribute-type: " + str(value_dict[dist.scope[0]]))

    pad_col = 5
    feature_names = [value_dict[x][1] for x in sorted(spn1.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    
    
    pad_row = 5
    info = ""
    for i, prob in enumerate(ps):
        info += str(value_dict[target_id][1]) + "=" + str(value_dict[target_id][2][i]) + " " + str(round(prob*100,4)) + "%\n"
    axes[0][0].annotate(info, xy=(0, 0.5), xytext=(-axes[0][0].yaxis.labelpad - pad_row, 0), xycoords=axes[0][0].yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
    axes[0][0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.9)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)




def visualize_target_based_conds_overall_distribution_compact(spn, target_conds, value_dict=None, rang=None, target_names=None, numeric_prec=50, save_path=None):
    '''
    TODOOOO
    '''
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    target_ids = set([cond for conds in target_conds for cond in conds])
    if rang is not None :
        for conds in target_conds:
            for target_id in conds:
                assert(rang[target_id] is None)
    if rang is not None: _, spn = fn.marg_rang(spn, rang)
    
    n_vals = len(target_conds)
    
    ncols = len(spn.scope)-1
    nrows = 1
    figsize_x = ncols*3
    figsize_y = nrows*3
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)
    
    ps = []
    plot_data = {f_id : [] for f_id in spn.scope if f_id not in target_ids}
    for v in range(n_vals):
        tmp_rang = [None] * (np.max(spn.scope)+1)
        for target_id, cond in target_conds[v].items(): tmp_rang[target_id] = cond
        
        p, spn1 = fn.marg_rang(spn, tmp_rang)
        ps.append(p)
        overall_population = fn.get_overall_population(spn1, value_dict=value_dict, numeric_prec=numeric_prec)
        for f_id in spn1.scope:
            plot_data[f_id].append(overall_population[f_id])
            
    for i, f_id in enumerate(plot_data):
        
        if value_dict[f_id][0] == "discrete":
            y_means = []
            y_errs = []
            for j, dist in enumerate(plot_data[f_id]):
                y_means.append(dist["y_means"])
                y_errs.append(dist["y_vars"])
            #viz_helper.multiple_bar_plot(axes[0][i], y_means, dist["x_labels"], y_errs=np.sqrt(y_errs), legend_labels=target_names, y_label="probability", ylim=[0,1])
            viz_helper.multiple_bar_plot(axes[0][i], y_means, dist["x_labels"], legend_labels=target_names, y_label="probability", ylim=[0,1])

        elif value_dict[f_id][0] == "numeric":
            for j, dist in enumerate(plot_data[f_id]):
                #viz_helper.line_plot(axes[0][i], dist["x_vals"], dist["y_means"], y_errs=np.sqrt(dist["y_vars"]), label=target_names[j], y_label="density")
                viz_helper.line_plot(axes[0][i], dist["x_vals"], dist["y_means"], label=target_names[j], y_label="density")
        else:
            raise Exception("Unknown attribute-type: " + str(value_dict[dist.scope[0]]))

    pad_col = 5
    feature_names = [value_dict[x][1] for x in sorted(spn1.scope)]
    for ax, col in zip(axes[0], feature_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad_col), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    
    
    #pad_row = 5
    #info = ""
    #for i, prob in enumerate(ps):
    #    info += str(value_dict[target_id][1]) + "=" + str(value_dict[target_id][2][i]) + " " + str(round(prob*100,4)) + "%\n"
    #axes[0][0].annotate(info, xy=(0, 0.5), xytext=(-axes[0][0].yaxis.labelpad - pad_row, 0), xycoords=axes[0][0].yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
    axes[0][0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    #fig.subplots_adjust(left=0.15, top=0.9)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)






        

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''
    
        
        
def visualize_expected_sub_populations(spn, value_dict=None, top=None, rang=None, numeric_prec=10, save_path=None):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    if rang is not None: spn = fn.marg_rang(spn, rang)
    sub_pops = fn.get_sub_populations(spn, sort=True, top=top)
    
    fig, axes = plt.subplots(1, 1, figsize=(16,6), squeeze=False)

    lines = []
    for [prob, dists] in sub_pops:
        line = []
        for dist in dists: 
            f_id = dist.scope[0]

            if value_dict[f_id][0] == "discrete":
                rang = [None] * (np.max(spn.scope)+1)
                expect = fn.expect(dist, f_id, rang)
                y_val = np.linspace(0,1,len(value_dict[f_id][2]))[int(expect)]
                line.append(y_val)
                
            elif value_dict[f_id][0] == "numeric":
                rang = [None] * (np.max(spn.scope)+1)
                expect = fn.expect(dist, f_id, rang)
                
                mi = value_dict[f_id][2][0]
                ma = value_dict[f_id][2][1]
                y_val = (expect-mi)/(ma-mi)
                line.append(y_val)
            else:
                raise Exception("Unknown attribute-type: " + str(value_dict[dist.scope[0]]))
            
        lines.append([prob, line])
    
    plot = axes[0][0]
    plot.set_yticklabels([])
    for [prob, line] in lines:
        x_vals = []
        y_vals = []
        for i in range(len(line)-1):
            y_val = line[i]
            next_y_val = line[i+1]
            for r in np.linspace(0, 1, numeric_prec):
                x_vals.append(i+r)
                y_vals.append(y_val + (next_y_val - y_val)*r + np.random.normal()*0.025)
                 
        plot.plot(x_vals, y_vals, linewidth=prob*100)
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
        


def visualized_target_based_expected_sub_populations(spn, target_id, value_dict=None, top=None, rang=None, numeric_prec=10, save_path=None):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    if rang is not None: spn = fn.marg_rang(spn, rang)
    
    n_vals = len(value_dict[target_id][2])
    
    ps = []
    all_lines = []
    for v in range(n_vals):
        
        
        tmp_rang = [None] * (np.max(spn.scope)+1)
        tmp_rang[target_id] = NominalRange([v])
        p, spn1 = fn.marg_rang(spn, tmp_rang)
        ps.append(p)
        sub_pops = fn.get_sub_populations(spn1, sort=True, top=top)
        sub_pops = [[p*p1, dists] for p1, dists in sub_pops]
        
        lines = []
        for [p, dists] in sub_pops:
            line = []
            for dist in dists: 
                f_id = dist.scope[0]
    
                if value_dict[f_id][0] == "discrete":
                    rang = [None] * (np.max(spn.scope)+1)
                    expect = fn.expect(dist, f_id, rang)
                    y_val = np.linspace(0,1,len(value_dict[f_id][2]))[int(expect)]
                    line.append(y_val)
                    
                elif value_dict[f_id][0] == "numeric":
                    rang = [None] * (np.max(spn.scope)+1)
                    expect = fn.expect(dist, f_id, rang)
                    
                    mi = value_dict[f_id][2][0]
                    ma = value_dict[f_id][2][1]
                    y_val = (expect-mi)/(ma-mi)
                    line.append(y_val)
                else:
                    raise Exception("Unknown attribute-type: " + str(value_dict[dist.scope[0]]))
                
            lines.append([p, line])
        all_lines.append(lines)
    
    fig, axes = plt.subplots(n_vals, 1, figsize=(16,6*n_vals), squeeze=False)
    for i, lines in enumerate(all_lines):
    
        plot = axes[i][0]
        plot.set_yticklabels([])
        for [p, line] in lines:
            x_vals = []
            y_vals = []
            for i in range(len(line)-1):
                y_val = line[i]
                next_y_val = line[i+1]

                for r in np.linspace(0, 1, numeric_prec):
                    x_vals.append(i+r)
                    y_vals.append(y_val + (next_y_val - y_val)*r + np.random.normal()*0.025)
                     
            plot.plot(x_vals, y_vals, linewidth=p*100)
        
        x_feature_ids = sorted(list(set(spn.scope)-set([target_id])))
        plot.set_xticks(np.arange(len(x_feature_ids)))
        if value_dict is not None:
            plot.set_xticklabels([value_dict[scope][1] for scope in x_feature_ids])

        for j, feature_id in enumerate(x_feature_ids):
            
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
            
    
    pad_row = 5
    for i, (ax, p) in enumerate(zip(axes[:,0], ps)):
        info = value_dict[target_id][1] + "=" + value_dict[target_id][2][i] + " " + str(round(p*100,4)) + "%\n"
        ax.annotate(info, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad_row, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
    plt.tight_layout()
    fig.subplots_adjust(left=0.15)
       
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''
        


def visualize_likeliness_heatmap(spn, target_id_x, target_id_y, value_dict=None, rang=None, numeric_intervals=10, save_path=None):
    
    if value_dict is None : value_dict = fn.generate_adhoc_value_dict(spn)
    if rang is not None: spn = fn.marg_rang(spn, rang)
    assert(target_id_x in spn.scope and target_id_y in spn.scope)
    
    _, axes = plt.subplots(1, 1, figsize=(10,5), squeeze=False)
    ax = axes[0][0]
    
    
    x_conds, x_labels = _generate_conds(target_id_x, value_dict, numeric_intervals)
    y_conds, y_labels = _generate_conds(target_id_y, value_dict, numeric_intervals)
    
    ranges = []
    for y_cond in y_conds:
        for x_cond in x_conds:
            r = [None] * (np.max(spn.scope)+1)
            r[target_id_x] = x_cond
            r[target_id_y] = y_cond
            ranges.append(r)
    
    
    data = fn.probs(spn, ranges).reshape((len(y_conds),len(x_conds)))
    viz_helper.heatmap_plot(ax, data, x_labels, y_labels, x_label=value_dict[target_id_x][1], y_label=value_dict[target_id_y][1])
    
    plt.tight_layout()
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



def _generate_conds(target_id, value_dict, numeric_intervals=10):
    conds = []
    labels = []
    if value_dict[target_id][0] == "discrete":
        for val in sorted(value_dict[target_id][2]):
            conds.append(NominalRange([val]))
            labels.append(value_dict[target_id][2][val])
    elif value_dict[target_id][0] == "numeric":
        val_space = np.linspace(value_dict[target_id][2][0], value_dict[target_id][2][1], numeric_intervals+1)
        for interval in zip(val_space[1:], val_space[:-1]):
            conds.append(NumericRange([list(interval)]))
            labels.append(str(list(interval)))
    else:
        raise Exception("Not implemented for other than discrete or numeric ...: " + str(value_dict[target_id][0]))
    return conds, labels



if __name__ == '__main__':
    
    
    from simple_spn.example import example_spns
    from util import io
    
    
    spn = example_spns.get_gender_spn()
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.1)
    spn, value_dict, _ = io.load(ident, "titanic", loc)
    spn = fn.marg(spn, keep=[0,1,2,3,6])
    
    #visualize_sub_populations(spn)
    #visualize_overall_distribution(spn, value_dict=value_dict, save_path="overall_visualization.pdf")
    #visualize_target_based_overall_distribution_single(spn, 0, value_dict=value_dict, save_path="overall_visualization_target_based.pdf")
    #visualize_target_based_overall_distribution_compact(spn, 0, value_dict=value_dict, save_path="overall_visualization_target_based_compact.pdf")
    visualize_expected_sub_populations(spn, value_dict=value_dict, save_path="expectation_line_plot.pdf")
    visualized_target_based_expected_sub_populations(spn, 0, value_dict=value_dict, save_path="target_based_expectation_line_plot.pdf")
    pass
















