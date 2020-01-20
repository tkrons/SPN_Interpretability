'''
Created on 28.08.2019

@author: Moritz
'''

from spn.experiments.AQP.Ranges import NominalRange, NumericRange

import os
import numpy as np
import matplotlib.pyplot as plt

from util import io
from simple_spn import learn_SPN
from simple_spn import functions as fn
from data import real_data



def visualize_density(spn, value_dict, rang=None, n_steps=50, max_density=None, save_path=None):
    
    #Only select numeric features
    selected_features = []
    for feature_id in spn.scope:
        if value_dict[feature_id][0] == "numeric":
            selected_features.append(feature_id)
    
    #Create ranges
    if rang is None:
        rang = np.array([None]*(max(spn.scope)+1))
    
    ranges = []
    for i, feature_id in enumerate(selected_features):
        for x_val in np.linspace(value_dict[feature_id][2][0], value_dict[feature_id][2][1], num=n_steps):
            n_rang = rang.copy()
            n_rang[feature_id] = NumericRange([[x_val]])
            ranges.append(n_rang)
    
    #Evaluate densities
    res = fn.probs(spn, np.array(ranges))
    
    #Visualize
    ncols = 1
    nrows = len(selected_features)
    figsize_x = 16
    figsize_y = 6 * len(selected_features)
    _, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False, sharey=True, sharex=False)
    
    for i, feature_id in enumerate(selected_features):
        plot = axes[i][0]
        
        x_vals = np.linspace(value_dict[feature_id][2][0], value_dict[feature_id][2][1], num=n_steps)
        y_vals = res[n_steps*i:n_steps*i+n_steps]
        plot.plot(x_vals, y_vals)
    
        if max_density is not None:
            plot.set_ylim(0,max_density) 
        plot.set_title(value_dict[feature_id][1])
        
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        


def visualize_density_target(spn, target_id, value_dict, rang=None, n_steps=50, max_density=None, save_path=None):
    
    #Only select numeric features
    selected_features = []
    for feature_id in spn.scope:
        if value_dict[feature_id][0] == "numeric":
            selected_features.append(feature_id)
    
    #Create ranges
    if rang is None:
        rang = np.array([None]*(max(spn.scope)+1))

    results = []
    assert(value_dict[target_id][0] == "discrete")
    for v in value_dict[target_id][2]:
        rang[target_id] = NominalRange([v])
        
        ranges = []
        for i, feature_id in enumerate(selected_features):
            for x_val in np.linspace(value_dict[feature_id][2][0], value_dict[feature_id][2][1], num=n_steps):
                n_rang = rang.copy()
                n_rang[feature_id] = NumericRange([[x_val]])
                ranges.append(n_rang)
                
        results.append(fn.probs(spn, np.array(ranges)))
    
    #Visualize
    ncols = len(results)
    nrows = len(selected_features)
    figsize_x = 16
    figsize_y = 6 * len(selected_features)
    _, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False, sharey=True, sharex=False)
    
    for j, res in enumerate(results):
        for i, feature_id in enumerate(selected_features):
            plot = axes[i][j]
            
            x_vals = np.linspace(value_dict[feature_id][2][0], value_dict[feature_id][2][1], num=n_steps)
            y_vals = res[n_steps*i:n_steps*i+n_steps]
            plot.plot(x_vals, y_vals)
        
            if max_density is not None:
                plot.set_ylim(0,max_density)  
            plot.set_title(value_dict[feature_id][1] + " - " + value_dict[target_id][1] + "=" + value_dict[target_id][2][j])
        
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    





def visualize_density_2d(spn, value_dict=None, save_path=None):
    
    '''
    TODOOOOOOOOOOOOOOOOOOO
    '''
    
    pass




def demo_visualize_density():
    #data, parametric_types = real_data.get_p_value_dataset()
    #learn_SPN.create_parametric_spns(data, parametric_types, [0.3], [0.01], folder="p_value_test")
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.01)
    spn, _ = io.load(ident, "p_value_test", loc)
    value_dict = real_data.get_p_value_test_value_dict()
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/blackbox/density1.pdf"
    visualize_density(spn, value_dict, rang=rang, max_density=10, save_path=save_path)
    
    rang = [None]*5 + [NominalRange([0])]
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/blackbox/density2.pdf"
    visualize_density(spn, value_dict, rang=rang, max_density=10, save_path=save_path)
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/blackbox/density3.pdf"
    visualize_density_target(spn, 5, value_dict, rang=rang, max_density=10, save_path=save_path)
    
    
    
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.01)
    spn, _ = io.load(ident, "titanic", loc)
    value_dict = real_data.get_titanic_value_dict()
    
    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/blackbox/density5.pdf"
    visualize_density(spn, value_dict, max_density=0.1, save_path=save_path)

    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/blackbox/density6.pdf"
    visualize_density_target(spn, 0, value_dict, max_density=0.1, save_path=save_path)

    rang = None
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/blackbox/density7.pdf"
    visualize_density_target(spn, 2, value_dict, max_density=0.1, save_path=save_path)

    rang = [None]*2 + [NominalRange([0])] + [None]*5
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_plots/interpretability/blackbox/density8.pdf"
    visualize_density_target(spn, 0, value_dict, max_density=0.1, save_path=save_path)

if __name__ == '__main__':
    demo_visualize_density()
    
    
    
    
    
    
    