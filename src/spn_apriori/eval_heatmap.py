'''
Created on 19.07.2019

@author: Moritz, Tim
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data import real_data, synthetic_data
from simple_spn import spn_handler
from spn_apriori.apriori_evaluation import cross_eval
from spn.structure.leaves.parametric.Parametric import Categorical


dataset_name = 'UCI'
rdc_threshold, min_instances_slice = 0.1, 0.05
recalc_spn = False

rdc_range = [0.1, 0.2, 0.3]
mis_range = [0.1, 0.01, 0.001]
min_sup_range = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4]

if dataset_name == 'UCI':
    transactional_df, value_dict, parametric_types = real_data.get_adult_41_items()

# SPN generation
if recalc_spn or not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
    print("======================== Creating SPN ... ===============")
    parametric_types = [Categorical for _ in transactional_df.columns]
    # Creates the SPN and saves to a file
    spn_handler.create_parametric_spns(transactional_df.values, parametric_types, dataset_name, value_dict=value_dict,
                                       rdc_thresholds=[rdc_threshold],
                                       min_instances_slices=[min_instances_slice],
                                       silence_warnings=True)


# eval different hyper params
cross_eval_hyperparams = []
for rdc in rdc_range:
    for mis in mis_range:
        print('SPN Params: rdc {}, mis {}'.format(rdc, mis))
        eval_spn_params = cross_eval(transactional_df, dataset_name, min_sup_range, value_dict, recalc_spn=recalc_spn,
                                     min_instances_slice=mis, rdc_threshold=rdc)
        eval_spn_params.reset_index(inplace=True)
        eval_spn_params['SPN Params'] = [(rdc, mis)] * len(eval_spn_params)  # assigning a list doesnt work
        cross_eval_hyperparams.append(eval_spn_params)
cross_eval_hyperparams = pd.concat(cross_eval_hyperparams, ignore_index=True).set_index(
    ['SPN Params', 'min_sup', 'compare'])
print(cross_eval_hyperparams.to_string())
cross_eval_hyperparams.to_csv('cross_eval_hyper.csv', sep=',')

error_to_use = 'MAE'
for min_sup in min_sup_range:

    df = cross_eval_hyperparams[error_to_use].xs([min_sup, 'spn_vs_test'], level=[1,2])
    df = df.reset_index()
    df['rdc'] = df['SPN Params'].apply(lambda x: x[0])
    df['mis'] = df['SPN Params'].apply(lambda x: x[1])
    df = df.drop(columns=['SPN Params']).pivot(index='mis', columns='rdc').sort_index(ascending=False)
    print(df)
    mat = df.values

    fig, ax = plt.subplots()
    im = ax.imshow(mat)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(rdc_range)))
    ax.set_yticks(np.arange(len(mis_range)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(rdc_range)
    ax.set_yticklabels(mis_range)


    # Loop over data dimensions and create text annotations.
    for i in range(len(mis_range)):
        for j in range(len(rdc_range)):
            text = ax.text(j, i, r'{:3f}'.format(mat[i, j]), ha="center", va="center", color="w")

    plt.colorbar(im, ax=ax)
    ax.set_title('{} spn_vs_train for different rdc and mis values. minsup: {}'.format(error_to_use, min_sup))
    fig.tight_layout()
    # plt.savefig("heatmap_test.pdf")

    plt.show()




    #
    # ncols = 2
    # nrows = 2
    # figsize_x = 6
    # figsize_y = 6
    # fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x,figsize_y), squeeze=False)
    #
    #
    #
    # im = axes[0][0].imshow(harvest)
    # axes[0][0].set_xticks(np.arange(len(farmers)))
    # axes[0][0].set_yticks(np.arange(len(vegetables)))
    # axes[0][0].set_xticklabels(farmers)
    # axes[0][0].set_yticklabels(vegetables)
    #
    #
    # im = axes[1][0].imshow(harvest)
    # axes[1][0].set_xticks(np.arange(len(farmers)))
    # axes[1][0].set_yticks(np.arange(len(vegetables)))
    # axes[1][0].set_xticklabels(farmers)
    # axes[1][0].set_yticklabels(vegetables)
    #
    # im = axes[0][1].imshow(harvest)
    # axes[0][1].set_xticks(np.arange(len(farmers)))
    # axes[0][1].set_yticks(np.arange(len(vegetables)))
    # axes[0][1].set_xticklabels(farmers)
    # axes[0][1].set_yticklabels(vegetables)
    #
    #
    # im = axes[1][1].imshow(harvest)
    # axes[1][1].set_xticks(np.arange(len(farmers)))
    # axes[1][1].set_yticks(np.arange(len(vegetables)))
    # axes[1][1].set_xticklabels(farmers)
    # axes[1][1].set_yticklabels(vegetables)
    #
    #
    # plt.tight_layout()
    # plt.show()

















