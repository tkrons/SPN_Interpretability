'''
Created on 19.07.2019

@author: tkrons
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from data import real_data, synthetic_data
from simple_spn import spn_handler
from spn_apriori.itemsets_utils import cross_eval
from spn.structure.leaves.parametric.Parametric import Categorical

dataset_name = 'adult_one_hot'
recalc_spn = False


rdc_range = [0.1, 0.2, 0.3]
mis_range = [0.1, 0.01, 0.001]
min_sup_range = [0.01, 0.03, 0.1, 0.3]

transactional_df, value_dict, parametric_types = real_data.get_real_data(dataset_name)
# transactional_df, value_dict, parametric_types = synthetic_data.get_synthetic_data(dataset_name)


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

# HEATMAP
for which_comparison in ['train_vs_test', 'spn_vs_train', 'spn_vs_test']:
    error_to_use = 'MAE'
    # which_comparison = 'spn_vs_train'

    # fig, axes = plt.subplots(int(np.ceil(len(min_sup_range) / 3)), 3, figsize=(12,8),
    #                          sharex=True, sharey=True)
    nc, nr = 2,2
    fig, axes = plt.subplots(nr, nc, figsize=(8,8), sharex=True, sharey=True)
    assert len(axes.flat) == int(nr * nc)
    for i, min_sup in enumerate(min_sup_range):
        ax = axes.flat[i]
        df = cross_eval_hyperparams[error_to_use].xs([min_sup, which_comparison], level=[1,2])
        df = df.reset_index()
        df['rdc'] = df['SPN Params'].apply(lambda x: x[0])
        df['mis'] = df['SPN Params'].apply(lambda x: x[1])
        df = df.drop(columns=['SPN Params']).pivot(index='mis', columns='rdc').sort_index(ascending=False)
        print(df)
        mat = df.values
        zmax = df[error_to_use].max().max()
        im = ax.imshow(mat, vmax=zmax, cmap=plt.get_cmap('coolwarm'))

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(rdc_range)))
        ax.set_yticks(np.arange(len(mis_range)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(rdc_range)
        ax.set_yticklabels(mis_range)

        # Loop over data dimensions and create text annotations.
        for i in range(len(mis_range)):
            for j in range(len(rdc_range)):
                text = ax.text(j, i, r'{:3f}'.format(mat[i, j]), ha="center", va="center", color="black")

        ax.set_title('{} minsup: {}'.format(error_to_use, min_sup))

    # fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    # # plt.colorbar(im, ax=ax)
    # fig.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    fig.colorbar(im, cax=cax)

    # add a big axes, hide frame
    totalax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    totalax.grid(False)

    plt.xlabel("rdc_threshold")
    plt.ylabel("min_instances_split")
    # plt.title('MAE of SPN-apriori {}'.format(which_comparison))

    p="../../_figures/{}".format(dataset_name)
    if not os.path.exists(p):
        os.makedirs(p)
    plt.savefig("../../_figures/{}/{}_subplots_heatmap.pdf".format(dataset_name, which_comparison))
    plt.show()

for min_sup in min_sup_range:
    xs = cross_eval_hyperparams['MAE'].xs([(0.1, 0.01),], level=[0]).reset_index()
    xs = xs[xs['compare'].isin(['train_vs_test', 'spn_vs_test'])]
    glz = pd.pivot_table(xs, columns=['compare'], index=['min_sup'], values=['MAE'])
    print('=========== Generalization Comparison =============')
    print(glz.to_string)
    print(glz.to_latex())


















