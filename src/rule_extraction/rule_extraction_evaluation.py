'''
Created on 5.3.2020

@author: tkrons
'''
import numpy as np
import pandas as pd
from simple_spn import spn_handler
from data import real_data
from simple_spn import functions as fn
from rule_extraction import methods as rule_ex
import pickle
import itertools
import random

from spn.structure.Base import Rule
res_path = '../../_results/rule_extraction/'

def evaluate_rules(data, rules_df, value_dict, metrics=['sup', 'conf', 'head_sup', 'F'], beta=1):
    results = []
    for _, row in rules_df.iterrows():
        head, body = row['head'], row['body']
        pos = body.apply(data, head, value_dict)
        body_true = body.apply(data, value_dict=value_dict)
        head_true = Rule(head).apply(data, value_dict=value_dict)
        support = np.mean(body_true)
        confidence = np.mean(pos) / support
        head_sup = np.mean(head_true)
        results.append([head, body, *metrics])
    df = pd.DataFrame(results, columns = ['head', 'body', *metrics])
    return df

def hyperparam_grid_search(data, spn, value_dict,):
    # hyperparams = {'min_target_js': [0., 0.01, 0.1, 0.2, 0.4],
    #                'min_global_conf': [0.3, 0.6, 0.7, 0.8],
    #                'body_max_len': [4, 5],
    #                'min_local_js': [0., 0.01, 0.1, 0.2, 0.4],
    #                'min_local_p': [0., 0.3]}
    hyperparams = {'min_target_js': [0., 0.1, 0.4],
                   'min_global_conf': ['above_random'],
                   'body_max_len': [4],
                   'min_local_js': [0.],
                   'min_local_p': [0.,],
                   'min_global_F': [0., 0.2, 0.4, 0.6, 0.8],
                   'beta': [0.3],
                   'metrics': ['sup', 'conf', 'conviction', 'F']}

    combinations = list(itertools.product(*hyperparams.values()))
    random.shuffle(combinations)
    n = 1000
    combinations = combinations[:n]
    print('Testing {} combinations of hyperparameters'.format(len(combinations)))
    df_list = []
    for comb in combinations:
        params = dict(zip(hyperparams.keys(), comb))
        intra = rule_ex.IntraNode(**params)
        targts = [3, 5, 6]
        print('Targets: ', [data.columns[i] for i in targts])
        intra_df = intra.intra_rules_df(spn, target_vars=targts, value_dict=value_dict, max_candidates=50,
                                        labels=True)

        eval_intra = evaluate_rules(data, intra_df, value_dict, metrics=metrics, beta=beta)
        # eval_top = evaluate_rules(onehot_df, topdown_rules, vd_onehot, metrics=metrics)
        eval_intra['method'] = 'IntraNode'
        if len(eval_intra) != 0:
            eval_intra = eval_intra.groupby('method').head(50).groupby('method').mean()
        else:
            eval_intra.append(dict(zip(eval_intra.columns, [None]*len(eval_intra.columns))), ignore_index=True).drop(columns=['method'])
        for param, val in params.items():
            eval_intra[param] = val
        df_list.append(eval_intra)
    df = pd.concat(df_list, ignore_index=True).groupby([*hyperparams.keys()]).mean().reset_index()
    df.to_csv(res_path + 'intra_rules_complete_{}.csv'.format(dataset_name))
    l = []
    for p in params.keys():
        group = df.groupby([p])[['F']].mean().reset_index()
        group['param'] = p
        group = group.rename(columns={p: 'param_value'})
        group = group[['param', 'param_value', 'F']]
        l.append(group)
    mean_by_param = pd.concat(l, ignore_index=False, sort=False).reset_index(drop=True)
    mean_by_param.to_csv(res_path + 'intra_rules_hyperparams_{}.csv'.format(dataset_name))
    return df


if __name__ == '__main__':

    # dataset_name = 'adult41'
    # targts = [3,5]
    # dataset_name = 'lending'
    # targts = [0, 7]
    dataset_name = 'titanic'
    targts = [0, 1]
    recalc_SPN = True
    rdc_threshold, min_instances_slice = 0.1, 0.05
    n_rows = 10000
    beta = 0.2

    # get data
    df, value_dict, parametric_types = real_data.get_real_data(dataset_name, only_n_rows=n_rows, seed=1, onehot=False)
    # if not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice) or recalc_SPN:
    #     print("Creating SPN ...")
    #
    #     # Creates the SPN and saves to a file
    #     spn_handler.create_parametric_spns(df.values, parametric_types, dataset_name, [rdc_threshold], [min_instances_slice],
    #                                        clustering = 'km_rule_clustering', value_dict=value_dict)
    #
    # # Load SPN
    # spn, value_dict, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
    spn = spn_handler.load_or_create_spn(df, value_dict, parametric_types, dataset_name, rdc_threshold, min_instances_slice,
                                   nrows=n_rows, seed=1, force_create=recalc_SPN, clustering='km_rule_clustering')

    onehot_df, vd_onehot, pt_onehot = real_data.get_real_data(dataset_name, only_n_rows=n_rows, seed=1, onehot = True)
    spn_one_hot = spn_handler.load_or_create_spn(onehot_df, vd_onehot, pt_onehot, dataset_name + '_one_hot', rdc_threshold, min_instances_slice,
                                   nrows=n_rows, seed=1, force_create=recalc_SPN, clustering='km_rule_clustering')

    # print(fn.get_sub_populations(spn, rule=True))
    # subpops = fn.get_leaf_populations(spn,)
    # l = get_interesting_leaves(spn, subpops[0])
    # rules = [get_labeled_rule(pop[0], df.columns) for pop in l]
    # rules = rule_ex.topdown_interesting_rules(spn, df, value_dict)

    # #todo method: choosing rules based on overlap / overall support
    lax_hyperparams = {'min_target_js': 0.1, 'min_global_conf': 'above_random', 'body_max_len': 6, 'min_local_js': 0.,
                       'min_global_F': 0.05, 'beta': beta, 'metrics': ['sup', 'conf', 'conviction', 'F']}
    rules_per_value = 100

    print('Num of rules expected: ', sum([len(value_dict[t][2].keys()) for t in targts]) * rules_per_value)
    intra = rule_ex.IntraNode(**lax_hyperparams)
    print('Targets: ', [df.columns[i] for i in targts])
    intra_df = intra.intra_rules_df(spn, target_vars=targts, value_dict=value_dict,
                                    rules_per_value = rules_per_value,
                                    # max_candidates=1000,
                                    labels=True,)

    intra_df = intra_df.sort_values(['head', 'F'], ascending=False)
    intra_df.to_csv(res_path + 'intra_rules_{}.csv'.format(dataset_name))
    print(rule_ex.df_display(intra_df))

    if len(vd_onehot[0][2]) == 2:
        topdown_rules = rule_ex.topdown_interesting_rules(spn_one_hot, vd_onehot, full_value_dict = value_dict, beta=beta)
        topdown_rules = topdown_rules.sort_values(['F'], ascending=False)
        topdown_rules.to_csv(res_path + 'topdown_df_{}.csv'.format(dataset_name))
        print(rule_ex.df_display(topdown_rules))
    else:
        print('Data not compatible for topdown')
    # labeled = rule_ex.df2labeled(intra_df, value_dict)

    rules_intra = intra_df.head(len(topdown_rules))
    metrics = ['sup', 'conf', 'conviction', 'F']

    eval_intra = evaluate_rules(df, rules_intra, value_dict, metrics=metrics, beta=beta)
    eval_top = evaluate_rules(onehot_df, topdown_rules, vd_onehot, metrics=metrics, beta=beta)
    eval_intra['method'] = 'IntraNode'
    eval_top['method'] = 'Topdown'
    comp = pd.concat([eval_intra, eval_top], )
    comp.drop_duplicates(['head', 'body'], inplace=True)
    comp = comp.sort_values('F', ascending=False)

    comp.to_csv(res_path + 'comparison_{}.csv'.format(dataset_name))

    # mean F of first N rules
    print(comp.groupby('method').mean())

    hyperparam_grid_search(df, spn, value_dict)

    fn.print_statistics(spn)

