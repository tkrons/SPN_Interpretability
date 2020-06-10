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
from scipy.stats import norm
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import os

from spn.structure.Base import Rule
res_path = '../../_results/rule_extraction/'

import rule_extraction.rule_extraction_evaluation as Eval
from data.data_utils import onehot2transactional
from spn_apriori.itemsets_utils import calc_itemsets_df
from spn_apriori.spn_based_apriori import generate_rules_apriori, spn_apriori
from mlxtend.frequent_patterns import association_rules, apriori

# todo rdc mis analysis?
# dataset_name = 'adult41'
# targts = [3,5]
dataset_name = 'titanic'
n_rows = 8000
# targts = [0, 1]
# dataset_name = 'lending'
recalc_SPN = False
rdc_threshold, min_instances_slice = 0.1, 0.01
beta = 0.2
metrics = ['sup', 'conf', 'lift', 'F', 'leverage', 'interestingness', 'PiSh', 'jaccard', 'cosine_distance']

# get data
df, value_dict, parametric_types = real_data.get_real_data(dataset_name, only_n_rows=n_rows, seed=1, onehot=False)
spn = spn_handler.load_or_create_spn(df, value_dict, parametric_types, dataset_name, rdc_threshold, min_instances_slice,
                               nrows=n_rows, seed=1, force_create=recalc_SPN, clustering='km_rule_clustering')

onehot_df, vd_onehot, pt_onehot = real_data.get_real_data(dataset_name, only_n_rows=n_rows, seed=1, onehot = True)
spn_one_hot = spn_handler.load_or_create_spn(onehot_df, vd_onehot, pt_onehot, dataset_name + '_one_hot', rdc_threshold, min_instances_slice,
                               nrows=n_rows, seed=1, force_create=recalc_SPN, clustering='rule_clustering')

if dataset_name == 'lending':
    targts = [0, 7]
else:
    targts = []
    for t in df.nunique()[df.nunique() < 4].index: # todo all targets!
        targts.append(list(df.columns).index(t))

# print(fn.get_sub_populations(spn, rule=True))
# subpops = fn.get_leaf_populations(spn,)
# l = get_interesting_leaves(spn, subpops[0])
# rules = [get_labeled_rule(pop[0], df.columns) for pop in l]
# rules = rule_ex.topdown_interesting_rules(spn, df, value_dict)

# #todo method: choosing rules based on overlap / overall support
test_hyper = {'min_target_js': 0.1, 'min_global_conf': 0,
                          'body_max_len': 6, 'min_local_js': 0.1,
                          'min_global_criterion': 1.3, 'criterion': 'lift',
                          'beta': beta, 'metrics': metrics}
comparable_hyperparams = {'min_target_js': 0.05, 'min_global_conf': 0,
                          'body_max_len': 6, 'min_local_js': 0.,
                          'min_global_criterion': 1.2, 'criterion': 'lift',
                          'beta': beta, 'metrics': metrics}
min_threshold_ap = 1.0
criterion_ap = 'lift'
rules_per_value = 5
print('possible num of target values: {}'.format(df[df.columns[targts]].nunique().sum() * rules_per_value))

print('INTRA')

path_intra = res_path + '/cache/intra_{}.pckl'.format(dataset_name)
if os.path.exists(path_intra):
    print('loading')
    with open(path_intra, 'rb') as f:
        intra_df = pickle.load(f)
else:
    print('calculating')
    intra = rule_ex.IntraNode(**comparable_hyperparams)
    intra_df = intra.intra_rules_df(df, spn, target_vars=targts, value_dict=value_dict,
                                    rules_per_value = rules_per_value,
                                    # max_candidates=1000,
                                    labels=True,)

    intra_df = intra_df.sort_values(['head', 'F'], ascending=False)
    with open(path_intra, 'wb') as f:
        pickle.dump(intra_df, f)
# print(rule_ex.df_display(intra_df))
print('Targets: ', [df.columns[i] for i in targts])

print('TOPDOWN')

path_topdown = res_path + '/cache/topdown_df_{}.csv'.format(dataset_name)
if os.path.exists(path_topdown):
    print('loading')
    topdown_rules = pd.read_csv(path_topdown, index_col=0)
else:
    print('calculating')
    topdown_rules = rule_ex.topdown_interesting_rules(spn_one_hot, vd_onehot, metrics, full_value_dict = value_dict, beta=beta)
    topdown_rules.to_csv(path_topdown)
topdown_rules = topdown_rules.sort_values(['F'], ascending=False)
print(rule_ex.df_display(topdown_rules))

print('APRIORI')
# min_sup = 0.05
# min_sup = min([intra_df.sup.min(), topdown_rules.sup.min()])
min_sup = min(np.round(intra_df.sup.sort_values(ascending=True).head(3).mean() * 2, 2) , 0.02)

transactional = onehot2transactional(onehot_df)

# path_itemsets = res_path + '/cache/rules_{}_{}_{}.pckl'.format('spn_ap', dataset_name, min_sup)
path_ap = res_path + '/cache/rules_{}_{}_{}.pckl'
if os.path.exists(path_ap.format('spn_ap', dataset_name, min_sup)) and os.path.exists(path_ap.format('ap', dataset_name, min_sup)):
    print('=================== Reading itemsets: {} ============'.format(path_ap))
    with open(path_ap.format('spn_ap', dataset_name, min_sup), 'rb') as f:
        spn_ap_rules = pickle.load(f)
    with open(path_ap.format('ap', dataset_name, min_sup), 'rb') as f:
        ap_rules = pickle.load(f)
else:
    all_itemsets = calc_itemsets_df(onehot_df, spn_one_hot, min_sup, value_dict=vd_onehot)
    print('len(all_itemset) == {}'.format(len(all_itemsets)))
    #filter itemsets == target
    target_str = []
    for t in targts:
        for _, name, _ in vd_onehot.values():
            if value_dict[t][1] in name:
                target_str.append(name)
    def target_in_set(targets, s):
        for t in targets:
            if t in s:
                return True
        return False
    # all_itemsets = all_itemsets[all_itemsets['itemsets'].apply(lambda x: target_in_set(target_str, x),)]

    spn_apriori_df = all_itemsets[all_itemsets.support_pred >= min_sup].reset_index()[['itemsets', 'support']]
    normal_apriori_df = all_itemsets[all_itemsets.support >= min_sup].reset_index()[['itemsets', 'support']]
    spn_ap_rules = association_rules(spn_apriori_df, metric=criterion_ap, min_threshold=min_threshold_ap)
    ap_rules = association_rules(normal_apriori_df, metric=criterion_ap, min_threshold=min_threshold_ap)

    spn_ap_rules = spn_ap_rules[spn_ap_rules['consequents'].apply(lambda x: target_in_set(target_str, x),)]
    ap_rules = ap_rules[ap_rules['consequents'].apply(lambda x: target_in_set(target_str, x),)]

    spn_ap_rules['F'] = rule_ex.fbeta_score(spn_ap_rules.confidence, spn_ap_rules.support, 0.2)
    spn_ap_rules = spn_ap_rules.loc[spn_ap_rules.consequents.apply(len) == 1]

    ap_rules['F'] = rule_ex.fbeta_score(ap_rules.confidence, ap_rules.support, 0.2)
    ap_rules = ap_rules.loc[ap_rules.consequents.apply(len) == 1]

    # del all_itemsets, spn_apriori_df, normal_apriori_df #todo ?

    # frozenset format to Rule(Condition) format
    spn_ap_rules = spn_ap_rules[['consequents', 'antecedents']].apply(
        lambda x: rule_ex.format_mlxtend2rule_ex(*x),
        axis=1, result_type='expand',
    ).rename(columns={0:'head', 1:'body'})
    #filter head==target
    # spn_ap_rules = spn_ap_rules[spn_ap_rules['head'].apply(lambda x: target_in_set(target_str, x))]
    ap_rules = ap_rules[['consequents', 'antecedents']].apply(
        lambda x: rule_ex.format_mlxtend2rule_ex(*x),
        axis=1, result_type='expand'
    ).rename(columns={0:'head', 1:'body'})
    # ap_rules = ap_rules[ap_rules['head'].apply(lambda x: target_in_set(target_str, x))]

    spn_ap_stats = spn_ap_rules.apply(
        lambda x: tuple(rule_ex.rule_stats(spn_one_hot, x['body'], x['head'], metrics,
                                           real_data=onehot_df, beta=beta, value_dict=vd_onehot)),
        axis=1,
        result_type='expand'
    ).rename(columns={i: metr for i, metr in enumerate(metrics)})
    spn_ap_rules = spn_ap_rules.merge(spn_ap_stats, left_index=True, right_index=True)

    ap_stats = ap_rules.apply(
        lambda x: tuple(rule_ex.rule_stats(spn_one_hot, x['body'], x['head'], metrics,
                                           real_data=onehot_df, beta=beta, value_dict=vd_onehot)),
        axis=1,
        result_type='expand'
    ).rename(columns={i: metr for i, metr in enumerate(metrics)})
    ap_rules = ap_rules.merge(ap_stats, left_index=True, right_index=True)
    del spn_ap_stats, ap_stats

    with open(path_ap.format('ap', dataset_name, min_sup), 'wb') as f:
        pickle.dump(ap_rules, f)
    with open(path_ap.format('spn_ap', dataset_name, min_sup), 'wb') as f:
        pickle.dump(ap_rules, f)


print('compile comparison_df')
intra_df['method'] = 'IntraNode'
topdown_rules['method'] = 'Topdown'
spn_ap_rules['method'] = 'SPN-Apriori'
ap_rules['method'] = 'Apriori'

dfs = [
    intra_df,
    topdown_rules,
    spn_ap_rules,
    ap_rules
]
eval = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
eval.sort_values(['method', 'head', 'F'])
eval.to_csv(res_path + 'evaluation_{}.csv'.format(dataset_name))

print('SUMMARY')

# SAMPLING
height, bins, _ = plt.hist(intra_df.sup, bins=10)
bins = bins[1:]
plt.close()
height = [(h * rules_per_value) / sum(height) for h in height]

def sample_rules(ap_rules, bins, height, rules_per_value, target_sup):
    idx = pd.Index([])
    for head in ap_rules['head'].unique():
        # print(head)
        # print(len(idx))
        added = 0
        assert ap_rules[~ap_rules.index.isin(idx) & (ap_rules['head'] == head)].__len__() > rules_per_value, \
            'Not enough rules for head: {}'.format(head)
        for i, (h, bin) in enumerate(zip(height, bins)):
            h = int(rule_ex.prob_round(h))
            if (h + added) > rules_per_value:
                h = rules_per_value - added
            elif i == len(bins) - 1 and h + added != rules_per_value: # need correction
                h = max(rules_per_value - added, 0)
            if i == 0:
                lower_bound=0
            else:
                lower_bound = bins[i-1]
            d = ap_rules[(ap_rules['head'] == head) & (ap_rules.sup.between(lower_bound, bin)) &
                         ~ap_rules.index.isin(idx)]
            if h < len(d):
                s = d.sample(h, weights=d.sup).index
            else:
                s = d.index
            idx = idx.append(s)
            added += len(s)
        # end of loop
        if added > rules_per_value:
            raise ValueError()
        elif added < rules_per_value: # add the rest
            idx = idx.append(ap_rules.loc[(~ap_rules.index.isin(idx)) & (ap_rules['head'] == head)].sample(rules_per_value - added).index)
            added += rules_per_value - added
        assert added == rules_per_value
        assert len(idx) % rules_per_value == 0, head
    assert len(idx) == len(ap_rules.groupby('head')['sup'].count()) * rules_per_value
    ap_rand = ap_rules.loc[idx]
    idx = list(idx)
    # if still not properly distributed ... switch sampled rows with more fitting ones
    while not np.isclose(ap_rules.loc[idx].sup.mean(), target_sup, rtol=0.05):
        # s0 = ap_rules.loc[idx].sup.mean()
        if ap_rules.loc[idx].sup.mean() < target_sup:
            i = ap_rules[ap_rules.sup < target_sup].reindex(idx).sample().index[0]
            current_sup = ap_rules.loc[i, 'sup']
            better = ap_rules.loc[ap_rules.sup > current_sup]
        elif ap_rules.loc[idx].sup.mean() > target_sup:
            i = ap_rules[ap_rules.sup > target_sup].reindex(idx).sample().index[0]
            current_sup = ap_rules.loc[i, 'sup']
            better = ap_rules.loc[ap_rules.sup < current_sup]
        head = ap_rules.loc[i, 'head']
        samehead = ap_rules['head'] == head
        better = better.loc[(~ap_rules.index.isin(idx)) & samehead]
        if len(better) > 0:
            idx[idx.index(i)] =  better.sample().index[0]

        # s1 = ap_rules.loc[idx].sup.mean()
        # assert s0 != s1
    return ap_rules.loc[idx]

l_ap, l_spn_ap, iterations = [], [], 3
for i in range(iterations):
    ap_rand = sample_rules(ap_rules, bins, height, rules_per_value, intra_df.sup.mean())

    ap_rand.method.replace({'Apriori': 'Apriori-rand_{}'.format(i)}, inplace=True)
    l_ap.append(ap_rand)
    spn_ap_rand = sample_rules(spn_ap_rules, bins, height, rules_per_value, intra_df.sup.mean())
    spn_ap_rand.method.replace({'SPN-Apriori': 'SPN-Apriori-rand_{}'.format(i)}, inplace=True)
    l_spn_ap.append(spn_ap_rand)
ap_rand = pd.concat(l_ap, ignore_index=True).reset_index(drop=True)
spn_ap_rand = pd.concat(l_spn_ap, ignore_index=True).reset_index(drop=True)

sort_arg = {'by': 'cosine_distance', 'ascending': True}
ap_topn = eval[eval.method.isin(['Apriori', 'SPN-Apriori'])].sort_values(**sort_arg).groupby(
    ['method', 'head']).head(rules_per_value)
ap_topn.method = ap_topn.method.replace({'SPN-Apriori': 'SPN-Apriori TopN', 'Apriori': 'Apriori TopN'})
summary = pd.concat([eval[eval.method.isin(['Topdown', 'IntraNode'])],
                     ap_rand,
                     spn_ap_rand,
                     ap_topn], ignore_index=True, sort=False).reset_index(drop=True)
summary['head'] = summary['head'].astype(str)
aggs = {'sup': ['mean', 'count'], 'conf': 'mean', 'lift': 'median', 'F': 'mean', 'leverage': 'mean', 'interestingness': 'mean', 'PiSh': 'mean', 'jaccard': 'mean', 'cosine_distance': 'mean'} # list of list would be mutable references
summary = summary.groupby('method').agg(aggs).round(4)
# group iteration samples
labls_sap = ['SPN-Apriori-rand_{}'.format(i) for i in range(iterations)]
labls_ap = ['Apriori-rand_{}'.format(i) for i in range(iterations)]
summary.loc['Apriori-rand', :] = summary.loc[labls_ap].mean()
summary.loc['SPN-Apriori-rand', :] = summary.loc[labls_sap].mean()
summary.drop(index=labls_sap + labls_ap, inplace=True)
summary.columns = summary.columns.droplevel(1)
summary.columns.values[1] = 'count'
summary = summary[['count', *metrics]]
summary = summary.loc[['IntraNode', 'Topdown', 'Apriori TopN', 'SPN-Apriori TopN', 'Apriori-rand', 'SPN-Apriori-rand', ]]
print('rdc_treshold: {} min_instances_slice: {}'.format(rdc_threshold, min_instances_slice))
# summary.to_csv(res_path+'summary_{}.csv'.format(dataset_name))

to_display_metrics = ['count', 'sup', 'conf', 'lift', 'F', 'cosine_distance']
summary[to_display_metrics].to_latex(res_path+'summary_{}.txt'.format(dataset_name), )
print(summary[to_display_metrics])

pass




