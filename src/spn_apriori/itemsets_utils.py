from time import perf_counter

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules
from orangecontrib.associate import frequent_itemsets as orange_fpgrowth

from spn_apriori.spn_based_apriori import spn_apriori
import simple_spn.functions as fn

def perf_comparison(one_hot_df, min_sup, spn, value_dict):
    print('============= Benchmark ==================================')
    time_spn = []
    time_spn.append(perf_counter())
    for i in range(10):
        s_apriori = spn_apriori(one_hot_df, min_support=min_sup, spn=spn, use_colnames=True, value_dict=value_dict)
        time_spn.append(perf_counter())
    time_diff_spn = [time_spn[i] - time_spn[i - 1] for i in range(1, len(time_spn[1:]))]

    time_apriori = []
    time_apriori.append(perf_counter())
    for i in range(10):
        apriori = mlxtend_apriori(one_hot_df, min_support=min_sup, use_colnames=True, )
        time_apriori.append(perf_counter())
    time_diff_apriori = [time_apriori[i] - time_apriori[i - 1] for i in range(1, len(time_apriori[1:]))]

    #temp time orange fgrowth
    time_fpgrowth = []
    time_fpgrowth.append(perf_counter())
    for i in range(10):
        fpgrowth = fpgrowth_wrapper_orange(one_hot_df, min_sup)
        time_fpgrowth.append(perf_counter())
    time_diff_fpgrowth = [time_fpgrowth[i] - time_fpgrowth[i - 1] for i in range(1, len(time_fpgrowth[1:]))]

    print(time_diff_spn)
    print('Mean time SPN: {}'.format(np.mean(time_diff_spn)))
    print(time_apriori)
    print('Mean time normal apriori: {}'.format(np.mean(time_diff_apriori)))
    print(time_diff_fpgrowth)
    print('Mean time fpgrowth: {}'.format(np.mean(time_diff_fpgrowth)))


def fpgrowth_wrapper_orange(transactional_df, min_sup, ):
    columns = {i: c for i, c in enumerate(transactional_df.columns)}
    total_sup = int(np.floor(min_sup * len(transactional_df)))
    results = []
    for itemset, support in orange_fpgrowth(transactional_df.values, total_sup):
        new_set = frozenset([columns[item] for item in itemset])
        new_support = support / len(transactional_df)
        results.append((new_set, new_support))
    return pd.DataFrame(results, columns=['itemsets', 'support'])

def simple_interpretable_rules(itemsets, top = 30,):
    return _get_interpretable_best_lift_rules(association_rules(itemsets, metric='confidence', min_threshold=0.7), top = 20)

def _get_interpretable_best_lift_rules(rules, top = 30, suffix=''):
    '''
    :param rule_comparison_df: needs cols: [antecedent, consequent, lift]
    :param suffix: suffix of the metric columns. If you have a df with both SPN and apriori metrics, you'll need this
    to specify which metrics to use for rule extraction. Default is empty string ('')
    :return:
    '''
    def _rule_basic_score(row):
        len_ant, len_cons = len(row['antecedents']), len(row['consequents'])
        if len_cons != 1.:
            return 0. #dont want longer consequent for now
        lift = row['lift'+suffix]
        # lambd = [0.2, 0.3, 0.5] #lift = 50%, I = 50%
        return lift / len_ant

    rules['score'] = rules.apply(_rule_basic_score, axis=1)
    return rules.sort_values('score', ascending=False,).head(top)

def set_contains_x(itemset, x):
    '''Usage: rule_comparison.loc[rule_comparison.antecedents.apply(set_contains_x, args='Black')]'''
    return x in itemset

def support_of_set(df, s, ):
    '''manually check support of set s in df. For lookup of missing itemsets'''
    # if not np.isnan(s):
    cols = [str(x) for x in s]
    return len(df[df[cols].all(axis=1)]) / len(df)

def spn_support_of_set(spn, s, value_dict):
    rang = [np.NaN] * len(value_dict)
    for i, l in value_dict.items():
        if l[1] in s:
            rang[i] = 1
    return fn.prob_spflow(spn, rang)

if __name__ == '__main__':
    pass