from time import perf_counter

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from orangecontrib.associate import frequent_itemsets as orange_fpgrowth

from spn_apriori.spn_based_apriori import spn_apriori


def perf_comparison(one_hot_df, min_sup, spn, value_dict):
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
        fpgrowth = fpgrowth_wrapper(one_hot_df, min_sup)
        time_fpgrowth.append(perf_counter())
    time_diff_fpgrowth = [time_fpgrowth[i] - time_fpgrowth[i - 1] for i in range(1, len(time_fpgrowth[1:]))]

    print(time_diff_spn)
    print('Mean time SPN: {}'.format(np.mean(time_diff_spn)))
    print(time_apriori)
    print('Mean time normal apriori: {}'.format(np.mean(time_diff_apriori)))
    print(time_diff_fpgrowth)
    print('Mean time fpgrowth: {}'.format(np.mean(time_diff_fpgrowth)))


def fpgrowth_wrapper(transactional_df, min_sup,):
    columns = {i: c for i, c in enumerate(transactional_df.columns)}
    total_sup = int(np.floor(min_sup * len(transactional_df)))
    results = []
    for itemset, support in orange_fpgrowth(transactional_df.values, total_sup):
        new_set = frozenset([columns[item] for item in itemset])
        new_support = support / len(transactional_df)
        results.append((new_set, new_support))
    return pd.DataFrame(results, columns=['itemsets', 'support'])

def rule_interpretability_metric(rule_comparison_df):
    #todo
    #criteria length, high confidence, high conviction
    def _calc_I_row(row):
        row['antecedent']