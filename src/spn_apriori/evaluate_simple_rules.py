from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
from mlxtend.preprocessing import TransactionEncoder
transaction_encoder = TransactionEncoder()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from data import real_data
from data import synthetic_data
from simple_spn import spn_handler
from spn.structure.leaves.parametric.Parametric import Categorical
from spn_apriori.itemsets_utils import simple_interpretable_rules, _get_interpretable_best_lift_rules, calc_itemsets_df
import simple_spn.functions as fn

dataset_name = 'UCI'
rdc_threshold, min_instances_slice = 0.1, 0.05
min_sup=0.01
recalc_spn =False

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

# Load SPN
spn, _, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)


all_itemsets = calc_itemsets_df(transactional_df, spn, min_sup, value_dict=value_dict)

print('================ Calculating Rules and Metrics ===============')
spn_apriori_df = all_itemsets.reset_index()[['itemsets', 'support_pred']]
normal_apriori_df = all_itemsets.reset_index()[['itemsets', 'support']]
spn_rules = association_rules(spn_apriori_df.rename(columns={'support_pred': 'support'}),
                              metric='confidence', min_threshold=0.80)
normal_apriori_rules = association_rules(normal_apriori_df,
                                         metric='confidence', min_threshold=0.80)

# ======= Analyze resulting rule quality
rule_comparison = pd.merge(spn_rules, normal_apriori_rules, on=['antecedents', 'consequents'], how='outer', suffixes=['_SPN', '_Apriori'])
# apply interpretability score
rule_comparison = _get_interpretable_best_lift_rules(rule_comparison, top=50, suffix='_SPN')
rule_comparison.set_index(['antecedents', 'consequents'], inplace=True)
# rule_comparison.dropna(inplace=True)

for col in rule_comparison.columns:
    if '_SPN' in col:
        metric = col.split('_SPN')[0]
        rule_comparison[metric + '_Diff'] = rule_comparison[col] - rule_comparison[metric + '_Apriori']
rule_comparison = rule_comparison.reset_index()
# print(rule_comparison.std(axis=0))
ordered_cols = ['antecedents', 'consequents']
for  metric in list(set(spn_rules.columns) - set(['antecedents', 'consequents'])):
    ordered_cols = ordered_cols + [metric +'_SPN', metric +'_Apriori', metric +'_Diff']
rule_comparison = rule_comparison.reindex(ordered_cols, axis=1)
print(rule_comparison.columns)
rule_comparison.sort_values(by='lift_SPN', ascending=False, inplace=True)
print(rule_comparison.sort_values(by='lift_SPN', ascending=False).head(10).to_string())
print(rule_comparison.mean())