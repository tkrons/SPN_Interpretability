from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules as mlxtend_association_rules
from spn_apriori.spn_based_apriori import spn_apriori
from mlxtend.preprocessing import TransactionEncoder
transaction_encoder = TransactionEncoder()
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from data import real_data
from simple_spn import spn_handler
from simple_spn import functions as fn
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.experiments.AQP.Ranges import NominalRange, NumericRange
from time import perf_counter

import pandas as pd
import numpy as np
from itertools import compress

def perf_comparison(one_hot_df, min_sup, spn, value_dict):
    time_spn = []
    time_spn.append(perf_counter())
    for i in range(10):
        spn_apriori(one_hot_df, min_support=min_sup, spn=spn, use_colnames=True, value_dict=value_dict)
        time_spn.append(perf_counter())
    time_spn = [x - time_spn[i - 1] for i, x in enumerate(time_spn[1:])]

    time_apriori = []
    time_apriori.append(perf_counter())
    for i in range(10):
        mlxtend_apriori(one_hot_df, min_support=min_sup, use_colnames=True, )
        time_apriori.append(perf_counter())
    time_apriori = [x - time_apriori[i - 1] for i, x in enumerate(time_apriori[1:])]
    print(time_spn)
    print('Mean time SPN: {}'.format(np.mean(time_spn)))
    print(time_apriori)
    print('Mean time normal apriori: {}'.format(np.mean(time_apriori)))


'''
results: bigger support: SPN_apriori approaches normal apriori. smaller generalization error
support < 0.3: spn.prob < apriori
support > 0.3: spn.prob ~= apriori

#todo was macht die mit hoher positiver diff besonders? wann hat das SPN höhe Werte? 
einfach Zufall?

#Performance 
synthetic(n=200):
Mean time SPN: 0.009864396915714632
Mean time normal apriori: 0.007377231038096182
synthetic(n=5000):
Mean time SPN: 0.01371689496158297
Mean time normal apriori: 0.007232121828589372
'''

## PARAMETERS ##
dataset_name = "UCI_tabular"
only_n_rows = 3
min_sup = 0.01
# min_sup = 0 oder nah an null lässt PC einfrieren..
recalc_spn = True
benchmark = False
# np.random.seed(1)
seed = np.random.randint(1000)
print('Seed: {}'.format(seed))
np.random.seed(seed)

#when changing the dataset, dont forget to recalculate the spn!
if dataset_name == 'apriori_test':
    # apple: P(A) = 0.7
    # cheese P(C | A) = 0.5 P(C | !A) = 0.2
    # beer: P(B | C) = 0.3 P(B | !C) = 0.6
    # dumpling P(D) = 0.7 uncorrelated
    data = np.random.choice([0, 1], size=(only_n_rows or 1000, 4), p=[0.3, 0.7])
    df = pd.DataFrame(data, columns=['apple', 'beer', 'cheese', 'dumpling'], dtype=np.bool)
    df['cheese'] = df['apple'].apply(lambda x: np.random.random() < 0.5 if x else np.random.random() < 0.2)
    df['beer'] = df['cheese'].apply(lambda x: np.random.random() < 0.3 if x else np.random.random() < 0.6)
    value_dict = {0: ['discrete', 'apple', {0: False, 1: True}],
                  1: ['discrete', 'beer', {0: False, 1: True}],
                  2: ['discrete', 'cheese', {0: False, 1: True}],
                  3: ['discrete', 'dumpling', {0: False, 1: True}]}
    parametric_types = [Categorical, Categorical, Categorical, Categorical]
    transactional_df = df # equal in this case

elif dataset_name == 'UCI':
    df, value_dict, parametric_types = real_data.get_adult_transactional()
    transactional_df = df # equal in this case
elif dataset_name == 'UCI_tabular':
    df, value_dict, parametric_types = real_data.get_adult_transactional(convert_tabular=True)
    #get transactions for normal apriori
    transactional_df, _, _ = real_data.get_adult_transactional()

if only_n_rows and only_n_rows < len(df):
    df = df.sample(only_n_rows)

def strings_from_onehot(one_hot):
    transactions = one_hot.apply(lambda x: set(compress(x.index, x)) or 'empty transaction', axis=1)
    return transactions[transactions != 'empty transaction']
if dataset_name != 'UCI_tabular': #transactions descriptions
    transactions = strings_from_onehot(df)
    print('Transactions:\n', transactions.sample(3))
    itemlist = list(set().union(*transactions.values))
    print('Items:\n{}'.format(itemlist))
    print('Number of Items: {} Number of Transactions: {}'.format(len(itemlist), len(transactions)))

#SPN generation
rdc_threshold, min_instances_slice = 0.3, 0.05
if recalc_spn or not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
    print("======================== Creating SPN ... ===============")
    parametric_types = [Categorical for _ in df.columns]
    # Creates the SPN and saves to a file
    spn_handler.create_parametric_spns(df.values, parametric_types, dataset_name,
                                       rdc_thresholds=[rdc_threshold], min_instances_slices=[min_instances_slice])

# Load SPN
spn, _, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
index2name = {i: name for i, name in enumerate(df.columns)}

# sets, rules = effiecient_apriori(transactions, min_support=0.2,
#                       # spn=(spn, index2name)
#                       )
# print(sets, rules)


if benchmark:
    perf_comparison(transactional_df, min_sup, spn, value_dict)

print('==================== Calculating Itemsets ==============')
spn_apriori_df = spn_apriori(transactional_df, min_support=min_sup, spn=spn, value_dict=value_dict, use_colnames=True, )
normal_apriori_df = mlxtend_apriori(transactional_df, min_support=min_sup, use_colnames=True, )
spn_apriori_df = spn_apriori_df.sort_values('support', ascending=False).reset_index(drop=True).rename(
            columns={'support': 'spn.prob'}).set_index('itemsets')
normal_apriori_df = normal_apriori_df.sort_values('support', ascending=False).reset_index(drop=True).set_index('itemsets')

both = spn_apriori_df.join(how='outer', other=normal_apriori_df,)
if min_sup < 0.01: # ease comparison and visualization
    both = both.fillna(value={'support': 0, 'spn.prob': 0})

both['support_mean'] = both[['spn.prob', 'support']].mean(axis=1)
both['difference'] = both.apply(lambda x: np.round(x['spn.prob'] - x['support'], 4), axis=1)
both['difference_percent'] = both.apply(lambda x: np.round((x['spn.prob'] - x['support']) / x['spn.prob'], 4), axis=1)
both['length'] = both.index.to_series().apply(lambda x: len(x))
both = both[both['length'] >= 2]
both = both.sort_values('support_mean', ascending=False)
print(both.to_string())

#only keep sets we got from both algorithms. feel free to modify
both = both.dropna(subset=['spn.prob', 'support'])

#polynomial regression to check trend of the difference
regX = np.c_[both['spn.prob'].values, #(both['support'] ** 2).values
]
regy = both['difference'].values.reshape(-1, 1)
poly_reg = Ridge(alpha=0.5).fit(regX, regy)
print('Coef: ', poly_reg.intercept_, poly_reg.coef_)
linearspace = np.c_[np.linspace(0, 1, 100),]
# reg_yhat = poly_reg.intercept_ + linearspace.dot(poly_reg.coef_.reshape(-1))
reg_yhat = poly_reg.predict(linearspace)

# #reg on difference.abs() to check for heteroscedasticity? doesnt work like this
# abs_reg = Ridge(alpha=1).fit(both['support'].values.reshape(-1,1), both['difference'].abs().values.reshape(-1,1))
# abs_reg_yhat = abs_reg.intercept_ + linearspace.dot(abs_reg.coef_)
# plt.scatter(both['spn.prob'], both['difference'].abs(), color='blue', label='SPN', marker='o', s=6)
# plt.plot(linearspace[:, 0], abs_reg_yhat, color='red')
# plt.show()

# xy = sorted(xy, key=lambda x: x[0])
plt.scatter(both['spn.prob'], both['difference'], color='blue', label='SPN', marker='o', s=6)
plt.plot(np.linspace(0, 1, 100), np.zeros([100, 1]), color='black', ) # 0 line
plt.plot(linearspace[:, 0], reg_yhat, color='red')
xy = [both['support'].values, both['difference'].values]
plt.xlim(0, xy[0].max().item() + 0.05)
# plt.ylim(-0.05, 0.05)
plt.xlabel('itemset spn support')
plt.ylabel('support_SPN - support_Apriori (difference)')
# both = both.drop(columns=['support_mean'])
plt.show()


print('itemsets with the biggest difference:')
# print(both.sort_values('difference', ascending=False).head(4).to_string())
print(both.reindex(both['difference'].abs().sort_values(ascending=False).index).head(4).to_string())
print('itemsets with the biggest percentage difference:')
print(both.reindex(both['difference_percent'].abs().sort_values(ascending=False).index).head(4).to_string())
print(both.sort_values('difference_percent', ascending=False).head(4).to_string())
print('Std. by support brackets:')
brackets = [0, 0.01, 0.1, 0.2, 0.5, 1]
for i in range(1, len(brackets)):
    std = np.round(both[ both['support_mean'].between(brackets[i-1], brackets[i]) ].difference.std(), 7)
    print('{}: {}'.format(brackets[i], std), )

print('================ Calculating Rules and Metrics ===============')

spn_rules = mlxtend_association_rules(spn_apriori_df.rename(columns={'spn.prob': 'support'}).reset_index(),
                              metric='confidence', min_threshold=0.65)
normal_apriori_rules = mlxtend_association_rules(normal_apriori_df.reset_index(),
                                                 metric='confidence', min_threshold=0.65)

#======= Analyze differences between sets and metrics
# spn_rules, normal_apriori_rules = spn_rules.set_index(['antecedents', 'consequents']), normal_apriori_rules.set_index(['antecedents', 'consequents'])
# rule_comparison = spn_rules.join(normal_apriori_rules, how='outer', lsuffix = '_SPN', rsuffix = '_Apriori')
rule_comparison = pd.merge(spn_rules, normal_apriori_rules, on=['antecedents', 'consequents'], how='outer', suffixes=['_SPN', '_Apriori'])
rule_comparison.set_index(['antecedents', 'consequents'], inplace=True)

for col in rule_comparison.columns:
    if '_SPN' in col:
        metric = col.split('_SPN')[0]
        rule_comparison[metric + '_Diff'] = rule_comparison[col] - rule_comparison[metric + '_Apriori']
rule_comparison = rule_comparison.reset_index()
# print(rule_comparison.std(axis=0))
ordered_cols = ['antecedents', 'consequents']
for  metric in list(set(spn_rules.columns) - set(['antecedents', 'consequents'])):
    ordered_cols = ordered_cols + [metric+'_SPN', metric+'_Apriori', metric+'_Diff']
rule_comparison = rule_comparison.reindex(ordered_cols, axis=1)

print(rule_comparison)