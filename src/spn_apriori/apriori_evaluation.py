from mlxtend.frequent_patterns import association_rules as mlxtend_association_rules
from mlxtend.preprocessing import TransactionEncoder
transaction_encoder = TransactionEncoder()
from spn_apriori.spn_based_apriori import spn_apriori

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
# from hanging_threads import start_monitoring
import warnings
import pandas as pd
import numpy as np
from itertools import compress

from data import real_data
from simple_spn import spn_handler
from spn.structure.leaves.parametric.Parametric import Categorical
from spn_apriori.itemsets_utils import perf_comparison, fpgrowth_wrapper

'''
results: bigger support: SPN_apriori approaches normal apriori. smaller generalization error
support < 0.3: spn.prob < apriori
support > 0.3: spn.prob ~= apriori

#wovon hängt differenz ab? warum werden sets um 0.3 vom SPN niedriger geschätzt?

#Performance 
synthetic(n=200):
Mean time SPN: 0.011837113818172354
Mean time normal apriori: 0.005772764722472174
synthetic(n=5000):
Mean time SPN: 0.014125372747143788
Mean time normal apriori: 0.01017157568628354
UCI (n=32000+, minsup=10%) SPN faster !!
Mean time SPN: 0.6654770773333333
Mean time normal apriori: 0.7643690240000002
UCI (n=32000+, minsup=0.1%) SPN only slightly slower 
Mean time SPN: 6.742353120878661
Mean time normal apriori: 6.450652480835174

Std. by support brackets UCI n=32000:
0<x<0.01:	nan
0.01<x<0.1:	0.006511
0.1<x<0.2:	0.0105208
0.2<x<0.5:	0.016204
0.5<x<1:	0.0210718
'''

def get_error_totals(df, errors):
    res = []
    for e in errors:
        if 'AE' == e:
            res.append(df.difference.abs().sum())
        elif 'MAE' == e:
            res.append(df.difference.abs().mean())
        elif 'SE' == e:
            res.append((df.difference ** 2).sum())
        elif 'MSE' == e:
            res.append((df.difference ** 2).mean())
        elif 'MRE' == e:
            res.append((df.difference / df.support).abs().mean())
        else:
            raise ValueError('Unknown Error Type: {}'.format(e))
    return res


def calc_itemsets_df(transactional_df, spn, min_sup, value_dict)   :
    print('==================== Calculating Itemsets ==============')
    spn_apriori_df = spn_apriori(transactional_df, low_memory=False, min_support=min_sup, spn=spn, value_dict=value_dict, use_colnames=True, )
    # normal_apriori_df = mlxtend_apriori(transactional_df, min_support=min_sup, use_colnames=True, )
    normal_apriori_df = fpgrowth_wrapper(transactional_df, min_sup)
    spn_apriori_df = spn_apriori_df.sort_values('support', ascending=False).reset_index(drop=True).rename(
                columns={'support': 'spn.prob'}).set_index('itemsets')
    normal_apriori_df = normal_apriori_df.sort_values('support', ascending=False).reset_index(drop=True).set_index('itemsets')
    # Indexing does not work on frozenset
    itemsets = spn_apriori_df.join(how='outer', other=normal_apriori_df, )

    itemsets['support_mean'] = itemsets[['spn.prob', 'support']].mean(axis=1)
    itemsets['difference'] = itemsets.apply(lambda x: np.round(x['spn.prob'] - x['support'], 4), axis=1)
    itemsets['difference_percent'] = itemsets.apply(lambda x: np.round((x['spn.prob'] - x['support']) / x['spn.prob'], 4), axis=1)
    itemsets['length'] = itemsets.index.to_series().apply(lambda x: len(x))
    # itemsets = itemsets[itemsets['length'] >= 2]
    itemsets = itemsets.sort_values('support_mean', ascending=False)

    # print(itemsets.to_string())

    #only keep sets we got from both algorithms. #todo replace with get missing ones manually
    itemsets = itemsets.dropna(subset=['spn.prob', 'support'])
    print('Num. of Sets:\t{}'.format(len(itemsets)))
    return itemsets


## PARAMETERS ##
dataset_name = "UCI"
only_n_rows = None
min_sup = 0.01
# min_sup = 0 oder nah an null lässt PC einfrieren..
recalc_spn = False
benchmark = False
spn_hyperparam_grid_search = False
# bug spn.prob=1.0 @ seed(893) minsup10% 50 rows
# seed = np.random.randint(1000)
seed = 755
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
    sample_indices = np.random.random_integers(0, len(df), only_n_rows)
    df = df.iloc[sample_indices]
    transactional_df = transactional_df.iloc[sample_indices]

# transactional overview
transactions = transactional_df.apply(lambda x: set(compress(x.index, x)) or 'empty transaction', axis=1)
transactions = transactions[transactions != 'empty transaction']
print('Transactions:\n', transactions.sample(3))
itemlist = list(set().union(*transactions.values))
print('Items:\n{}'.format(itemlist))
print('Number of Items: {} Number of Transactions: {}'.format(len(itemlist), len(transactions)))

#SPN generation #todo hyperparam optimization/test
rdc_threshold, min_instances_slice = 0.1, 0.05
if recalc_spn or not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
    print("======================== Creating SPN ... ===============")
    parametric_types = [Categorical for _ in df.columns]
    # Creates the SPN and saves to a file
    spn_handler.create_parametric_spns(df.values, parametric_types, dataset_name,
                                       rdc_thresholds=[rdc_threshold], min_instances_slices=[min_instances_slice])

# Load SPN
spn, _, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)

if benchmark:
    print('============= Benchmark ==================================')
    perf_comparison(transactional_df, min_sup, spn, value_dict)

if spn_hyperparam_grid_search:
    print('============= SPN Hyperparameter Optimization ================')
    error_types = ['AE', 'MAE', 'MRE']
    for i in range(20):
        rdc_threshold, min_instances_slice = 0.1, 0.05
        if recalc_spn or not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
            print("======================== Creating SPN ... ===============")
            parametric_types = [Categorical for _ in df.columns]
            # Creates the SPN and saves to a file
            spn_handler.create_parametric_spns(df.values, parametric_types, dataset_name,
                                               rdc_thresholds=[rdc_threshold],
                                               min_instances_slices=[min_instances_slice])
        spn, _, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
        get_error_totals(calc_itemsets_df(transactional_df, spn, min_sup, value_dict))

itemsets = calc_itemsets_df(transactional_df, spn, min_sup, value_dict)
#polynomial regression to check trend of the difference
regX = np.c_[itemsets['spn.prob'].values, #(both['support'] ** 2).values
]
regy = itemsets['difference'].values.reshape(-1, 1)
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
plt.scatter(itemsets['support'], itemsets['difference'], color='blue', label='SPN', marker='o', s=6)
plt.plot(np.linspace(0, 1, 100), np.zeros([100, 1]), color='black', ) # 0 line
plt.plot(linearspace[:, 0], reg_yhat, color='red')
xy = [itemsets['support'].values, itemsets['difference'].values]
plt.xlim(0, xy[0].max().item() + 0.05)
# plt.ylim(-0.05, 0.05)
plt.xlabel('support')
plt.ylabel('support_SPN - support_Apriori (difference)')
# both = both.drop(columns=['support_mean'])
plt.show()

#todo fix spn.prob = 1.0 Bug. for values in valuedict if not occuring in dataset
print('itemsets with the biggest difference:')
# print(both.sort_values('difference', ascending=False).head(4).to_string())
print(itemsets.reindex(itemsets['difference'].abs().sort_values(ascending=False).index).head(4).to_string())
print('itemsets with the biggest percentage difference:')
print(itemsets.reindex(itemsets['difference_percent'].abs().sort_values(ascending=False).index).head(4).to_string())
print(itemsets.sort_values('difference_percent', ascending=False).head(4).to_string())

# bracket wise analysis
print('Std. by support brackets:')
# brackets = [i for i in np.linspace(0, 1, 11,) if i < both.support.max()]

print('=============== Error Metrics ==================')
with warnings.catch_warnings():
    warnings.simplefilter('error')
    # function_raising_warning()

# brackets = list(np.linspace(0, 1, 11,))
brackets = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
for i in range(1, len(brackets)):
    std = np.round(itemsets[ itemsets['support_mean'].between(brackets[i - 1], brackets[i])].difference.std(), 7)
    # print('{}<x<{}:\t{}'.format(brackets[i-1], brackets[i], std), )
AE, MAE, SE, MSE, MRE, num_sets = [], [], [], [], [], []
for i in range(1, len(brackets)):
    #differing num
    set_range = itemsets[itemsets['support'].between(brackets[i - 1], brackets[i])]
    diff =  set_range.difference * len(transactional_df)
    num_sets.append(len(diff))
    # if len(diff) == 0:
    #     diff = np.NaN
    AE.append(diff.abs().sum())
    MAE.append(diff.abs().mean())
    SE.append((diff ** 2).sum())
    MSE.append((diff ** 2).mean())
    MRE.append((diff / itemsets.support).abs().mean())
    # print('{}<x<{}:\t{}'.format(brackets[i - 1], brackets[i], abs_err), )
totalAE, totalMAE, totalSE, totalMSE, totalMRE = get_error_totals(itemsets, ['AE', 'MAE', 'SE', 'MSE', 'MRE'])
# plt.yscale('log')
brackets_str = ['[{}, {}] '.format(str(np.round(brackets[i-1], 2)),
                                   str(np.round(brackets[i], 2)))
                for i,_ in enumerate(num_sets, start=1)]

def plot_Error(E, total, name,  brackets_str, ylog=False):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(brackets_str, E,)
    if ylog:
        plt.yscale('symlog')
        plt.ylim(bottom=10)
    plt.xticks(brackets_str, rotation='vertical')
    for i, rect in enumerate(bars.patches):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 5, '({})'.format(num_sets[i]),
                ha='center', va='bottom', rotation='horizontal')
    # plt.annotate('number of itemsets in brackets', xy=(0.95, 0.95), xycoords='figure fraction')
    plt.title('{} on all itemsets: {}'.format(name, total))
    plt.ylabel(name)
    plt.xlabel('buckets of itemsets')
    plt.tight_layout()
    plt.savefig('../../_figures/{}.pdf'.format(name), bboxinches='tight', padinches=0)
    plt.show()

# with warnings.catch_warnings():
#     warnings.simplefilter('error')
# plot_Error(AE, totalAE, 'AE', brackets_str, ylog=True)
plot_Error(AE, totalAE, 'AE', brackets_str)
plot_Error(MAE, totalMAE, 'MAE', brackets_str)
# plot_Error(SE, totalSE, 'SE', brackets_str)
plot_Error(MRE, totalMRE, 'MRE', brackets_str)
# plot_Error(MSE, totalMSE, 'MSE', brackets_str)

print('AE: {}\tMAE: {}\tSE: {}\tMSE: {}'.format(totalAE, totalMAE, totalSE, totalMSE))


print('================ Calculating Rules and Metrics ===============')
print('TODO FIX RULES!') #todo
spn_apriori_df = itemsets.reset_index()[['itemsets', 'spn.prob']]
normal_apriori_df = itemsets.reset_index()[['itemsets', 'support']]
spn_rules = mlxtend_association_rules(spn_apriori_df.rename(columns={'spn.prob': 'support'}).reset_index(),
                              metric='confidence', min_threshold=0.80)
normal_apriori_rules = mlxtend_association_rules(normal_apriori_df.reset_index(),
                                                 metric='confidence', min_threshold=0.80)

#======= Analyze resulting rule quality
rule_comparison = pd.merge(spn_rules, normal_apriori_rules, on=['antecedents', 'consequents'], how='outer', suffixes=['_SPN', '_Apriori'])
rule_comparison.set_index(['antecedents', 'consequents'], inplace=True)
rule_comparison.dropna(inplace=True)

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
print(rule_comparison.columns)
print(rule_comparison.sort_values(by='lift_SPN', ascending=False).head(10).to_string())
print(rule_comparison.mean())