from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
transaction_encoder = TransactionEncoder()

from sklearn.model_selection import train_test_split
# from hanging_threads import start_monitoring
import pandas as pd
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt

from data import real_data
from data import synthetic_data
from simple_spn import spn_handler
from spn.structure.leaves.parametric.Parametric import Categorical
from spn_apriori.itemsets_utils import perf_comparison, scatter_plots, cross_eval, get_error_totals, calc_itemsets_df

'''
results: bigger support: SPN_apriori approaches normal apriori. smaller generalization error
support < 0.3: support_pred < apriori
support > 0.3: support_pred ~= apriori

#wovon hängt differenz ab? warum werden sets um 0.3 vom SPN niedriger geschätzt?

"zero itemsets" frequently have support >0 in SPN

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

Rules:
 - Too high confidence, might imply trivial (married-civ-spouse, female -> wife), uninteresting relationship
 - (only in case of highly correlated variables)
 
'''


def spn_hyperparam_opt(df, value_dict, test_frac = 0.5):
    print('============= SPN Hyperparameter Optimization ================')
    error_types = ['AE', 'MAE', 'MRE']
    rows = []
    train, test = train_test_split(df, test_size = test_frac, random_state=100)
    dataset_name = 'UCI_half'
    np.random.seed(5)
    from spn.structure.Base import get_nodes_by_type, Node
    #rdc, mis = [0.1, 0.2, 0.3], [0.001, 0.01, 0.1]
    rdc, mis = np.linspace(0., 0.7, 5), np.linspace(0., 0.5, 5)
    for rdc_threshold in rdc:
        for min_instances_slice in mis:
        # for i in range(100):
            # rdc_threshold, min_instances_slice = np.random.uniform(0., 0.7), np.random.uniform(0., 0.5)
            row = {'rdc_threshold': rdc_threshold, 'min_instances_slice': min_instances_slice}
            if True or not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
                print("======================== Creating SPN ... ===============")
                parametric_types = [Categorical for _ in train.columns]
                # Creates the SPN and saves to a file
                spn_handler.create_parametric_spns(train.values, parametric_types, dataset_name, value_dict=value_dict,
                                                   rdc_thresholds=[rdc_threshold],
                                                   min_instances_slices=[min_instances_slice],
                                                   silence_warnings=True)
            spn, value_dict, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
            num_nodes = len(get_nodes_by_type(spn, Node))
            row['num_nodes'] = num_nodes
            error_values = get_error_totals(calc_itemsets_df(train, spn, min_sup, test=test, value_dict=value_dict), min_sup, errors=error_types)
            for e_name, e_val in zip(error_types, error_values):
                row[e_name] = e_val
            rows.append(row)
    spn_hyperparam_results = pd.DataFrame(rows)
    spn_hyperparam_results.sort_values(by=['rdc_threshold', 'min_instances_slice'], inplace=True)
    return spn_hyperparam_results


def plot_error_brackets(itemsets, error_names, ylog=False): #todo discuss MRE = +inf problem and fix, plots are bad
    # bracket wise analysis
    # brackets = [i for i in np.linspace(0, 1, 11,) if i < both.support.max()]
    print('=============== Error Metrics ==================')
    # brackets = list(np.linspace(0, 1, 11,))
    brackets = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.9, 1.]
    PRED = itemsets.loc[itemsets.support_pred >= min_sup]
    for i in range(1, len(brackets)):
        std = np.round(PRED[PRED['support_mean'].between(brackets[i - 1], brackets[i])].difference.std(), 7)
        # print('{}<x<{}:\t{}'.format(brackets[i-1], brackets[i], std), )
    errors = {name: [] for name in error_names}
    num_sets = []
    for i in range(1, len(brackets)):
        # differing num
        set_range = PRED[PRED['support_pred'].between(brackets[i - 1], brackets[i])]
        diff = set_range.difference * len(transactional_df)
        num_sets.append(len(diff))
        # if len(diff) == 0:
        #     diff = np.NaN
        if 'AE' in errors:
            errors['AE'].append(diff.abs().sum())
        if 'MAE' in errors:
            errors['MAE'].append(diff.abs().mean())
        if 'SE' in errors:
            errors['SE'].append((diff ** 2).sum())
        if 'MSE' in errors:
            errors['MSE'].append((diff ** 2).mean())
        if 'MRE' in errors:
            errors['MRE'].append((diff / PRED.support).abs().mean())
        # print('{}<x<{}:\t{}'.format(brackets[i - 1], brackets[i], abs_err), )
    # plt.yscale('log')
    brackets_str = ['[{}, {}] '.format(str(np.round(brackets[i - 1], 2)),
                                       str(np.round(brackets[i], 2)))
                    for i, _ in enumerate(num_sets, start=1)]
    for name, E in errors.items():
        total = get_error_totals(PRED, min_sup=min_sup, errors=[name])[0]
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

if __name__ == '__main__':
    ## PARAMETERS ##
    dataset_name = "play_store"
    only_n_rows = None
    min_sup = 0.01
    # min_sup = 0 oder nah an null lässt PC einfrieren..
    rdc_threshold, min_instances_slice = 0.1, 0.01
    recalc_spn = True
    benchmark = False
    spn_hyperparam_grid_search = False
    # bug support_pred=1.0 @ seed(893) minsup10% 50 rows
    # seed = np.random.randint(1000)
    seed = 755
    print('Seed: {}'.format(seed))
    np.random.seed(seed)

    #transactional df and df are for distinction between tabular and transactional data,
    # incase we want to use a tabular SPN, else remove? Makes it much more complicated
    #when modifying the dataset, dont forget to recalculate the spn!
    if dataset_name == 'apriori_test':
        df, value_dict, parametric_types = synthetic_data.generate_simple_transactions(only_n_rows)
        transactional_df = df  # equal in this case
    elif dataset_name == 'UCI':
        df, value_dict, parametric_types = real_data.get_adult_41_items()
        transactional_df = df # equal in this case
    elif dataset_name == 'UCI_tabular':
        df, value_dict, parametric_types = real_data.get_adult_41_items(convert_tabular=True)
        #get transactions for normal apriori
        transactional_df, _, _ = real_data.get_adult_41_items()
    elif dataset_name == 'play_store':
        df, value_dict, parametric_types = real_data.get_play_store()
        transactional_df = df

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

    #SPN generation
    if recalc_spn or not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
        print("======================== Creating SPN ... ===============")
        parametric_types = [Categorical for _ in df.columns]
        # Creates the SPN and saves to a file
        spn_handler.create_parametric_spns(df.values, parametric_types, dataset_name, value_dict=value_dict,
                                           rdc_thresholds=[rdc_threshold],
                                           min_instances_slices=[min_instances_slice],
                                           silence_warnings=True)

    # Load SPN
    spn, _, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)

    if benchmark:
        perf_comparison(transactional_df, min_sup, spn, value_dict)

    if spn_hyperparam_grid_search:
        hyperparam_results = spn_hyperparam_opt(transactional_df, value_dict)
        print(hyperparam_results)

    # includes excess and missing itemsets!!
    all_itemsets = calc_itemsets_df(transactional_df, spn, min_sup, value_dict=value_dict)
    scatter_plots(all_itemsets)

    # evals = cross_eval(df, dataset_name, [0.01, 0.03, 0.05, 0.1, 0.2, 0.4], value_dict, recalc_spn=recalc_spn)
    # print(evals.to_string())

    #todo fix support_pred = 1.0 Bug. for values in valuedict if not occuring in dataset
    print('itemsets with the biggest difference:')
    # print(both.sort_values('difference', ascending=False).head(4).to_string())
    print(all_itemsets.reindex(all_itemsets['difference'].abs().sort_values(ascending=False).index).head(4).to_string())
    print('itemsets with the biggest percentage difference:')
    print(all_itemsets.reindex(all_itemsets['difference_percent'].abs().sort_values(ascending=False).index).head(4).to_string())
    print(all_itemsets.sort_values('difference_percent', ascending=False).head(4).to_string())

    totalAE, totalMAE, totalSE, totalMSE, totalMRE = get_error_totals(all_itemsets, min_sup, ['AE', 'MAE', 'SE', 'MSE', 'MRE'])


    # with warnings.catch_warnings():
    #     warnings.simplefilter('error')
    plot_error_brackets(all_itemsets, ['AE', 'MAE', 'MRE'],)

    print('AE: {}\tMAE: {}\tSE: {}\tMSE: {}'.format(totalAE, totalMAE, totalSE, totalMSE))
