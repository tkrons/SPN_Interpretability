from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
from mlxtend.preprocessing import TransactionEncoder
transaction_encoder = TransactionEncoder()
from spn_apriori.spn_based_apriori import spn_apriori

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
# from hanging_threads import start_monitoring
import warnings
import pandas as pd
import numpy as np
from itertools import compress

from data import real_data
from data import synthetic_data
from simple_spn import spn_handler
from spn.structure.leaves.parametric.Parametric import Categorical
from spn_apriori.itemsets_utils import perf_comparison, fpgrowth_wrapper_orange, support_of_set, spn_support_of_set
import simple_spn.functions as fn

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

def get_error_totals(df, min_sup, errors):
    res = []
    # df with only excess sets calculated afterwards, missing sets missing in error calculation
    PRED = df[df.support_pred >= min_sup]
    for e in errors:
        if 'AE' == e:
            res.append(PRED.difference.abs().sum())
        elif 'MAE' == e:
            res.append(PRED.difference.abs().mean())
        elif 'SE' == e:
            res.append((PRED.difference ** 2).sum())
        elif 'MSE' == e:
            res.append((PRED.difference ** 2).mean())
        elif 'MRE' == e: #todo check case df.support = 0 div by zero
            res.append((PRED.difference / df.support).replace([np.inf, -np.inf], np.nan).abs().mean())
        elif 'Missing Sets' == e: #percentage of predictions? (but missing sets not in total predictions!)
            res.append(df.loc[df.support_pred < min_sup, 'itemsets'].count()) #/ df.loc[df.support_pred >= min_sup, 'itemsets'].count())
        elif 'Excess Sets' == e: #percentage of predictions
            res.append(df.loc[df.support < min_sup, 'itemsets'].count()) # / df.loc[df.support_pred >= min_sup, 'itemsets'].count())
        elif 'Number of Sets' == e:
            res.append(len(PRED))
        else:
            raise ValueError('Unknown Error Type: {}'.format(e))
    return res

def calc_itemsets_df(train, spn, min_sup, test = None, value_dict=None, test_use ='apriori', train_use ='SPN'):
    '''
    Gets missing and excess itemsets ! therefore support_pred and support can be less min_sup
    use itemsets.loc[itemsets.support_pred >= min_sup] if you need only predicted itemsets
    :param train:
    :param spn:
    :param min_sup:
    :param test: test data or None for test=train
    :param value_dict:
    :param test_use:
    :param train_use:
    :return: all itemsets either algorithm found
    '''
    # if isinstance(PRED_GT, list):
    #     train, test = PRED_GT[0], PRED_GT[1]
    # elif isinstance(PRED_GT, pd.DataFrame): #all the same
    #     train, test, whole_df = PRED_GT, PRED_GT, PRED_GT
    if test is None:
        test, whole_df = train, train
    else: #not none
        whole_df = pd.concat([train, test], ignore_index=True)


    print('==================== Calculating Itemsets ==============')
    if train_use == 'SPN':
        PRED = spn_apriori(train, low_memory=False, min_support=min_sup, spn=spn, value_dict=value_dict, use_colnames=True, )
    elif train_use == 'apriori':
        # mlxtend fpgrowth completely equal
        PRED = mlxtend_fpgrowth(train, min_sup, use_colnames=True)
        # PRED = apriori(train, min_sup, use_colnames=True)
    if test_use == 'SPN':
        GT = spn_apriori(test, min_support=min_sup, spn=spn, value_dict=value_dict, use_colnames=True)
    elif test_use == 'apriori':
        # GT = fpgrowth_wrapper_orange(GT_df, min_sup)
        GT = apriori(test, min_sup, use_colnames=True)
    # GT = mlxtend_apriori(transactional_df, min_support=min_sup, use_colnames=True, )
    PRED = PRED.sort_values('support', ascending=False).rename(
                columns={'support': 'support_pred'}).set_index('itemsets')
    GT = GT.sort_values('support', ascending=False).set_index('itemsets')

    itemsets = PRED.join(how='outer', other=GT, ).reset_index()
    # short DRY methods
    def _catch_up_apriori(transactional_df, itemsets, col_name):
        """col_name: either support or support_pred, defines the column in which to fill up NaNs"""
        itemsets.loc[itemsets[col_name].isna(), col_name] = itemsets.loc[
            itemsets[col_name].isna(),
            'itemsets'
        ].apply(lambda s: support_of_set(transactional_df, s))
        return itemsets
    def _catch_up_SPN(spn, itemsets, value_dict, col_name):
        itemsets.loc[itemsets[col_name].isna(), col_name] = itemsets.loc[
            itemsets[col_name].isna(),
            'itemsets'
        ].apply(lambda s: spn_support_of_set(spn, s, value_dict))
        return itemsets

    # get the support of excess sets in PRED (missing sets in GT)
    if test_use == 'apriori':
        itemsets = _catch_up_apriori(whole_df, itemsets, 'support')
    elif test_use == 'SPN':
        itemsets = _catch_up_SPN(spn, itemsets, value_dict, 'support')
    #get the support of missing sets in PRED
    if train_use == 'apriori':
        itemsets = _catch_up_apriori(whole_df, itemsets, 'support_pred')
    elif train_use == 'SPN':
        itemsets = _catch_up_SPN(spn, itemsets, value_dict, 'support_pred')
    itemsets['support_mean'] = itemsets[['support_pred', 'support']].mean(axis=1)
    itemsets['difference'] = itemsets.apply(lambda x: np.round(x['support_pred'] - x['support'], 4), axis=1)
    itemsets['difference_percent'] = itemsets.apply(lambda x: np.round((x['support_pred'] - x['support']) / x['support_pred'], 4), axis=1)
    itemsets['length'] = itemsets['itemsets'].apply(lambda x: len(x))
    # itemsets = itemsets[itemsets['length'] >= 2]
    itemsets = itemsets.sort_values('support_mean', ascending=False)

    # print(itemsets.to_string())

    print('Num. of Sets:\t{}'.format(len(itemsets)))
    return itemsets


def spn_hyperparam_opt(df, test_frac = 0.5):
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
            if recalc_spn or not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
                print("======================== Creating SPN ... ===============")
                parametric_types = [Categorical for _ in train.columns]
                # Creates the SPN and saves to a file
                spn_handler.create_parametric_spns(train.values, parametric_types, dataset_name,
                                                   rdc_thresholds=[rdc_threshold],
                                                   min_instances_slices=[min_instances_slice],
                                                   silence_warnings=True)
            spn, _, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
            num_nodes = len(get_nodes_by_type(spn, Node))
            row['num_nodes'] = num_nodes
            #todo why is there a test in calc_itemsets
            raise ValueError('why is there a test in calc_itemsets')
            error_values = get_error_totals(calc_itemsets_df(test, spn, min_sup, value_dict), min_sup, errors=error_types)
            for e_name, e_val in zip(error_types, error_values):
                row[e_name] = e_val
            rows.append(row)
    spn_hyperparam_results = pd.DataFrame(rows)
    spn_hyperparam_results.sort_values(by=['rdc_threshold', 'min_instances_slice'], inplace=True)
    return spn_hyperparam_results

def scatter_plots(itemsets,): #todo scatter generalized (spn_test etc)
    # using ALL itemsets, missing and excess
    #polynomial regression to check trend of the difference
    regX = np.c_[itemsets['support_pred'].values, #(both['support'] ** 2).values
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
    # plt.scatter(both['support_pred'], both['difference'].abs(), color='blue', label='SPN', marker='o', s=6)
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

    # diagonal scatterplot y = spn_ap x = normal_ap
    #todo 'reverse' log to visualize whole value range
    # https://stackoverflow.com/questions/5395554/custom-axis-scales-reverse-logarithmic
    fig, ax = plt.subplots()
    ax.set_xlim([0.005, 0.05])
    ax.set_ylim([0.005, 0.05])
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", zorder=0)
    plt.scatter(itemsets['support'], itemsets['support_pred'], s = 2, marker='.', zorder=1)
    plt.xlabel('support')
    plt.ylabel('support_pred')
    plt.title('SPN_apriori support and actual support')
    plt.tight_layout()
    plt.savefig('../../_figures/{}.pdf'.format('scatter_support_deviation'))
    plt.show()

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

def cross_eval(transactional_df, dataset_name, min_sup_steps, value_dict,
               recalc_spn = False, rdc_threshold = 0.1, min_instances_slice = 0.05):
    print('================= Cross Eval =====================')
    #1 apriori_train
    #2 apriori_test
    #3 SPN train
    # calc: 1v1 (=0), 3v1 (generalization error), 1v2 (GT difference betweeen train/test)
    # 3v2 (does the SPN generalize apriori?, compare with 1v2)
    train, test = train_test_split(transactional_df, test_size=0.5, random_state=None) # rstate = 100 for reproducability
    if recalc_spn or not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
        print("======================== Creating SPN ... ===============")
        parametric_types = [Categorical for _ in train.columns]
        spn_handler.create_parametric_spns(train.values, parametric_types, dataset_name, value_dict=value_dict,
                                           rdc_thresholds=[rdc_threshold],
                                           min_instances_slices=[min_instances_slice])
    spn_train, _, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)

    rows, error_names = [], ['AE', 'MAE', 'MRE', 'Missing Sets', 'Excess Sets', 'Number of Sets']
    for min_sup_eval in min_sup_steps:
        # one_v_one = get_error_totals(calc_itemsets_df(train, spn_train, min_sup_eval, GT_use='apriori', PRED_use='apriori'),
        #                              min_sup=min_sup_eval)
        one_v_two = calc_itemsets_df(train, spn_train, min_sup_eval, test=test, test_use='apriori', train_use='apriori')
        one_v_three = calc_itemsets_df(train, spn_train, min_sup_eval, value_dict=value_dict)
        three_v_two = calc_itemsets_df(train, spn_train, min_sup_eval, test, value_dict=value_dict,)
        results = {
            ind: get_error_totals(df, min_sup_eval, error_names) for ind, df in
            {'train_vs_test': one_v_two, 'spn_vs_train': one_v_three, 'spn_vs_test': three_v_two}.items()
        }
        for ind, errors in results.items():
            d = dict(zip(error_names, errors))
            d.update({'compare': ind,  'min_sup': min_sup_eval,})
            rows.append(d)

    evals = pd.DataFrame(data = rows, ).set_index(['min_sup', 'compare'])
    return evals

if __name__ == '__main__':
    ## PARAMETERS ##
    dataset_name = "UCI" #todo fix synthetic data: AttributeError: 'float' object has no attribute 'item'
    only_n_rows = None
    min_sup = 0.01
    # min_sup = 0 oder nah an null lässt PC einfrieren..
    rdc_threshold, min_instances_slice = 0.1, 0.01
    recalc_spn = False
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
        hyperparam_results = spn_hyperparam_opt(transactional_df)

    # includes excess and missing itemsets!!
    all_itemsets = calc_itemsets_df(transactional_df, spn, min_sup, value_dict=value_dict)
    scatter_plots(all_itemsets)

    evals = cross_eval(df, dataset_name, [0.01, 0.03, 0.05, 0.1, 0.2, 0.4], value_dict, recalc_spn=recalc_spn)
    print(evals.to_string())



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
    plot_error_brackets(all_itemsets, ['AE', 'MAE', 'MRE'])

    print('AE: {}\tMAE: {}\tSE: {}\tMSE: {}'.format(totalAE, totalMAE, totalSE, totalMSE))


    print('================ Calculating Rules and Metrics ===============')
    spn_apriori_df = all_itemsets.reset_index()[['itemsets', 'support_pred']]
    normal_apriori_df = all_itemsets.reset_index()[['itemsets', 'support']]
    spn_rules = association_rules(spn_apriori_df.rename(columns={'support_pred': 'support'}),
                                  metric='confidence', min_threshold=0.80)
    normal_apriori_rules = association_rules(normal_apriori_df,
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
    rule_comparison.sort_values(by='lift_SPN', ascending=False, inplace=True)
    print(rule_comparison.sort_values(by='lift_SPN', ascending=False).head(10).to_string())
    print(rule_comparison.mean())