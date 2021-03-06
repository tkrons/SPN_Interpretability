from time import perf_counter

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from mlxtend.frequent_patterns import apriori as mlxtend_apriori, fpgrowth as mlxtend_fpgrowth, apriori
from mlxtend.frequent_patterns import association_rules
from orangecontrib.associate import frequent_itemsets as orange_fpgrowth
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from spn.structure.leaves.parametric.Parametric import Categorical

from simple_spn import spn_handler

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

def simple_interpretable_rules(itemsets, top=None):
    return _get_interpretable_best_lift_rules(association_rules(itemsets, metric='confidence', min_threshold=0.70), top = top)

def _get_interpretable_best_lift_rules(rules, top, suffix=''):
    '''
    :param rule_: needs cols: [antecedent, consequent, lift]
    :param suffix: suffix of the metric columns. If you have a df with both SPN and apriori metrics, you'll need this
    to specify which metrics to use for rule extraction. Default is empty string ('')
    :return:
    '''
    assert rules['confidence' + suffix].min() >= 0.5
    lift_s = rules['lift' + suffix]
    # unit variance normalized
    rules['norm_lift'] = (lift_s-lift_s.min()) / lift_s.std()
    def _rule_basic_score(row):
        len_ant, len_cons = len(row['antecedents']), len(row['consequents'])
        if len_cons != 1.:
            return 0. #dont want longer consequent for now
        conf = row['confidence' + suffix]
        lift = row['norm_lift']
        X = [conf, lift, 1/len_ant]
        W = [0.2, 0.4, 0.4]
        return np.dot(X, W)

    rules['score'] = rules.apply(_rule_basic_score, axis=1)
    rules = rules.sort_values('score', ascending=False,)
    return rules.head(top)


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


def difference_plot(itemsets, fname=None, reg_line=True, dataset_name=None):
    '''
    :param itemsets: itemsets with support (GT) and support_pred (PRED)
    :param GT_name: type of GT data, either 'test' or 'train'
    :return: None
    '''
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
    plt.scatter(itemsets['support'], itemsets['difference'], color='blue', label='SPN', marker='.', s=3, alpha=0.5)
    plt.plot(np.linspace(0, 1, 100), np.zeros([100, 1]), color='black', ) # 0 line
    if reg_line:
        plt.plot(linearspace[:, 0], reg_yhat, color='red')
    xy = [itemsets['support'].values, itemsets['difference'].values]
    plt.xlim(0, float(xy[0].max()) + 0.05)
    # ymax = itemsets.difference.abs().max()
    # plt.ylim(-ymax - 0.01, ymax + 0.01)
    plt.ylim(-0.02, 0.02)
    # plt.yscale('symlog')
    plt.xlabel('support')
    plt.ylabel('residual (support - support_pred)')
    if fname:
        # plt.title(fname.split('.pdf')[0])
        path = '../../_figures/' + dataset_name
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/residuals_' + fname)
    # both = both.drop(columns=['support_mean'])
    # plt.savefig('../../_figures/{}'.format('difference_scatter_' + fname))
    plt.show()

def diagonal_support_scatter(itemsets, fname=None, dataset_name=None):
    # diagonal scatterplot y = spn_ap x = normal_ap
    # https://stackoverflow.com/questions/5395554/custom-axis-scales-reverse-logarithmic
    fig, ax = plt.subplots()
    def forward(x):
        r = np.log(np.log(x) + 10)
        r[np.isneginf(r)] = 0
        return r

    def backward(x):
        # if x <= 0:
        #     return 0
        return np.exp(np.exp(x) - 10)
    if itemsets.support.min() < 0.05:
        ax.set_xlim([0.005, 0.05])
        ax.set_ylim([0.005, 0.05])
    else:
        # min sup too small
        return -1
    xymax = itemsets[['support', 'support_pred']].max().max()
    # ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", zorder=0)
    plt.scatter(itemsets['support'], itemsets['support_pred'], s = 2, marker='.', zorder=1)
    # ax.set_yscale('function', functions=((lambda x: np.log(x)+10,
    #                                       lambda x: np.exp(x - 10))))
    # ax.set_xscale('function', functions=((lambda x: np.log(x)+10,
    #                                       lambda x: np.exp(x - 10))))
    # ax.set_xscale('function', functions = (forward, backward))
    # ax.set_yscale('function', functions=(forward, backward))
    # ax.set_xlim([0., xymax+0.03])
    # ax.set_ylim([0., xymax+0.03])

    plt.xlabel('support')
    plt.ylabel('support_pred')
    # plt.tight_layout()
    if fname:
        plt.title(fname.split('.pdf')[0])
        path = '../../_figures/' + dataset_name
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/residuals_' + fname)
    plt.show()



def scatter_plots(itemsets, fname=None, reg_line=True, dataset_name=None):
    '''wrapper for multiple evaluation plots'''
    difference_plot(itemsets, fname, reg_line, dataset_name=dataset_name)
    diagonal_support_scatter(itemsets, fname, dataset_name=dataset_name)

def cross_eval(transactional_df, dataset_name, min_sup_steps, value_dict,
               recalc_spn = False, rdc_threshold = 0.1, min_instances_slice = 0.05):
    print('================= Cross Eval =====================')
    #1 apriori_train
    #2 apriori_test
    #3 SPN train
    # calc: 1v1 (=0), 3v1 (generalization error), 1v2 (GT difference betweeen train/test)
    # 3v2 (does the SPN generalize apriori?, compare with 1v2)
    train, test = train_test_split(transactional_df, test_size=0.5, random_state=100) # rstate = 100 for reproducability
    if recalc_spn or not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
        print("======================== Creating SPN ... ===============")
        parametric_types = [Categorical for _ in train.columns]
        spn_handler.create_parametric_spns(train.values, parametric_types, dataset_name, value_dict=value_dict,
                                           rdc_thresholds=[rdc_threshold],
                                           min_instances_slices=[min_instances_slice])
    spn_train, _, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
    print('Num. nodes: {}'.format(fn.get_num_nodes(spn_train)))

    rows, error_names = [], ['AE', 'MAE', 'MRE', 'Missing Sets', 'Excess Sets', 'Number of Sets']
    for min_sup_eval in min_sup_steps:
        # one_v_one = get_error_totals(calc_itemsets_df(train, spn_train, min_sup_eval, GT_use='apriori', PRED_use='apriori'),
        #                              min_sup=min_sup_eval)
        one_v_two = calc_itemsets_df(train, spn_train, min_sup_eval, test=test, test_use='apriori', train_use='apriori')
        three_v_one = calc_itemsets_df(train, spn_train, min_sup_eval, value_dict=value_dict)
        three_v_two = calc_itemsets_df(train, spn_train, min_sup_eval, test, value_dict=value_dict,)

        if min_sup_eval == min(min_sup_steps):
            # do scatter plots for spn_vs_train and spn_vs_test
            scatter_plots(one_v_two, 'train_vs_test.pdf', reg_line=False, dataset_name=dataset_name)
            scatter_plots(three_v_one, 'rdc={}_mis={}_GT=train.pdf'.format(rdc_threshold,min_instances_slice), reg_line=False, dataset_name=dataset_name)
            scatter_plots(three_v_two, 'rdc={}_mis={}_GT=test.pdf'.format(rdc_threshold,min_instances_slice), reg_line=False, dataset_name=dataset_name)

        results = {
            ind: get_error_totals(df, min_sup_eval, error_names) for ind, df in
            {'train_vs_test': one_v_two, 'spn_vs_train': three_v_one, 'spn_vs_test': three_v_two}.items()
        }
        for ind, errors in results.items():
            d = dict(zip(error_names, errors))
            d.update({'compare': ind,  'min_sup': min_sup_eval,})
            rows.append(d)

    evals = pd.DataFrame(data = rows, ).set_index(['min_sup', 'compare'])
    return evals


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
        elif 'MRE' == e:
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
    print('min_sup: {} \t {}_vs_{}'.format(min_sup, train_use, test_use))
    if train_use == 'SPN':
        PRED = spn_apriori(train, min_support=min_sup, spn=spn, value_dict=value_dict, use_colnames=True, )
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
    itemsets['difference'] = itemsets.support - itemsets.support_pred
    itemsets['difference_percent'] = itemsets.difference / itemsets.support.replace({0.0: np.NaN})
    itemsets['length'] = itemsets['itemsets'].apply(lambda x: len(x))
    # itemsets = itemsets[itemsets['length'] >= 2]
    itemsets = itemsets.sort_values('support_mean', ascending=False)

    # print(itemsets.to_string())

    print('Num. of Sets:\t{}'.format(len(itemsets)))
    return itemsets

