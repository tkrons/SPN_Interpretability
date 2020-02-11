'''
Created on 10.10.2019

@author: Tim
'''

from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
from mlxtend.frequent_patterns import association_rules as mlxtend_association_rules
from mlxtend.preprocessing import TransactionEncoder
transaction_encoder = TransactionEncoder()

import pandas as pd
import numpy as np

from data import real_data, synthetic_data

import spn_apriori.apriori_evaluation as evaluation

def test_spn_apriori_synthetic():
    df,_,_ = synthetic_data.generate_simple_transactions(seed = 123)

def test_spn_apriori_real():
    df, value_dict, parametric_types = real_data.get_adult_41_items()
    eval = evaluation.cross_eval(df, 'UCI', [0.01, 0.4], value_dict, recalc_spn=False)
    expect = np.array([ # AE    MAE     MRE     MISS        EXCESS
         [2.94790000e+00, 1.38986327e-03, 4.62524330e-02, 3.50000000e+01, 4.00000000e+01, 2.12100000e+03],
         [4.53490000e+00, 2.08981567e-03, 5.15308517e-02, 9.40000000e+01, 1.73000000e+02, 2.17000000e+03],
         [4.62710000e+00, 2.13230415e-03, 5.60047765e-02, 1.03000000e+02, 1.68000000e+02, 2.17000000e+03],
         [6.45000000e-02, 4.03125000e-03, 8.31297871e-03, 0.00000000e+00, 0.00000000e+00, 1.60000000e+01],
         [6.41000000e-02, 4.00625000e-03, 8.83491304e-03, 0.00000000e+00, 0.00000000e+00, 1.60000000e+01],
         [3.95000000e-02, 2.46875000e-03, 4.94322745e-03, 0.00000000e+00, 0.00000000e+00, 1.60000000e+01],
    ])
    # print(repr(eval.values))
    print(eval.values)
    assert np.allclose(eval.values, expect, rtol=1e-6), 'False:\n' + str(eval.values)


def test_fpgrowth_apriori_equality():
    df, value_dict, parametric_types = real_data.get_adult_41_items()
    min_sup = 0.01
    rel_tol = 0., # works for now! take care with floating point inaccuracy

    # fpgrowth_df = fpgrowth_wrapper_orange(df, min_sup)
    fpgrowth_df = mlxtend_fpgrowth(df, min_sup, use_colnames=True)
    apriori_df = mlxtend_apriori(df, min_support=min_sup, use_colnames=True, )
    #test the sets along min_sup:
    merged = pd.merge(fpgrowth_df,
             apriori_df,
             how='outer', on='itemsets', suffixes=('_FPGrowth', '_Apriori'))
    print(len(merged))
    # assert missing sets are abs_tol close to min_sup
    # assert merged.loc[
    #            (merged.support_FPGrowth >= min_sup) or (merged.support_Apriori >= min_sup),
    #            ['support_FPGrowth', 'support_Apriori']
    #        ].max().max() < (min_sup + abs_tol)
    # assert no missing no excess sets
    set_diff = merged.loc[merged.isnull().any(axis=1)]
    assert len(set_diff) == 0, 'Sets differ by: ' + str(set_diff)

    # simple assert equal support
    assert np.allclose(merged.support_FPGrowth, merged.support_Apriori, rtol = rel_tol), 'arrays differ'

    # fpgrowth_df.dropna(inplace=True)
    # apriori_df.dropna(inplace=True)
    # assert len(fpgrowth_df) == len(apriori_df), 'Length differs, fpgrowth: {} apriori: {}'.format(len(fpgrowth_df), len(apriori_df))
    # assert equality of sets. naive costly iteration.
    # diff=[]
    # for row_a in apriori_df.iterrows():
    #     aprioriset, apriori_support = row_a[1]['itemsets'], row_a[1]['support']
    #     existsinother = False
    #     for row_b in fpgrowth_df.iterrows():
    #         fpgrowthset, fpgrowth_support = row_b[1]['itemsets'], row_b[1]['support']
    #         if len(aprioriset.symmetric_difference(fpgrowthset)) == 0:
    #             existsinother = True
    #             assert isclose(apriori_support, fpgrowth_support, rel_tol=rel_tol), \
    #                 'apriori set {} and fpgrowth set {} differ by more than 0.5%'.format(aprioriset, fpgrowthset)
    #             break
    #     if not existsinother:
    #         diff.append(('apriori:', list(aprioriset)))
    #
    # for row_a in fpgrowth_df.iterrows():
    #     fpgrowthset, fpgrowth_support = row_a[1]['itemsets'], row_a[1]['support']
    #     existsinother = False
    #     for row_b in apriori_df.iterrows():
    #         aprioriset, apriori_support = row_b[1]['itemsets'], row_b[1]['support']
    #         if len(aprioriset.symmetric_difference(fpgrowthset)) == 0:
    #             existsinother = True
    #             assert isclose(apriori_support, fpgrowth_support, rel_tol=rel_tol), \
    #                 'apriori set {} and fpgrowth set {} differ by more than 0.5%'.format(aprioriset, fpgrowthset)
    #             break
    #     if not existsinother:
    #         diff.append(('fpgrowth:', list(fpgrowthset)))
    #
    # assert len(diff) == 0, 'Differences between apriori and fpgrowth:\n{}'.format(diff)

# if __name__ == '__main__':
#     test_spn_apriori_real()

    # does not work, somewhat high differences todo try other lib
    # test_fpgrowth_equality()





    