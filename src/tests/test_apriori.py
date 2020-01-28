'''
Created on 10.10.2019

@author: Tim
'''

#todo rem unnecessary imports
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
from mlxtend.frequent_patterns import association_rules as mlxtend_association_rules
from mlxtend.preprocessing import TransactionEncoder
transaction_encoder = TransactionEncoder()
from orangecontrib.associate.fpgrowth import frequent_itemsets as orange_fpgrowth
from spn_apriori.spn_based_apriori import spn_apriori

import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.linear_model import Ridge
# from hanging_threads import start_monitoring
import warnings
from time import perf_counter
import pandas as pd
import numpy as np
from itertools import compress
from math import isclose

from data import real_data
from simple_spn import spn_handler
from simple_spn import functions as fn
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.experiments.AQP.Ranges import NominalRange, NumericRange
from spn_apriori.itemsets_utils import perf_comparison, fpgrowth_wrapper


def test_fpgrowth_equality():
    spn = real_data.get_adult_transactional()
    
    '''
    
    '''
    dataset_name = 'UCI'
    df, value_dict, parametric_types = real_data.get_adult_transactional()
    min_sup = 0.4

    fpgrowth_df = fpgrowth_wrapper(df, min_sup)
    # fpgrowth_df = mlxtend_fpgrowth(df, min_sup, use_colnames=True)
    apriori_df = mlxtend_apriori(df, min_support=min_sup, use_colnames=True, )

    assert len(fpgrowth_df) == len(apriori_df), 'Length differs, fpgrowth: {} apriori: {}'.format(len(fpgrowth_df), len(apriori_df))
    # assert equality of sets. naive costly iteration.
    diff=[]
    for row_a in apriori_df.iterrows():
        aprioriset, apriori_support = row_a[1]['itemsets'], row_a[1]['support']
        existsinother = False
        for row_b in fpgrowth_df.iterrows():
            fpgrowthset, fpgrowth_support = row_b[1]['itemsets'], row_b[1]['support']
            if len(aprioriset.symmetric_difference(fpgrowthset)) == 0:
                existsinother = True
                assert isclose(apriori_support, fpgrowth_support, rel_tol=0.20), \
                    'apriori set {} and fpgrowth set {} differ by more than 0.5%'.format(aprioriset, fpgrowthset)
                break
        if not existsinother:
            diff.append(('apriori:', list(aprioriset)))

    for row_a in fpgrowth_df.iterrows():
        fpgrowthset, fpgrowth_support = row_a[1]['itemsets'], row_a[1]['support']
        existsinother = False
        for row_b in apriori_df.iterrows():
            aprioriset, apriori_support = row_b[1]['itemsets'], row_b[1]['support']
            if len(aprioriset.symmetric_difference(fpgrowthset)) == 0:
                existsinother = True
                assert isclose(apriori_support, fpgrowth_support, rel_tol=0.05), \
                    'apriori set {} and fpgrowth set {} differ by more than 0.5%'.format(aprioriset, fpgrowthset)
                break
        if not existsinother:
            diff.append(('fpgrowth:', list(fpgrowthset)))

    assert len(diff) == 0, 'Differences between apriori and fpgrowth:\n{}'.format(diff)

if __name__ == '__main__':
    test_fpgrowth_equality()





    