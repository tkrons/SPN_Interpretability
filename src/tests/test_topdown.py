'''
Created on 5.3.2020

@author: tkrons
'''
import numpy as np
from simple_spn import spn_handler
from data import real_data, synthetic_data
from simple_spn import functions as fn

def test_rule_clustering(): #todo spflow automatically summarizes chains of sums
    dataset_name = 'gender'
    recalc_SPN = True
    rdc_threshold, min_instances_slice = 0.2, 0.1

    if not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice) or recalc_SPN:
        print("Creating SPN ...")

        # get data
        df, value_dict, parametric_types = synthetic_data.get_synthetic_data(dataset_name)

        # Creates the SPN and saves to a file
        spn_handler.create_parametric_spns(df.values, parametric_types, dataset_name, [rdc_threshold], [min_instances_slice],
                                           clustering = 'rule_clustering')

    # Load SPN
    spn, value_dict, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
    fn.print_statistics(spn)

    pass



    # def learn_CLTSPN():
    #     import numpy as np
    #
    #     np.random.seed(123)
    #
    #     train_data = np.random.binomial(1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1], size=(100, 10))
    #     print(np.mean(train_data, axis=0))
    #
    #     from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
    #     from spn.structure.Base import Context
    #     from spn.structure.leaves.parametric.Parametric import Bernoulli
    #
    #     ds_context = Context(
    #         parametric_types=[
    #             Bernoulli,
    #             Bernoulli,
    #             Bernoulli,
    #             Bernoulli,
    #             Bernoulli,
    #             Bernoulli,
    #             Bernoulli,
    #             Bernoulli,
    #             Bernoulli,
    #             Bernoulli,
    #         ]
    #     ).add_domains(train_data)
    #
    #     from spn.algorithms.LearningWrappers import learn_parametric
    #
    #     spn = learn_parametric(
    #         train_data,
    #         ds_context,
    #         min_instances_slice=20,
    #         min_features_slice=1,
    #         multivariate_leaf=True,
    #         leaves=create_cltree_leaf,
    #     )
    #
    #     from spn.algorithms.Statistics import get_structure_stats
    #
    #     print(get_structure_stats(spn))
    #
    #     from spn.io.Text import spn_to_str_equation
    #
    #     print(spn_to_str_equation(spn))
    #
    #     from spn.algorithms.Inference import log_likelihood
    #
    #     ll = log_likelihood(spn, train_data)
    #     print(np.mean(ll))