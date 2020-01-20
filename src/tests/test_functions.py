'''
Created on 10.10.2019

@author: Moritz
'''

import numpy as np

from simple_spn.example import example_spns
from simple_spn import functions as fn

from spn.experiments.AQP.Ranges import NominalRange, NumericRange



def test_sampling():
    spn = example_spns.get_gender_spn()
    
    '''
    Always same random number generator
    '''
    
    samples = fn.sampling(spn, n_samples=10, random_seed=1)
    print(samples)
    
    samples = fn.sampling_rang(spn, rang=[None, None, None, None], n_samples=10, random_seed=1)
    print(samples)
    
    samples = fn.sampling_rang(spn, rang=[None, None, NumericRange([[10,11], [29,30]])], n_samples=10, random_seed=1)
    print(samples)
    
    samples = fn.sampling_rang(spn, rang=[NominalRange([0]), None, NumericRange([[14,15], [29,30]])], n_samples=10, random_seed=1)
    print(samples)
    
    

def test_plot():
    spn = example_spns.get_gender_spn()
    fn.plot_spn(spn, "sample_spn.pdf")
    
    spn = example_spns.get_credit_spn()()
    fn.plot_spn(spn, "sample_spn2.pdf")
    
    
    

def test_prob():
    spn = example_spns.get_gender_spn()
    
    rang = [None, None, None]
    prob = fn.prob(spn, rang)
    print(prob)
    
    rang = [NominalRange([0]), NominalRange([1]), NumericRange([[20]])]
    prob = fn.prob(spn, rang)
    print(prob)
    
    ranges = np.array([[None, None, NumericRange([[0,20]])],
                       [NominalRange([0]), None, None],
                       [None, NominalRange([1]), None]])
    probs = fn.probs(spn, ranges)
    print(probs)
    
    inst = [0, np.nan, np.nan]
    prob = fn.prob_spflow(spn, inst)
    print(prob)
    
    
    data = np.array([[0, np.nan, np.nan], [0, 1, np.nan]])
    probs = fn.probs_spflow(spn, data)
    print(probs)
    



def test_marg():
    spn = example_spns.get_gender_spn()

    spn1 = fn.marg(spn, [2])
    fn.plot_spn(spn1, "marg1.pdf")
    
    spn2 = fn.marg(spn, [0])
    fn.plot_spn(spn2, "marg2.pdf")
    
    spn3 = fn.marg(spn, [1])
    fn.plot_spn(spn3, "marg3.pdf")
    
    spn4 = fn.marg(spn, [1,2])
    fn.plot_spn(spn4, "marg4.pdf")
    
    
    
    
    rang = [None, NominalRange([1]), None]
    prob, spn5 = fn.marg_rang(spn, rang)
    fn.plot_spn(spn5, "marg5.pdf")
    
    rang = [None, NominalRange([1]), NumericRange([[10,12]])]
    prob, spn6 = fn.marg_rang(spn, rang)
    fn.plot_spn(spn6, "marg6.pdf")
    
    
    rang = [NominalRange([0]), NominalRange([1]), None]
    prob = fn.prob(spn, rang)
    print(prob)
    prob = fn.prob(spn6, rang)
    print(prob)
    

def test_expect():
    spn = example_spns.get_gender_spn()
    
    rang = [None, None, None]
    expect = fn.expect(spn, feature_id=2, rang=rang)
    print(expect)
    
    rang = [NominalRange([0]), None, None]
    expect = fn.expect(spn, feature_id=2, rang=rang)
    print(expect)
    
    rang = [NominalRange([1]), None, None]
    expect = fn.expect(spn, feature_id=2, rang=rang)
    print(expect)
    
    rang = [None, NominalRange([0]), None]
    expect = fn.expect(spn, feature_id=2, rang=rang)
    print(expect)
    
    feature_scope = {2}
    data = np.array([[np.nan, np.nan, np.nan]])
    expect = fn.expects_spnflow(spn, feature_scope, data)
    print(expect)
    
    feature_scope = {2}
    data = np.array([np.nan, np.nan, np.nan])
    expect = fn.expect_spnflow(spn, feature_scope, data)
    print(expect)
    
    

def test_get_subpopulations():
    spn = example_spns.get_gender_spn()
    #rang = [NominalRange([0]), NominalRange([1]), None]
    sub_pops = fn.get_sub_populations(spn)

    print(sub_pops)
    

def test_generate_value_dict():
    spn = example_spns.get_gender_spn()
    val_dict = fn.generate_adhoc_value_dict(spn)
    
    print(val_dict)


def test_get_overall_population():
    spn = example_spns.get_gender_spn()
    overall_pop = fn.get_overall_population(spn)
    
    print(overall_pop)


def test_classify():
    
    from util import io
    from data import real_data
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.1)
    spn, _ = io.load(ident, "titanic", loc)
    value_dict = real_data.get_titanic_value_dict()
    #spn = fn.marg(spn, keep=[0,1,2,4,5,7])

    ranges = np.array([[None, NominalRange([1]), None, None, None, None, None, None],
                       [None, NominalRange([0]), None, None, None, None, None, None], 
                       [None, NominalRange([0]), None, None, None, None, None, None]])
    res = fn.classifies(spn, target_id=0, ranges=ranges, value_dict=value_dict)
    print(res)
    
    res = fn.classify(spn, target_id=0)
    print(res)
    
    df, _ =  real_data.get_titanic()
    a = { v[1]: v[2] for _, v in value_dict.items() if v[0] == "discrete"}
    df = df.replace(a)
    
    preds = fn.classify_dataset(spn, target_id=0, df=df, transform=True, value_dict=value_dict)
    print(preds)


def test_mpe_old():
    
    from util import io
    from data import real_data
    
    loc = "_spns"
    ident = "rdc=" + str(0.3) + "_mis=" + str(0.1)
    spn, _ = io.load(ident, "titanic", loc)
    value_dict = real_data.get_titanic_value_dict()
    #spn = fn.marg(spn, keep=[0,1,2,4,5,7])
    
    df, _ =  real_data.get_titanic()
    input_data = df.values
    res = fn.mpe_spflow(spn, 0, input_data)
    print(res)
    
    
    df, _ =  real_data.get_titanic()
    a = { v[1]: v[2] for _, v in value_dict.items() if v[0] == "discrete"}
    df = df.replace(a)
    
    preds = fn.classify_dataset(spn, target_id=0, df=df, transform=True, value_dict=value_dict)
    print(preds)
    
    
def test_value_dict():
    import os
    import pandas as pd
    from util import io
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/titanic/train.csv"
    df = pd.read_csv(path)
    df = df[["Survived", "Sex", "Age", "Fare", "Pclass"]]
    df, val_dict, param_types = fn.transform_dataset(df)
    
    io.print_pretty_table(df)
    print(val_dict)
    print(param_types)
    
    

if __name__ == '__main__':
    #test_prob()
    #test_marg()
    #test_sampling()
    #test_expect()
    #test_get_subpopulations()
    #test_generate_value_dict()
    #test_get_overall_population()
    #test_classify()
    #test_mpe_old()
    test_value_dict()







    