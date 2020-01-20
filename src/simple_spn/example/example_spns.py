'''
Created on 11.07.2019

@author: Moritz
'''


def get_gender_spn():
    from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

    spn1 = Categorical(p=[0.0, 1.0], scope=[0]) * Categorical(p=[0.2, 0.8], scope=[1]) 
    spn2 = Categorical(p=[1.0, 0.0], scope=[0]) * Categorical(p=[0.7, 0.3], scope=[1]) 
    spn3 = 0.4 * spn1 + 0.6 * spn2
    spn = spn3 * Gaussian(mean=20, stdev=3, scope=[2])
    
    spn.scope = sorted(spn.scope)
    return spn


def get_credit_spn():
    from spn.structure.Base import Product
    from spn.structure.leaves.parametric.Parametric import Categorical 

    spn1 = Categorical(p=[0.0, 1.0], scope=[2]) * Categorical(p=[0.5, 0.5], scope=[3]) 
    spn2 = Categorical(p=[1.0, 0.0], scope=[2]) * Categorical(p=[0.1, 0.9], scope=[3]) 
    spn3 = 0.3 * spn1 + 0.7 * spn2
    spn4 = Categorical(p=[0.0, 1.0], scope=[1]) * spn3
    
    spn6 = Product([Categorical(p=[1.0, 0.0], scope=[1]), Categorical(p=[0.0, 1.0], scope=[2]), Categorical(p=[1.0, 0.0], scope=[3])])
    spn6.scope = [1,2,3]
    
    spn7 = 0.8 * spn4 + 0.2 * spn6
    spn = spn7 * Categorical(p=[0.2, 0.8], scope=[0])
    
    spn.scope = sorted(spn.scope)
    return spn
