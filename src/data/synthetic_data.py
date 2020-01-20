'''
Created on 08.07.2019

@author: Moritz
'''

import numpy as np
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from sklearn.datasets import make_blobs


def generate_dummy_dataset():
    a = np.r_[np.random.normal(10, 5, (300, 1)), np.random.normal(20, 10, (700, 1))]
    b = np.r_[np.random.normal(3, 2, (300, 1)), np.random.normal(50, 10, (700, 1))]
    c = np.r_[np.random.normal(20, 3, (1000, 1))]
    train_data = np.c_[a, b, c]
    return train_data

def generate_icecream_data(n = 40, features=2):
    data, target = make_blobs(n_samples=n, n_features=features, random_state=6, cluster_std=3)
    # age = x, icecream_eaten/year = y 18 muss x~=0 entsprechen
    x, y = data[:, 0], data[:, 1]
    y = np.round(y - np.min(y))
    x = np.round((x - np.min(x) + 1) * 1.3)
    result = [x, y]
    value_dict = {}
    types = [Gaussian, Gaussian, Categorical]
    if features > 2: # add some other features
        for i in range(features)[2:]:
            gi = data[:, i]
            y = np.c_[y, gi]
            value_dict[i] = ['numeric', 'gaussian{}'.format(i), [np.min(gi), np.max(gi)]]
            types.insert(i, Gaussian)
    value_dict.update({0: ['numeric', 'Age', [np.min(x), np.max(x)]],
                  1: ['numeric', 'Icecream/Year', [np.min(y), np.max(y)]],
                  features: ['discrete', 'Man/Woman/Child', {0: 'Man', 1: 'Woman', 2: 'Child'}]})
    return np.column_stack((x, y, target)), value_dict, types

def generate_gender_age_data(num_instances, rand_seed):    
    '''
    Correlations:
    P(gender=male) = 50%
    P(gender=male) = 50%
    P(student=yes|gender=m) = 30%
    P(student=yes|gender=f) = 80%
    P(age) = N(mu=20, sigma=3) ...
    '''
    
    np.random.seed(rand_seed)

    data = [] 
    for _ in range(num_instances):
        inst = []
        if np.random.random() < 0.5:
            inst.append(0)
            if np.random.random() < 0.3:
                inst.append(1)
            else:
                inst.append(0)
            inst.append(int(np.random.normal(25, 1)))
            inst.append(int(np.random.normal(25, 4)))
            
        else:
            inst.append(1)
            if np.random.random() < 0.8:
                inst.append(1)
            else:
                inst.append(0)     
            inst.append(int(np.random.normal(20, 1)))
            inst.append(int(np.random.normal(20, 3)))
            
        #inst.append(int(np.random.normal(20, 3)))
        data.append(inst)
    
    return np.array(data), [Categorical, Categorical, Gaussian, Gaussian]




