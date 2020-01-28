'''
Created on 08.07.2019

@author: Moritz, Tim
'''


import os
import numpy as np
import pandas as pd
from simple_spn import functions as fn
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from mlxtend.preprocessing import TransactionEncoder


def get_T10I4D(max_num=100, max_insts=10000):
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/itemset/T10I4D100K.dat"
    file = open(path, "r")

    nums = set()
    all_vals = []
    for _ in range(max_insts):
        line = file.readline()
        
        vals = set()
        for str_num in line.split(" ")[:-1]:
            num = int(str_num)
            if num < max_num:
                nums.add(num)
                vals.add(num)
        
        if len(vals) > 1:
            all_vals.append(vals)
            
    max_val = max(nums) + 1
    insts = []
    for vals in all_vals:
        inst = np.zeros(max_val)
        inst[list(vals)] = 1
        insts.append(inst)
    
    return np.array(insts), [Categorical]*(max_val)



def get_titanic(col_names=None):
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/titanic/train.csv"

    df = pd.read_csv(path)
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    
    if col_names is not None:
        df = df[col_names]
    
    #Fill missing values    
    df["Age"].fillna(int(df["Age"].mean()), inplace=True)
    df["Embarked"].fillna("S", inplace=True)

    return fn.transform_dataset(df)

def get_adult_full():
    '''
    UCI adult dataset full
    :return:
    '''
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/adult/adult.dat"
    df = pd.read_csv(path, columns=columns)
    df.isnull().sum()
    df.dropna(inplace=True)
    return df

def get_adult_transactional(convert_tabular = False):
    '''
    UCI adult dataset. cleaned and in transactional form
    https://raw.githubusercontent.com/tommyod/Efficient-Apriori/master/efficient_apriori/tests/adult_data_cleaned.txt
    :return:
    '''
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/adult/adult_data_transactions.data"

    if not convert_tabular:
        transaction_encoder = TransactionEncoder()
        # data = []
        # with open(path) as f:
        #     for line in f.readlines():
        #         data.append(set(line.strip().replace(' ', '').split(',')))
        # fit = transaction_encoder.fit(data)
        # one_hot_df = pd.DataFrame(fit.transform(data), columns=fit.columns_)
        columns = ['education', 'marital-status', 'relationship', 'race', 'sex', 'income', 'age']
        tabular = pd.read_table(path, sep=',', names=columns, skipinitialspace=True)
        one_hot_df = pd.get_dummies(tabular, prefix='', prefix_sep='', dtype=np.bool)
        return fn.transform_dataset(one_hot_df)
    else:
        columns = ['education', 'marital-status', 'relationship', 'race', 'sex', 'income', 'age']
        tabular = pd.read_table(path, sep=',', names = columns, skipinitialspace=True)
        return fn.transform_dataset(tabular)
