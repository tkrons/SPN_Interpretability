'''
Created on 08.07.2019

@author: Moritz, Tim
'''


import os
import numpy as np
import pandas as pd
from simple_spn import functions as fn
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian, Bernoulli, Poisson
from simple_spn.functions import get_feature_types_from_dataset
from mlxtend.preprocessing import TransactionEncoder

def get_real_data(name, **kwargs):
    '''wrapper to get data by name
    @:param name: titanic, T10I4D, play_store, adult
    '''
    case = {'titanic': get_titanic,
            'T10I4D': get_T10I4D,
            'play_store': get_play_store,
            'adult41': get_adult_41_items}
    return case[name](**kwargs)

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

def get_adult_one_hot(): #todo do own data cleaning (reproducavbility)
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

def get_adult_41_items(convert_tabular = False):
    '''
    UCI adult dataset. cleaned and in transactional form
    The age was discretized. numeric columns (except age) were removed.
    The purpose of this test is to assure that the algorithm can deal with a
    small 2.2 MB (30k rows) data set reasonably efficiently.
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

def get_play_store(one_hot=True):
    #todo https://www.kaggle.com/lava18/google-play-store-apps
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/play_store/googleplaystore.csv"
    df = pd.read_table(path, sep=',')
    df.columns
    #filter weird stuff
    df = df[~(df.Category == "1.9")]

    #feature engineering
    # df.loc[df.Price == 0, 'Price'] = 0.
    # df.loc[0 < df.Price <= 1, 'Price'] = 1.
    # df.loc[1 < df.Price <= 3, 'Price'] = 3.
    #parse $ strings round price to ceil
    # df.Price = df.Price.apply(lambda x: np.ceil(float(x.split('$')[0])))
    def _parse(x):
        try:
            f = float(x.split('$')[-1])
        except ValueError:
            f = np.NaN
        # return int(np.ceil(f))
        return  f
    df.Price = df.Price.apply(_parse)
    df.Price = pd.cut(df.Price, [-np.inf, 0, 1, 2, 3, 5, 10, 20, 50, np.inf],
                      labels= ['0', '0-1', '1-2', '2-3', '3-5', '5-10', '10-20', '20-50', '50+']).astype(str)
    df = df[df.Genres.isin(df.Genres.value_counts()[:25].index)]
    # df['Reviews'] =
    df.Reviews = pd.cut(df['Reviews'].astype(float),
                        [-np.inf, 0, 10, 100, 1000, 1e4, 1e5, 1e6],
                        labels = ['0', '<= 10', '<= 100', '<= 1000', '<= 10,000', '<= 100,000', '<= 1,000,000'],
                        retbins=False, include_lowest=False).astype(str)
    df.Rating = pd.cut(df['Rating'].astype(float),
                        [1, 2, 3, 4, 5],
                        labels = ['1-2', '2-3', '3-4', '4-5'],
                        retbins=False, include_lowest=False).astype(str)
    cols = ['Category', 'Price', 'Rating', 'Genres', 'Content Rating', 'Reviews', 'Installs']
    # original data value_dict (tabular data!)
    value_dict = {i: ['discrete', dict(enumerate(df[c].value_counts().index))] for i, c in enumerate(cols)}
    parametric_types = [Categorical, Categorical, Categorical, Categorical, Categorical, Categorical, Categorical, Categorical]
    df = df[cols].dropna(axis=1, )

    #one hot data
    onehot = pd.get_dummies(df[cols], prefix=cols, prefix_sep='=')
    # value_dict_onehot = {i:}
    parametric_types_onehot = get_feature_types_from_dataset(onehot)
    _, value_dict_onehot, _ = fn.transform_dataset(onehot, ['discrete'] * len(onehot.columns))
    return onehot, value_dict_onehot, parametric_types_onehot

def get_movies_one_hot():
    #todo https://www.kaggle.com/tmdb/tmdb-movie-metadata
    # which actors make movies successful?
    # problem: needs lot of transformations
    pass
