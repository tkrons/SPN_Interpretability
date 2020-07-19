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
from mlxtend.preprocessing import TransactionEncoder, OnehotTransactions
from sklearn.preprocessing import KBinsDiscretizer

def get_real_data(name, **kwargs):
    '''wrapper to get data by name
    @:param name: titanic, T10I4D, play_store, adult

    value dict: [type, name {0: value0, 1: value1...}
    '''
    case = {'titanic': get_titanic_bins,
            'T10I4D': get_T10I4D,
            'play_store': get_play_store,
            'adult41': get_adult_41_items,
            'OnlineRetail': get_OnlineRetail,
            'Ecommerce': get_Ecommerce,
            'RecordLink': get_RecordLink,
            'adult_one_hot': get_adult_one_hot,
            'lending': get_lending,
            }
    return case[name](**kwargs)

def get_lending(only_n_rows = None, seed = None, onehot = True, original=False,):
    '''
    https://www.kaggle.com/wendykan/lending-club-loan-data
    about 2.200.000 rows,
    '''
    # discretizer = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='quantile')

    with open('../../_data/lending/loan.csv', 'r', encoding='latin-1') as f:
        used_cols = ['loan_amnt', 'loan_status', 'term', 'purpose', 'int_rate', 'grade', 'emp_length',
                    'home_ownership', 'annual_inc']
        df = pd.read_csv(f, usecols=used_cols)
        df = _shorten_df(df, only_n_rows, seed=seed)
    if original:
        return df
    df = df[~df.loan_status.isin(['Does not meet the credit policy. Status:Charged Off',
                                 'Does not meet the credit policy. Status:Fully Paid'])]
    df.loan_status.replace(['Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period'], 'Late', inplace=True)
    df.loan_status.replace('Default', 'Charged Off', inplace=True)
    df.emp_length.replace([str(i) + ' years' for i in range(2, 10)], '1-10 years', inplace=True)
    df.emp_length.replace(['< 1 year', '1 year'], '<= 1 year', inplace=True)
    df.grade.replace({'A': 'good', 'B': 'good', 'C': 'medium', 'D': 'medium','E': 'bad', 'F': 'bad', 'G': 'bad'}, inplace=True)

    keep = (df.purpose.value_counts()[df.purpose.value_counts().head(10).index]).index.to_list()
    df.purpose = df.purpose[df.purpose.isin(keep)]

    df.dropna(inplace=True)
    numeric_cols = df.columns[~(df.dtypes == np.object)]
    # df[numeric_cols] = discretizer.fit_transform(df[numeric_cols])

    for c in numeric_cols:
        # quantiles = np.round(df[c].quantile([0.25, 0.5, 0.75])).astype(int).tolist()
        # q_labels = [x.format(low=quantiles[0], mid=quantiles[1], high=quantiles[2]) for x in ['0 - {low}', '{low} - {mid}', '{mid} - {high}', '{high} - inf']]
        quantiles = np.round(df[c].quantile([0.25, 0.5, 0.75])).astype(int).tolist()
        q_labels = [x.format(low=quantiles[0], mid=quantiles[1], high=quantiles[2], max=int(df[c].max())) for x in
                    ['0 - {low}', '{low} - {mid}', '{mid} - {high}', '{high} - {max}']]
        df[c] = pd.cut(df[c],
                       bins = [-np.inf] + quantiles + [np.inf],
                       labels = q_labels,
                       ).astype(str)
    df.emp_length.replace([''])
    #remove rare items
    df = df[~df.home_ownership.isin(['OTHER', 'ANY'])]

    if onehot:
        df = pd.get_dummies(df, )

    return fn.transform_dataset(df)

def get_RecordLink(only_n_rows = None, seed =None):
    ''' https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php
    '''
    colnames = []
    for orig_col  in ['cmp_fname_c1','cmp_fname_c2', 'cmp_lname_c1', 'cmp_lname_c2', 'cmp_sex', 'cmp_bd', 'cmp_bm', 'cmp_by', 'cmp_plz']:
        colnames = colnames + [orig_col + val for val in [' = no', ' = yes', ' = uncertain']]
    colnames = colnames + ['is_match = no', 'is_match = yes']
    with open('../../_data/RecordLink/RecordLink.txt', 'r+',) as f:
        transactions = f.read().splitlines()
        one_hot = np.zeros([len(transactions), len(colnames)], dtype=bool)
        for ind, t in enumerate(transactions):
            if t != '':
                items = t.strip().split(' ')
                for item in items:
                    one_hot[ind, int(item)] = True

    df = pd.DataFrame(one_hot, columns=colnames, dtype=int)
    df = _shorten_df(df, only_n_rows, seed)
    value_dict = {i: ['discrete', c, {0: 0, 1: 1}] for i, c in enumerate(df.columns)}
    parametric_types = [Categorical] * len(df.columns)
    return df, value_dict, parametric_types



def get_Ecommerce(only_n_rows = None, seed =None):
    '''Data Set Information:
    This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
    InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
    StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
    Description: Product (item) name. Nominal.
    Quantity: The quantities of each product (item) per transaction. Numeric.
    InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
    UnitPrice: Unit price. Numeric, Product price per unit in sterling.
    CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
    Country: Country name. Nominal, the name of the country where each customer resides.'''
    with open('../../_data/Ecommerce/Ecommerce.xlsx', 'rb',) as f:
        source_df = pd.read_excel(f, usecols = [0,1,2,5,6,7],)
        # transform transactional (groupby customers)
    dummies = pd.get_dummies(source_df.set_index('CustomerID')['Description'], )
    transactions = dummies.reset_index().groupby('CustomerID').max().reset_index().drop(columns=['CustomerID'])
    colnames = source_df['Description'].unique().tolist()

    transactions = _shorten_df(transactions, only_n_rows, seed)

    value_dict = {i: ['discrete', c, {0: 0, 1: 1}] for i, c in enumerate(transactions.columns)}
    parametric_types = [Categorical] * len(transactions.columns)
    return transactions, value_dict, parametric_types

def get_OnlineRetail(only_n_rows = None, seed = None):
    '''This dataset is transformed from the Online Retail dataset, found at https://archive.ics.uci.edu/ml/datasets/ Online+Retail.
    https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php

    Insgesamt eher uninteressant, '&' ist das häufigste 'item' ...
    '''
    def __left_right_strip_once(x):
        r = str(x).replace('\'', '', 1)
        return r[::-1].replace('\'', '', 1)[::-1]
    colnames = pd.read_excel('../../_data/OnlineRetail/OnlineRetailZZAtrributes.xlsx',
                             converters={0: __left_right_strip_once}, header=None, index_col=1).values.reshape(-1)
    with open('../../_data/OnlineRetail/OnlineRetailZZ.txt', 'r+') as f:
        # onehot = OnehotTransactions().fit_transform(f.read().splitlines())
        transactions = f.read().splitlines()
        one_hot = np.zeros([len(transactions), len(colnames)], dtype=bool)
        for ind, t in enumerate(transactions):
            if t != '':
                items = t.strip().split(' ')
                for item in items:
                    one_hot[ind, int(item) - 1] = True
    df = pd.DataFrame(one_hot, columns=colnames, dtype=int)
    value_dict = {i: ['discrete', c, {0: 0, 1: 1}] for i,c in enumerate(df.columns)}
    parametric_types = [Categorical] * len(df.columns)
    df = _shorten_df(df, only_n_rows, seed)
    return df, value_dict, parametric_types

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

def _shorten_df(data, only_n_rows, seed=None):
    if only_n_rows and only_n_rows < len(data):
        np.random.seed(seed)
        sample_indices = np.random.choice(range(len(data)), only_n_rows, replace=False)
        if isinstance(data, pd.DataFrame):
            data = data.iloc[sample_indices]
        else:
            raise ValueError()
        # elif isinstance(data, np.ndarray):
        #     data = data[sample_indices]
        # elif isinstance(data, list):
        #     data = [data[i] for i in sample_indices]
        # else:
        #     raise ValueError(type(data))
    # remove non existing items (otherwise breaks SPN)
    for col in data.columns:
        if len(set(data.loc[data[col].notna(), col].unique())) <= 1:
            data.drop(col, inplace=True, axis=1)
    return data

def _one_hot_value_dict(df,):
    vd = {}
    for i, c in enumerate(df.columns):
        vd[i] = ['discrete', c, {0: 0, 1: 1}]
    return vd

def get_titanic(col_names=None, onehot=False, only_n_rows=None, seed=None, original=False):
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/titanic/train.csv"

    df = pd.read_csv(path)
    if original:
        return df
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    
    if col_names is not None:
        df = df[col_names]
    
    #Fill missing values    
    df["Age"].fillna(int(df["Age"].mean()), inplace=True)
    df["Embarked"].fillna("S", inplace=True)

    if only_n_rows and only_n_rows < len(df):
        df = df.sample(only_n_rows, random_state=seed)
    if onehot:
        df['Survived'] = df['Survived'].astype(bool).astype(str)
        df = pd.get_dummies(df)

    return fn.transform_dataset(df)

def mini_titanic():
    data={'Survived': {356: True,
  255: True,
  380: True,
  859: False,
  886: False,
  248: True,
  598: False,
  372: False,
  574: False,
  820: True},
 'Embarked': {356: 'Southampton',
  255: 'Cherbourg',
  380: 'Cherbourg',
  859: 'Cherbourg',
  886: 'Southampton',
  248: 'Southampton',
  598: 'Cherbourg',
  372: 'Southampton',
  574: 'Southampton',
  820: 'Southampton'},
 'Sex': {356: 'female',
  255: 'female',
  380: 'female',
  859: 'male',
  886: 'male',
  248: 'male',
  598: 'male',
  372: 'male',
  574: 'male',
  820: 'female'}}
    columns=['Survived', 'Embarked', 'Sex']
    small = pd.DataFrame(data, columns=columns)
    small = small.sort_values(['Survived', 'Sex', 'Embarked'])
    print(small.to_latex(index=False))
    return fn.transform_dataset(small)

def mini_leding(seed=1):
    with open('../../_data/lending/loan.csv', 'r', encoding='latin-1') as f:
        used_cols = ['loan_amnt', 'loan_status', 'term', 'purpose', 'int_rate', 'grade', 'emp_length',
                     'home_ownership', 'annual_inc']
        df = pd.read_csv(f, usecols=used_cols)
        df = _shorten_df(df, 100, seed=seed)
    df = df[~df.loan_status.isin(['Does not meet the credit policy. Status:Charged Off',
                                  'Does not meet the credit policy. Status:Fully Paid'])]
    df.loan_status.replace(['Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period'], 'Late', inplace=True)
    df.emp_length.replace([str(i) + ' years' for i in range(2, 10)], '1-10 years', inplace=True)
    df.dropna(inplace=True)
    df = df[~df.home_ownership.isin(['OTHER', 'ANY'])]
    return df

def get_titanic_bins(col_names=None, onehot=False, only_n_rows=None, seed=None):
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/titanic/train.csv"
    df = pd.read_csv(path)
    df['NumFamily'] = (df.SibSp + df.Parch).astype(int)
    df.loc[df.NumFamily >= 3, 'NumFamily'] = '3+'
    df.NumFamily = df.NumFamily.astype(str)
    df.Embarked.replace({'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}, inplace=True)
    df["Embarked"].fillna("Unknown", inplace=True)
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", 'SibSp', 'Parch', 'Fare'], inplace=True)
    if col_names is not None:
        df = df[col_names]
    df.Pclass = df.Pclass.astype(str)
    # Fill missing values
    df['Age_'] = np.NaN
    df.loc[df['Age'] < 16, 'Age_'] = 'child'
    df.loc[df['Age'].between(16, 30), 'Age_'] = 'young-adult'
    df.loc[df['Age'].between(31, 50), 'Age_'] = 'middle-aged'
    df.loc[df['Age'].between(50, df.Age.max()), 'Age_'] = 'old'
    df['Age_'].fillna('Unknown', inplace=True)
    df = df.drop(columns=['Age']).rename(columns={'Age_': 'Age'})

    if only_n_rows and only_n_rows < len(df):
        df = df.sample(only_n_rows, random_state=seed)
    if onehot:
        df['Survived'] = df['Survived'].astype(bool).astype(str)
        df = pd.get_dummies(df)

    return fn.transform_dataset(df)


def get_adult_one_hot(only_n_rows=None, seed=None):
    '''
    UCI adult dataset
    :return:
    '''
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/adult/adult.data"
    df = pd.read_csv(path, names=columns, na_values='?', index_col=None, skipinitialspace=True)

    df['marital-status'].replace(['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent'], 'Married',
                                 inplace=True)
    df = df.drop(columns=['fnlwgt', 'capital-gain', 'capital-loss', 'education-num'])
    df = df.dropna() # drop NaNs in workclass, occupation and native country
    df.age = pd.cut(df['age'], [0, 35, 60, np.inf],
                       labels=['young', 'middle-aged', 'old'],
                       retbins=False, include_lowest=True).astype(str)

    df['hours-per-week'] = pd.cut(df['hours-per-week'], [0, 35, 45, np.inf],
                                  labels=['work hours < 35', 'work hours 35-45', 'work hours > 45'])
    # filter rare countries
    # counts = df['native-country'].value_counts()
    # df =df[df['native-country'].isin(counts.index[counts > 20])]
    # # filter rare occupations
    # df = df[df.occupation.isin(counts.index[counts > 20])]

    # only use 51? items
    df.drop(columns=['hours-per-week', 'native-country', 'occupation'], inplace=True)
    one_hot = pd.get_dummies(df, prefix='', prefix_sep='')
    if only_n_rows:
        one_hot = one_hot.sample(only_n_rows)
    print('adult_one_hot: {} Rows, {} Items'.format(len(one_hot), len(one_hot.columns)))
    val_dict = _one_hot_value_dict(one_hot)
    parametric_types = [Categorical] * len(val_dict)
    return one_hot, val_dict, parametric_types
    # return fn.transform_dataset(one_hot, )

def get_adult_41_items(onehot = False, only_n_rows=None, seed = None):
    '''
    UCI adult dataset. cleaned and in transactional form
    The age was discretized. numeric columns (except age) were removed.
    The purpose of this test is to assure that the algorithm can deal with a
    small 2.2 MB (30k rows) data set reasonably efficiently.
    https://raw.githubusercontent.com/tommyod/Efficient-Apriori/master/efficient_apriori/tests/adult_data_cleaned.txt
    :return:
    '''
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/adult/adult_data_transactions.data"

    if onehot:
        transaction_encoder = TransactionEncoder()
        # data = []
        # with open(path) as f:
        #     for line in f.readlines():
        #         data.append(set(line.strip().replace(' ', '').split(',')))
        # fit = transaction_encoder.fit(data)
        # one_hot_df = pd.DataFrame(fit.transform(data), columns=fit.columns_)
        columns = ['education', 'marital-status', 'relationship', 'race', 'sex', 'income', 'age']
        tabular = pd.read_table(path, sep=',', names=columns, skipinitialspace=True)
        df = pd.get_dummies(tabular.astype(str), prefix=None, prefix_sep='_', dtype=np.bool)
    else:
        columns = ['education', 'marital-status', 'relationship', 'race', 'sex', 'income', 'age']
        df = pd.read_table(path, sep=',', names = columns, skipinitialspace=True)
    if only_n_rows and only_n_rows < len(df):
        df = df.sample(only_n_rows, random_state=seed)
    return fn.transform_dataset(df)

def get_play_store(one_hot=True):
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/play_store/googleplaystore.csv"
    df = pd.read_table(path, sep=',')
    df.columns
    #filter weird stuff
    df = df[~(df.Category == "1.9")]

    #feature engineering
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
                        [-np.inf, 0, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8],
                        labels = ['0', '1+', '10+', '100+', '1,000+', '10,000+', '100,000+', '1,000,000+', '10,000,000+'],
                        retbins=False, include_lowest=False).astype(str)
    df.Rating = pd.cut(df['Rating'].astype(float),
                        [1, 2, 3, 4, 5],
                        labels = ['1-2', '2-3', '3-4', '4-5'],
                        retbins=False, include_lowest=False).astype(str)
    cols = ['Category', 'Price', 'Rating', 'Content Rating', 'Reviews', 'Installs'] # 'Genres' viel identisch mit category
    # original data value_dict (tabular data!)
    value_dict = {i: ['discrete', dict(enumerate(df[c].value_counts().index))] for i, c in enumerate(cols)}
    parametric_types = [Categorical, Categorical, Categorical, Categorical, Categorical, Categorical, Categorical, Categorical]
    df = df[cols].dropna(axis=1, )

    #one hot data
    onehot = pd.get_dummies(df[cols], prefix=cols, prefix_sep=': ')
    # value_dict_onehot = {i:}
    parametric_types_onehot = get_feature_types_from_dataset(onehot)
    one_hot, value_dict_onehot, _ = fn.transform_dataset(onehot, ['discrete'] * len(onehot.columns))
    return onehot, value_dict_onehot, parametric_types_onehot

if __name__ == '__main__':
    from simple_spn import spn_handler
    import matplotlib.pyplot as plt
    from spn.structure.leaves.parametric.Parametric import Categorical

    from spn.io.Text import spn_to_str_equation
    #
    # df , vd, pars = mini_titanic()
    # from simple_spn import functions as fn
    # from spn.io.Text import spn_to_str_equation
    # spn = spn_handler.load_or_create_spn(df, vd, pars, 'mini_titanic', 0.1,
    #                                      0.1,
    #                                      nrows=None, seed=1, force_create=True, clustering='km_rule_clustering')
    # str_df = df.round(0)
    # repl_dict = {col: vd[icol][2] for icol, col in enumerate(str_df.columns)}
    # cols = str_df.columns
    # new_df = str_df.copy(deep=True)
    # new_df[cols[0]] = str_df[cols[0]].replace(repl_dict[cols[0]])
    # new_df[cols[1]] = str_df[cols[1]].replace(repl_dict[cols[1]])
    # new_df[cols[2]] = str_df[cols[2]].replace(repl_dict[cols[2]])
    # rang = [np.NaN] * len(spn.scope)
    #

    # mini spn example
    x = np.random.choice([1,2], int(1e4), replace=True, p=[0.3, 0.7])
    p = {1: [0.9, 0.1, 0.], 2: [0., 0.9, 0.1]}
    y=[]
    for v in x:
        y.append(np.random.choice([1, 2, 3], 1, replace=True, p=p[v]))
    y = np.array(y).reshape(-1,)
    z = np.random.choice([1, 2], int(1e4), replace=True, p=[0.4, 0.6])
    df = pd.DataFrame(dict(zip(['X', 'Y', 'Z'], [x,y,z]))).astype(str)
    df, vd, pars = fn.transform_dataset(df)
    spn = spn_handler.load_or_create_spn(df, vd, pars, 'mini_example', 0.4,
                                         0.5,
                                         nrows=None, seed=1, force_create=True, clustering='km_rule_clustering')
    spn = spn.children[1]
    manspn = ( 0.3 * (Categorical(p=[0.9, 0.1], scope=0) * Categorical(p=[0.55, 0.4, 0.05], scope=1))
               + 0.7 * (Categorical(p=[0., 1.], scope=0) * Categorical(p=[0.1, 0.2, 0.7], scope=1)) ) \
            * (Categorical(p=[0.4, 0.6], scope=2))
    # plot leaves from example
    p = [[0.9, 0.1], [0.4,0.55,0.05], [0.,1.], [0.1,0.2,0.7], [0.4,0.6]]
    y=2
    size = (2.88*y, y)
    fig, axes = plt.subplots(1, 4, sharey=True, squeeze=True, figsize=size)
    for i, var in enumerate(['X', 'Y', 'X', 'Y']):
        currp = p[i]
        ax = axes[i]
        # if i in [1,2]:
        #     d = df[var].value_counts(sort=False).divide(len(df))
        # if i in [3,4]:

        ticks=list(range(len(currp)))
        labels = ['{}{}'.format(var, i) for i, _ in enumerate(currp)]
        ax.bar(ticks, currp, )
        ax.set_xticks(ticks, )
        ax.set_xticklabels(labels)
        if i==0:
            ax.set_ylabel('probability')
        # plt.xticklabels
        # ax.set_ylim([0,1])
    plt.tight_layout()
    plt.savefig('../../_figures/rule_extraction/leaves4.png', bbox_inches='tight', dpi=400)
    plt.show()
    from spn.structure.leaves.parametric.Parametric import Poisson
    # Z plot
    fig, ax =plt.subplots(figsize=(size[0]/2.5, size[1]))
    currp = p[4]
    ticks = list(range(len(currp)))
    labels = ['{}{}'.format('Z', i) for i, _ in enumerate(currp)]
    ax.bar(ticks, currp, )
    ax.set_xticks(ticks, )
    ax.set_xticklabels(labels)
    ax.set_ylim([0.,1.05])
    ax.set_ylabel('probability')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.savefig('../../_figures/rule_extraction/leaf5.png', bbox_inches='tight', dpi=400)
    plt.show()

    fn.plot_spn(spn)
    df,_,_ = get_titanic_bins()
    c=df.Pclass.astype(int).value_counts()
    r = c / c.sum()
    r.index = pd.Index(['1st Class', '2nd Class', '3rd Class'])
    r.plot('bar')
    pass