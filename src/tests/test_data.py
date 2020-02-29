'''
Created on 22.10.2019

@author: Moritz
'''

from util import io
from data import real_data


def test_get_titanic():
    df, value_dict, param_types = real_data.get_titanic()
    
    
    io.print_pretty_table(df.head(10))
    print(value_dict)
    print(param_types)
    
def test_get_OnlineRetail():
    df, value_dict,  param_types = real_data.get_real_data('OnlineRetail')
    io.print_pretty_table(df.head(10))

def test_get_Ecommerce():
    df, value_dict, param_types = real_data.get_real_data('Ecommerce')
    io.print_pretty_table(df.head(10))
    assert len(df.columns) == len(value_dict)

def test_get_RecordLink():
    df, value_dict, param_types = real_data.get_real_data('RecordLink', only_n_rows=10000, seed = 5)
    io.print_pretty_table(df.head(10))
    assert len(df.columns) == len(value_dict)

def test_get_adult_one_hot():
    df, value_dict, param_types = real_data.get_real_data('adult_one_hot', only_n_rows=10000, seed=5)
    io.print_pretty_table(df.head(10))
    assert len(df.columns) == len(value_dict)

def test_get_lending():
    df, value_dict, param_types = real_data.get_real_data('lending', only_n_rows=10000, seed=5)
    io.print_pretty_table(df.head(10))
    assert len(df.columns) == len(value_dict)


if __name__ == '__main__':
    test_get_titanic()