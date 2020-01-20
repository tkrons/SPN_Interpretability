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
    

if __name__ == '__main__':
    test_get_titanic()