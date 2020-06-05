
import os
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder, OnehotTransactions

def onehot2transactional(df, ):
    assert df.max().max() == 1, 'Not onehot encoded'
    te = TransactionEncoder()
    te.columns_ = list(df.columns)
    te.columns_mapping_ = {}
    for i, c in enumerate(te.columns_):
        te.columns_mapping_[i] = c

    res = te.inverse_transform(df.values)
    return res