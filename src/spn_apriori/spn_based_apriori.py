# Sebastian Raschka 2014-2019
# myxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import simple_spn.functions as fn
import typing
import itertools

def generate_new_combinations(old_combinations):
    """
    Generator of all combinations based on the last state of Apriori algorithm
    Parameters
    -----------
    old_combinations: np.array
        All combinations with enough support in the last step
        Combinations are represented by a matrix.
        Number of columns is equal to the combination size
        of the previous step.
        Each row represents one combination
        and contains item type ids in the ascending order
        ```
               0        1
        0      15       20
        1      15       22
        2      17       19
        ```

    Returns
    -----------
    Generator of all combinations from the last step x items
    from the previous step. Every combination is a tuple
    of item type ids in the ascending order.
    No combination other than generated
    do not have a chance to get enough support
    df(columns=[0,1,2,3])
    combinations with len == 2
    [[0,1], [0,2], [0,3], [1,2] ... [2,3]]

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

    """

    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        max_combination = max(old_combination)
        for item in items_types_in_previous_step:
            if item > max_combination:
                res = tuple(old_combination) + (item,)
                yield res

def generate_rules_apriori(
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    spn,
    min_confidence: float,
    num_transactions: int,
    verbosity: int = 0,
):
    """
    Bottom up algorithm for generating association rules from itemsets, very
    similar to the fast algorithm proposed in the original 1994 paper by
    Agrawal et al.

    The algorithm is based on the observation that for {a, b} -> {c, d} to
    hold, both {a, b, c} -> {d} and {a, b, d} -> {c} must hold, since in
    general conf( {a, b, c} -> {d} ) >= conf( {a, b} -> {c, d} ).
    In other words, if either of the two one-consequent rules do not hold, then
    there is no need to ever consider the two-consequent rule.

    Parameters
    ----------
    itemsets : dict of dicts
        The first level of the dictionary is of the form (length, dict of item
        sets). The second level is of the form (itemset, count_in_dataset)).
    min_confidence :  float
        The minimum confidence required for the rule to be yielded.
    num_transactions : int
        The number of transactions in the data set.
    verbosity : int
        The level of detail printing when the algorithm runs. Either 0, 1 or 2.

    Examples
    --------
    >>> itemsets = {1: {('a',): 3, ('b',): 2, ('c',): 1},
    ...             2: {('a', 'b'): 2, ('a', 'c'): 1}}
    >>> list(generate_rules_apriori(itemsets, 1.0, 3))
    [{b} -> {a}, {c} -> {a}]
    """
    # Validate user inputs
    if not (
        (0 <= min_confidence <= 1)
    ):
        raise ValueError("`min_confidence` must be a number between 0 and 1.")

    if not (
        (num_transactions >= 0)
    ):
        raise ValueError("`num_transactions` must be a number greater than 0.")

    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    if verbosity > 0:
        print("Generating rules from itemsets.")

    # For every itemset of a perscribed size
    for size in itemsets.keys():

        # Do not consider itemsets of size 1
        if size < 2:
            continue

        if verbosity > 0:
            print(" Generating rules of size {}.".format(size))

        # For every itemset of this size
        for itemset in itemsets[size].keys():

            # Special case to capture rules such as {others} -> {1 item}
            for removed in itertools.combinations(itemset, 1):

                # Compute the left hand side
                lhs = set(itemset).difference(set(removed))
                lhs = tuple(sorted(list(lhs)))

                # If the confidence is high enough, yield the rule
                # conf = count(itemset) / count(lhs)

                if conf >= min_confidence:
                    yield Rule(
                        lhs,
                        removed,
                        count(itemset),
                        count(lhs),
                        count(removed),
                        num_transactions,
                    )

            # Generate combinations to start off of. These 1-combinations will
            # be merged to 2-combinations in the function `_ap_genrules`
            H_1 = list(itertools.combinations(itemset, 1))
            yield from _ap_genrules(
                itemset, H_1, itemsets, min_confidence, num_transactions
            )

    if verbosity > 0:
        print("Rule generation terminated.\n")


def spn_apriori(df, spn, value_dict, min_support=0.5, use_colnames=False, max_len=None, verbose=0, ):
    """Get frequent itemsets from a one-hot DataFrame

    Parameters
    -----------
    df : pandas DataFrame or pandas SparseDataFrame
      pandas DataFrame the encoded format.
      The allowed values are either 0/1 or True/False.
      For example,

    ```
             Apple  Bananas  Beer  Chicken  Milk  Rice
        0      1        0     1        1     0     1
        1      1        0     1        0     0     1
        2      1        0     1        0     0     0
        3      1        1     0        0     0     0
        4      0        0     1        1     1     1
        5      0        0     1        0     1     1
        6      0        0     1        0     1     0
        7      1        1     0        0     0     0
    ```
    spn:
        The SPN to query for calculation of support.

    value_dict:
        Dict translating the values to indices for the SPN

    min_support : float (default: 0.5)
      A float between 0 and 1 for minumum support of the itemsets returned.
      The support is computed as the fraction
      `transactions_where_item(s)_occur / total_transactions`.

    use_colnames : bool (default: False)
      If `True`, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths (under the apriori condition) are evaluated.

    verbose : int (default: 0)
      Shows the number of iterations if >= 1 and


    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that are >= `min_support` and < than `max_len`
      (if `max_len` is not None).
      Each itemset in the 'itemsets' column is of type `frozenset`,
      which is a Python built-in type that behaves similarly to
      sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

    """

    def _support(df, spn, value_dict, itemsets, n_columns, _x=None, _n_rows=None):
        """DRY private method to calculate support as the
        row-wise sum of values / number of rows

        Parameters
        -----------

        _x : matrix of bools or binary

        _n_rows : numeric, number of rows in _x

        _is_sparse : bool True if _x is sparse

        Returns
        -----------
        np.array, shape = (n_rows, )

        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

        """
        def search_value_dict(trans_item, value_dict):
            res = []
            for i, [type, col_name, values] in value_dict.items():
                for val, name in values.items():
                    if trans_item == name or (trans_item == col_name and name == True): # one hot case
                        res.append((i, name, val))
            assert len(res) in [0,1], "multiple values for {} in value_dict found: {}".format(trans_item, res)
            return res[0]

        # if spn.columns == x.columns => easy peasy spflow_prop(1, 0, 1, 1) = ?
        # elif cols != cols => get spn.columns and align spflow_prop(1, 0, np.NaN, 0)
        # out = (np.sum(_x, axis=0) / _n_rows)
        ranges = []
        for comb in itemsets:
            rang = [np.NaN] * len(value_dict)
            for col in comb:
                # rang[col] = 1
                val_string = df.columns[col]
                if spn_onehot_encoded:
                    rang[col] = 1.
                else: #translate to tabular spn indices:
                    i, col_name, val = search_value_dict(val_string, value_dict)
                    if np.isnan(rang[i]): # still unassigned
                        rang[i] = val
                    else: # already assigned therefore impossible. eg itemset (male, female)
                        rang = 'IMPOSSIBLE SET'
            ranges.append(rang)
        impossible_sets = set([i for i, rang in enumerate(ranges) if rang == 'IMPOSSIBLE SET'])
        possible = set(range(len(ranges))) - impossible_sets
        if len(possible) > 0:
            ranges = np.array([rang for i, rang in enumerate(ranges) if i in possible])
            probs = fn.probs_iter_spflow(spn, ranges)
            # probs = fn.probs_spflow(spn, np.array([rang for i, rang in enumerate(ranges) if i in possible]))
            # probs = np.nditer(probs)
        res = [np.NaN] * len(ranges)
        for i, _ in enumerate(ranges):
            if i in sorted(list(possible)):
                res[i] = next(probs)
            else:
                res[i] = 0.
        assert len(res) == len(itemsets)
        return np.array(res)


    # end def: _support

    # check whether SPN uses one hot encoded data (=binary data)
    # onehotencoded makes _support calculation slightly easier)
    spn_onehot_encoded = len(set([vals
         for x in value_dict.values()
         for vals in x[2].values()])) == 2

    idxs = np.where((df.values != 1) & (df.values != 0))
    # tim: idxs[0] is only sanity check idxs[1] is not used
    if len(idxs[0]) > 0:
        val = df.values[idxs[0][0], idxs[1][0]]
        s = ('The allowed values for a DataFrame'
             ' are True, False, 0, 1. Found value %s' % (val))
        raise ValueError(s)

    is_sparse = hasattr(df, "to_coo")
    if is_sparse:
        if not isinstance(df.columns[0], str) and df.columns[0] != 0:
            raise ValueError('Due to current limitations in Pandas, '
                             'if the SparseDataFrame has integer column names,'
                             'names, please make sure they either start '
                             'with `0` or cast them as string column names: '
                             '`df.columns = [str(i) for i in df.columns`].')
        X = df.to_coo().tocsc()
    else:
        X = df.values
    # trivial itemsets:
    first_combin = np.array(list(range(len(df.columns)))).reshape([-1, 1])
    support = _support(df, itemsets=first_combin, n_columns=len(df.columns), _x=X, _n_rows=X.shape[0], spn=spn, value_dict=value_dict)
    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    rows_count = float(X.shape[0])

    iter_count = 0
    all_ones = np.ones((int(rows_count), 1))

    while max_itemset and max_itemset < (max_len or float('inf')):
        #always the low_memory version

        next_max_itemset = max_itemset + 1
        combin = generate_new_combinations(itemset_dict[max_itemset])

        combin = np.array(list(combin))

        if combin.size == 0:
            break
        if verbose:
            print(
                '\rProcessing %d combinations | Sampling itemset size %d' %
                (combin.size, next_max_itemset), end="")

        if is_sparse: #never sparse
            _bools = X[:, combin[:, 0]] == all_ones
            for n in range(1, combin.shape[1]):
                _bools = _bools & (X[:, combin[:, n]] == all_ones)
        else:
            _bools = np.all(X[:, combin], axis=2)
        #_bools.shape: [n_transactions, current_combinations]
        support = _support(df, itemsets=combin, n_columns=len(df.columns), spn = spn, value_dict= value_dict, _x=np.array(_bools), _n_rows=rows_count,)
        _mask = (support >= min_support).reshape(-1)
        if any(_mask):
            itemset_dict[next_max_itemset] = np.array(combin[_mask])
            support_dict[next_max_itemset] = np.array(support[_mask])
            max_itemset = next_max_itemset
        else:
            # Exit condition
            break

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]])

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: frozenset([
                                                      mapping[i] for i in x]))
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used

    return res_df
