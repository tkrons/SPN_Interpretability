import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from rule_extraction.methods import prior_distributions_lazy, head_compatible_body, rule_stats, get_labeled_rule
from simple_spn import functions as fn
from spn.structure.Base import Condition
from spn.structure.leaves.parametric.Parametric import Categorical


def get_leaf_rules(leaf, ):
    # em = float(np.argmax(leaf.p))
    rules = []
    assert len(leaf.rule.get_similar_conditions(leaf.scope[0])) == 0
    for i in range(len(leaf.p)):
        conseq = Condition(leaf.scope[0], np.equal, i)
        rules.append([leaf.rule, conseq])
    return rules


def get_interesting_leaves(spn, subpop, value_dict, top=5, min_conf = 'above_random'):
    prior_gen = prior_distributions_lazy()

    weight, leaves = subpop

    res_leaves, diffs = [], []
    for leaf in leaves:
        # we are not interested in leaves already contained in the rule AND sometimes columns may have a constant value
        if not len(leaf.rule) == 0 and len(leaf.rule.get_similar_conditions(leaf.scope[0])) == 0 and not len(leaf.p) == 1:
            if isinstance(leaf, Categorical):
                assert len(leaf.scope) == 1
                # p = list
                    # prior = prior_dist[leaf.scope[0]]
                    # prior = [1 - prior, prior]7
                if np.argmax(leaf.p) == 0:
                    continue # only positive
                elif len(leaf.p) > 2:
                    raise ValueError()
                prior = prior_gen.calculate_prior(spn, leaf, value_dict)
                js = jensenshannon(leaf.p, prior, )
                if min_conf == 'above_random':
                    if max(leaf.p) > prior[np.argmax(leaf.p)]:
                        diffs.append(js)
                        res_leaves.append(leaf)
                else:
                    raise NotImplementedError()
            else:
                raise ValueError('Not implemented')
    sort = sorted(zip(res_leaves, diffs, [weight]*len(res_leaves)), key=lambda x: x[1], reverse=True)[:top]
    return sort


def topdown_interesting_rules(spn, value_dict, metrics = ['sup', 'conf', 'head_sup', 'F', 'cosine_distance'],
                              full_value_dict = None, beta=1., labeled=True):
    subpops = fn.get_sub_populations(spn,)
    l = []
    for sub in subpops:
        l.extend(get_interesting_leaves(spn, sub, value_dict, top=6))
    sorted(l, key=lambda x: x[2])
    # rules = [[get_leaf_rules(leaf), diff, weight] for leaf, diff, weight in l]
    rules = []
    for leaf, diff, weight in l:
        leafrules = get_leaf_rules(leaf)
        for r in leafrules:
            if head_compatible_body(r[1], r[0], one_hot_vd=value_dict, full_value_dict=full_value_dict):
                rules.append([r, diff, weight])
    # rrules, rheads, rsup, rconf = [], [], [], []
    final_rules = []
    for lst in rules:
        #get confidence
        rule, head = lst[0]
        if len(rule) == 0 or len(head) == 0:
            continue
        stats = rule_stats(spn, rule, head, metrics=metrics, beta=beta, )

        if stats[metrics.index('F')] > 0.03:
        # if True:
            final_rules.append((head, rule, *stats))
    if labeled:
        final_rules_labeled = [(*get_labeled_rule(r[0], r[1], value_dict), *r[2:]) for r in final_rules]
    rule_df = pd.DataFrame(final_rules_labeled, columns=['head', 'body', *metrics])
    rule_df = rule_df.drop_duplicates(['body', 'head'])
    return rule_df