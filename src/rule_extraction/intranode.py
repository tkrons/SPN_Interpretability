import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from rule_extraction.methods import prior_distributions_lazy, get_labeled_rule, p_from_scope, rule_stats
from spn.structure.Base import Sum, Product, Condition, Rule
from spn.structure.leaves.parametric.Parametric import Categorical


class IntraNode:
    def __init__(self, metrics, min_target_js = 0.2, min_local_js = 0.,
                 min_global_criterion=0., criterion='lift', min_recall=0.,
                 body_max_len = 6, head_max_len = 1, min_global_conf = 'above_random', min_local_p = 0., beta=1.,
                 ):
        '''
        @param criterion: primary cutoff criterion: must be in metrics
        @param min_global_criterion: cutoff value for criterion'''
        self.body_max_len = body_max_len
        self.head_max_len = head_max_len
        self.min_local_p = min_local_p
        self.min_global_conf = min_global_conf
        self.criterion = criterion
        self.min_target_js = min_target_js
        self.min_local_js = min_local_js
        self.min_local_p = min_local_p
        self.min_recall = min_recall
        self.min_global_criterion = min_global_criterion
        self.beta = beta
        self.metrics = metrics

        self.prior_gen = prior_distributions_lazy()
        self.rules_yielded = {} # True: yielded False: not qualified

    def intra_rules_df(self, df, spn, target_vars, value_dict, max_candidates=None, labels=False, rules_per_value=None):
        itr = self.rule_iterate(spn, target_vars, value_dict)
        self.rules_yielded = {}
        self.df = df
        # rules = list(itr)
        rules = []
        target_values = {t: {k: 0 for k in value_dict[t][2].keys()} for t in target_vars}
        i=0
        notchanged=0
        max_rules = df[df.columns[target_vars]].nunique().sum() * rules_per_value
        for e in itr:
            if notchanged > 1000 and sum([sum(v.values()) for v in target_values.values()]) > 0.7 * max_rules:
                print('Stopping early')
                break
            if rules_per_value:
                head = e[0]
                vals = target_values[head.var]
                if vals[head.threshold] < rules_per_value:
                    rules.append(e)
                    vals[head.threshold] += 1
                    i += 1
                else:
                    notchanged+=1
            elif max_candidates:
                if i >= max_candidates:
                    break
                rules.append(e)
                i += 1
            else:
                raise ValueError()
        print('len rules after iter: {}'.format(len(rules)))
        if labels:
            for lst in rules:

                head, body = lst[0], lst[1]
                head, body = get_labeled_rule(head, body, value_dict)
                lst[0], lst[1] = head, body
                # body, head = str(body), str(head)
        cols = ['head', 'body', *self.metrics]
        df = pd.DataFrame(rules, columns=cols, )
        return df.sort_values('F', ascending=False)

    def rule_iterate(self, spn, target_vars, value_dict):
        '''
        recursively yield rules according to IntraNode. for a more user friendly output use intra_rules_df(...).
        :param spn:
        :param target_vars:
        :param value_dict:
        :return:
        '''
        #reset
        self.rules_yielded = {}  # True: yielded successfully False: not qualified

        def _recurse_spn_local_rules(node, ):
            #first bottom up:
            for c in node.children:
                if isinstance(c, Sum) or isinstance(c, Product):
                        # yield from _recurse_spn_local_rules(c)
                        for e in _recurse_spn_local_rules(c):
                            yield e
            #then local rule
            if node.id != 0:
                local_targets = set(target_vars).intersection(set(node.scope))
                for target in local_targets:
                    p = p_from_scope(node, target, value_dict)
                    # if categorical TODO non-categorical data?
                    if max(p) >= self.min_local_p:
                        # prior = prior_dist[leaf.scope[0]]
                        # prior = [1 - prior, prior]7
                        prior = self.prior_gen.calculate_prior(spn, target, value_dict)
                        js = jensenshannon(p, prior, )
                        if js >= self.min_target_js:
                            other_vars = node.scope.copy()
                            other_vars.pop(other_vars.index(target))
                            # yield from _leaves_target_allrules(node, target, other_vars, value_dict, root=spn, targetp=p)
                            for res in _leaves_target_allrules(node, target, other_vars, value_dict, root=spn, targetp=p):
                                yield res
            # end recurse method

        def _leaves_target_allrules(node, target, vars, value_dict, root=None, targetp=None, ncandidates=None,):
            eligable_leaves = []

            for var in vars:
                if isinstance(var, Categorical):
                    varp = var.p
                    if self.min_local_p and max(var.p) < self.min_local_p:
                        continue
                    leaf = var.scope[0]
                elif isinstance(var, int):
                    varp = p_from_scope(node, var, value_dict)
                    if self.min_local_p and max(varp) < self.min_local_p:
                        continue
                    leaf = var
                else:
                    raise ValueError(type(var))
                if self.min_local_js:
                    leaf_prior = self.prior_gen.calculate_prior(root, var, value_dict)
                    js = jensenshannon(leaf_prior, varp)
                    if js >= self.min_local_js:
                        eligable_leaves.append(leaf)
                else:
                    eligable_leaves.append(leaf)

            def yield_rules(eligable_leaves, ):
                # def __powerset(iterable):
                #     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
                #     s = list(iterable)
                #     max_len = min((self.body_max_len, len(s)))
                #     for r in range(max_len):
                #         for e in itertools.combinations(s, r):
                #             yield e
                #     # all_sets = itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(max_len))
                #     # return itertools.islice(all_sets, 1, ncandidates)
                #     # for e in all_sets:
                #     #     yield e
                # rules = []

                s = list(eligable_leaves)
                max_len = min((self.body_max_len, len(s)))
                for r in range(max_len):
                    for rule_leaves in itertools.combinations(s, r):
                # for rule_leaves in __powerset(eligable_leaves):
                        if len(rule_leaves) > self.body_max_len:
                            break
                        conditions = []
                        for var in rule_leaves:
                            if isinstance(var, Categorical):
                                conditions.append(Condition(var.scope[0], np.equal, np.argmax(var.p)))
                            elif isinstance(var, int):
                                varp = p_from_scope(node, var, value_dict)
                                if max(varp) >= self.min_local_p:
                                    conditions.append(Condition(var, np.equal, np.argmax(varp)))
                            else:
                                raise ValueError(var)
                        if conditions:
                            yield Rule(conditions)

            if isinstance(target, Categorical):
                targetp = target.p
                target = target.scope[0]
            # head = Condition(target, np.equal, np.argmax(targetp))
            # todo use all heads where p[head] > p[prior_head] ?
            heads = []
            for val in range(len(targetp)):
                if targetp[val] > self.prior_gen.calculate_prior(spn, target, value_dict)[val]:
                    heads.append(Condition(target, np.equal, val))
            # local rule quality check
            l=list(yield_rules(eligable_leaves,))
            for r in l:
                for head in heads:
                    if (head, r) not in self.rules_yielded:
                        # exact stats only evaluation TODO if you don't need evaluation, only use spn_stats
                        real_stats = rule_stats(root, r, head, metrics=self.metrics,
                                           real_data=self.df, beta=self.beta, value_dict=value_dict)
                        # local quickly calculated stats
                        spn_stats = rule_stats(root, r, head, metrics=self.metrics, beta=self.beta, value_dict=value_dict)
                        if isinstance(self.min_global_conf, str) and self.min_global_conf == 'above_random':
                            min_conf = 1. / len(value_dict[head.var][2])
                        else:
                            min_conf = self.min_global_conf
                        if self.criterion == 'cosine_distance':
                            better_than_crit = spn_stats[self.metrics.index(self.criterion)] <= self.min_global_criterion
                        else:
                            better_than_crit = spn_stats[self.metrics.index(self.criterion)] >= self.min_global_criterion


                        if spn_stats[self.metrics.index('conf')] >= min_conf \
                            and spn_stats[self.metrics.index('recall')] >= self.min_recall \
                            and better_than_crit:
                            self.rules_yielded[(head, r)] = True
                            yield [head, r, *real_stats]
                        else:
                            self.rules_yielded[(head, r)] = False
            # leaves target rules END
        # recursion first call
        # yield from _recurse_spn_local_rules(spn)
        for e in _recurse_spn_local_rules(spn):
            yield e