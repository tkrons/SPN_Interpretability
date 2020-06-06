from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import numpy as np
import matplotlib.pyplot as plt
from simple_spn import functions as fn
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Leaf, Product, Sum, Condition, Rule
# from scipy.stats import kstest, chisquare
import random
import math
from scipy.spatial.distance import jensenshannon
import pandas as pd
import itertools

from spn.experiments.AQP.Ranges import NominalRange, NumericRange

class prior_distributions_lazy():
    def __init__(self,):
        self.scope_dict = {}

    def calculate_prior(self, spn, target, value_dict):
        '''which var to lazily calculate prior from (scope)
        and any leaf with the apprioriate var to get len(leaf.p)'''

        if isinstance(target, Leaf):
            var = target.scope[0]
        elif isinstance(target, int):
            var = target
        else:
            raise ValueError(target)
        kind, name, values = value_dict[var]
        if var in self.scope_dict:
            return self.scope_dict[var]
        elif kind == 'discrete': # need to calculate
            rang = [np.NaN] * len(spn.scope)
            prior_dist = []
            for i in range(len(values.keys())):
                rang[var] = i
                prior_dist.append(fn.prob_spflow(spn, rang))
            self.scope_dict[var] = prior_dist
        else:
            raise ValueError(kind + ' not supported')
        return prior_dist

        # elif isinstance(target, int):
        #     if var in self.scope_dict:
        #         return self.scope_dict[var]
        #     elif isinstance(target, Categorical):  # need to calculate
        #         rang = [np.NaN] * len(spn.scope)
        #         prior_dist = []
        #         for i, _ in enumerate(target.p):
        #             rang[var] = i
        #             prior_dist.append(fn.prob_spflow(spn, rang))
        #         self.scope_dict[var] = prior_dist
        #     else:
        #         raise ValueError(str(type(target)) + ' not supported')
        #     return prior_dist

def get_leaf_rules(leaf, ):
    # em = float(np.argmax(leaf.p))
    rules = []
    assert len(leaf.rule.get_similar_conditions(leaf.scope[0])) == 0
    for i in range(len(leaf.p)):
        conseq = Condition(leaf.scope[0], np.equal, i)
        rules.append([leaf.rule, conseq])
    return rules

def get_spn_range(rule, spn):
    '''get the range of the minimal rule body [np.NaN ... value ... np.NaN]
    with values such that rule.apply(rang) => True'''
    rang = [np.NaN] * len(spn.scope)
    if isinstance(rule, Rule):
        for c in rule:
            if c.op == np.equal:
                rang[c.var] = c.threshold
            elif c.op in [np.greater, np.greater_equal, np.less, np.less_equal]:
                raise ValueError('Not implemented') #todo for gaussian distributions?
    elif isinstance(rule, Condition):
        c = rule
        if c.op == np.equal:
            rang[c.var] = c.threshold
        elif c.op in [np.greater, np.greater_equal, np.less, np.less_equal]:
            raise ValueError('Not implemented')  # todo for gaussian distributions?
    return rang

def get_labeled_rule(head, body, value_dict):
    assert isinstance(body, Rule) and isinstance(head, Condition)
    '''simple EM to get the (body,head) and assign labels according to columns.
    only len(conseq)==1 for now'''
    #unlabeled:
    new_cs = []
    for cond in body:
        _, var_name, i2str = value_dict[cond.var]
        # new_cs.append(Condition(value_dict[cond[0]], cond[1], cond[2]))
        new_cs.append(Condition(var_name, cond.op, i2str[cond.threshold]))
    _, head_name, head_dict = value_dict[head.var]
    assert len(body.get_similar_conditions(head.var)) == 0
    labeled_head = Condition(head_name, np.equal, head_dict[head.threshold])
    return labeled_head, Rule(new_cs)

def format_mlxtend2rule_ex(head = None, body = None, ):
    if head:
        assert len(head) == 1, 'longer heads not implemented'
        res_h = Condition(tuple(head)[0], np.equal, 1)
    if body:
        conds = []
        for it in body:
            conds.append(Condition(it, np.equal, 1))
    if body and head:
        return res_h, Rule(conds)
    elif head:
        return res_h
    elif body:
        return Rule(conds)

def fbeta_score(prec, rec, beta):
    beta2 = beta ** 2
    return ((1 + beta2) * prec * rec) / (rec + (beta2 * prec))

def conviction(head_sup, conf):
    # conviction: http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
    if conf == 1:
        return np.NaN
    else:
        return np.divide(1 - head_sup, 1 - conf)

def reverse_label_rule(body, head, value_dict):
    new_cs = []
    for cond in body:
        for _, var_name, i2str in value_dict:
            if var_name == cond[0]:
                for i, s in i2str.items():
                    if s == cond.threshold:
                        new_cs.append(Condition(var_name, cond.op, i))
    # if head:
    #
    # return Rule(new_cs), conseq

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


def p_from_scope(node, target, value_dict):
    rang = [np.NaN] * len(value_dict)
    p = []
    values = value_dict[target][2]
    for i in range(len(values)): # values are 0 to N
        rang[target] = i
        p.append(fn.prob_spflow(node, rang))
    return p

def rule_str2idx(r, value_dict):
    if isinstance(r, Condition):
        r = [r] #conditions

    res_conds = []
    for cond in r:
        for i, (_, name, vals) in value_dict.items():
            if  name == cond.var:
                inv_vals = {v: k for k, v in vals.items()}
                res_conds.append(Condition(i, np.equal, inv_vals[cond.threshold]))
    assert len(res_conds) ==  len(r)

    if isinstance(r, Condition):
        return res_conds[0]
    else:
        return Rule(res_conds)


def rule_stats(root, body, head, metrics, local=None, beta=1, value_dict=None, real_data=None):
    '''
    :param root:
    :param body:
    :param head:
    :param metrics:
    :param local:
    :param beta:
    :param value_dict:
    :param real_data: dont use SPN, instead EXACT values for probabilities
    :return:
    '''
    if isinstance(head.var, str):
        body = rule_str2idx(body, value_dict)
        head = rule_str2idx(head, value_dict)
    rang = get_spn_range(body, root)
    res = []
    if local:
        body_sup = fn.prob_spflow(local, rang)
        head_rang = get_spn_range(head, local)
        head_sup = fn.prob_spflow(root, head_rang)
        totalrang = get_spn_range(body.merge(head), root)
        total_sup = fn.prob_spflow(local, totalrang)
    elif real_data:
        body_bool = body.apply(real_data, value_dict,)
        body_sup = body_bool.mean()
        head_bool = head.apply(real_data, value_dict)
        head_sup = body_bool.mean()
        total_sup = np.logical_and(body_bool, head_bool).mean()
    else: #global
        body_sup = fn.prob_spflow(root, rang)
        head_rang = get_spn_range(head, root)
        head_sup = fn.prob_spflow(root, head_rang)
        totalrang = get_spn_range(body.merge(head), root)
        total_sup = fn.prob_spflow(root, totalrang)
    # true_pos = np.logical_and(res, head.op(data[head.var], head.threshold))

    conf = total_sup / body_sup

    for m in metrics:
        if m == 'sup':
            res.append(body_sup)
        elif m == 'conf':
            res.append(conf)
        elif 'F' == m:
            # res.append((2 * conf * body_sup) / (conf + body_sup))
            res.append(fbeta_score(conf, body_sup, beta))
        elif m == 'head_sup':
            res.append(head_sup)
        elif m == 'conviction':
            res.append(conviction(head_sup, conf))
        elif m == 'lift':
            res.append(conf / head_sup)
        elif m == 'interestingness':
            res.append(total_sup / body_sup - head_sup)
        elif m == 'cosine_distance':
            res.append(1 - total_sup / np.sqrt( body_sup * head_sup ))
        elif m == 'PiSh':
            res.append(total_sup - body_sup * head_sup)
        elif m == 'leverage':
            res.append(conf - head_sup * body_sup)
        # elif m == 'success_rate':
        #     p_nanb, _ =  fn.marg_rang(root, fn.not_rang(totalrang, value_dict)) # P(!A!B) todo negate rang_total
        #     res.append(total_sup + p_nanb)
        elif m == 'jaccard':
            # P(A | B) / (P(A) + P(B)− P(AB))
            res.append((total_sup / head_sup) / (body_sup + head_sup - total_sup))
        else:
            raise ValueError('Unknown metric: {}'.format(m))
    return res

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
#todo add rules to increase data coverage: ie (body1, head1) -> choose body2 to maximize fn.prob(spn, rang(ALL - body1))
#todo m-estimate https://scholar.google.de/scholar?q=m-estimate+rule+induction&hl=de&as_sdt=0&as_vis=1&oi=scholart

class IntraNode:
    def __init__(self, metrics, min_target_js = 0.2, min_local_js = 0.,
                 min_global_criterion=0., criterion='lift',
                 body_max_len = 4, head_max_len = 1, min_global_conf = 'above_random', min_local_p = 0., beta=1.,
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
        self.min_global_criterion = min_global_criterion,
        self.beta = beta
        self.metrics = metrics

        self.prior_gen = prior_distributions_lazy()
        self.rules_yielded = {} # True: yielded False: not qualified

    def intra_rules_df(self, spn, target_vars, value_dict, max_candidates=1000, labels=False, rules_per_value=None):
        itr = self.rule_iterate(spn, target_vars, value_dict)
        # rules = list(itr)
        rules = []
        target_values = {t: {k: 0 for k in value_dict[t][2].keys()} for t in target_vars}
        i=0
        for e in itr:
            if rules_per_value:
                head = e[0]
                vals = target_values[head.var]
                if vals[head.threshold] < rules_per_value:
                    rules.append(e)
                    vals[head.threshold] += 1

            elif max_candidates:
                if i >= max_candidates:
                    break
                rules.append(e)
                i += 1
            else:
                raise ValueError()
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
                    # if categorical TODO non-categorical?
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

        def _leaves_target_allrules(node, target, vars, value_dict, root=None, targetp=None, ncandidates=None,
                                    ):
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
            # todo use all heads where p[head] > p[prior_head] ??
            heads = []
            for val in range(len(targetp)):
                if targetp[val] > self.prior_gen.calculate_prior(spn, target, value_dict)[val]:
                    heads.append(Condition(target, np.equal, val))
            # local rule quality check
            l=list(yield_rules(eligable_leaves,))
            for r in l:
                for head in heads:
                    if (head, r) not in self.rules_yielded:
                        stats = rule_stats(root, r, head, metrics=self.metrics, beta=self.beta, value_dict=value_dict)
                        if isinstance(self.min_global_conf, str) and self.min_global_conf == 'above_random':
                            min_conf = 1. / len(targetp)
                        else:
                            min_conf = self.min_global_conf
                        if stats[self.metrics.index('conf')] >= min_conf and \
                                stats[self.metrics.index(self.criterion)] >= self.min_global_criterion:
                            self.rules_yielded[(head, r)] = True
                            yield [head, r, *stats]
                        else:
                            self.rules_yielded[(head, r)] = False
            # leaves target rules END
        # recursion first call
        # yield from _recurse_spn_local_rules(spn)
        for e in _recurse_spn_local_rules(spn):
            yield e

# END CLASS

def df_display(df, ):
    df = df.copy(deep=True)
    # df.loc[['head', 'body']] = df[['head', 'body']].applymap(str, )
    df['body'] = df['body'].apply(str, )
    df['head'] = df['head'].apply(str, )
    return df

def df2labeled(df, value_dict):
    '''broken for some reason'''
    res_df = df.copy()
    r = res_df[['head', 'body']].apply(lambda x: get_labeled_rule(x['head'], x.body, value_dict), axis = 1)
    body, head = list(zip(*r))
    res_df = res_df.assign(head=head, body=body)
    return res_df

# def exact_stats(df, rules, heads, metrics=['exact_conf']):
#     res=[]
#     for body, head in zip(rules, heads):
#         sup =
#         for m in metrics:

# def rule_onehot2categorical(r, value_dict, vd_onehot):
#     iscond = False
#     if isinstance(r, Condition):
#         r = [r]
#         iscond = True
#     else:
#         for cond in r:
#             pass

def slice_head_iter(iter, n):
    res, count = [], 0
    for i in iter:
        if count >= n:
            break
        res.append(i)
        count += 1
    return res

def append_rules(rules):
    '''also conditions'''
    conds = []
    for r in rules:
        if isinstance(r, Rule):
            conds.extend(list(r))
        elif isinstance(r, Condition):
            conds.append(r)
    return conds

def head_compatible_body(head, body, one_hot_vd, full_value_dict):
    _, name, values = one_hot_vd[head.var]
    # name muss in fullvd[2].values() sein
    for _, (_, column_name, attributes) in full_value_dict.items():
        if name in attributes.values() or column_name in name:
            # found the column
            for c in body:
                cname = one_hot_vd[c.var][1]
                if cname in attributes.values() or column_name in cname:
                    return False
            return True
    return True

def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = random.random() < x-int(x)
    round_func = math.ceil if is_up else math.floor
    return sign * round_func(x)
