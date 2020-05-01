from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import numpy as np
import matplotlib.pyplot as plt
from simple_spn import functions as fn
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Leaf, Product, Sum, Condition, Rule
# from scipy.stats import kstest, chisquare
from scipy.spatial.distance import jensenshannon
import pandas as pd
import itertools

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

def get_interesting_leaves(spn, subpop, value_dict, top=5, min_conf = 0.75):
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
                prior = prior_gen.calculate_prior(spn, leaf, value_dict)
                js = jensenshannon(leaf.p, prior, )
                if len(leaf.p) > 2 or np.argmax(leaf.p) == 0:
                    continue # only positive
                if max(leaf.p) > min_conf:
                    diffs.append(js)
                    res_leaves.append(leaf)
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



def rule_stats(root, body, head, local=None, metrics=['sup', 'conf', 'F', 'head_sup']):
    rang = get_spn_range(body, root)
    res = []
    if local:
        body_sup = fn.prob_spflow(local, rang)
        head_rang = get_spn_range(head, local)
        head_sup = fn.prob_spflow(root, head_rang)
    else: #global
        body_sup = fn.prob_spflow(root, rang)
        head_rang = get_spn_range(head, root)
        head_sup = fn.prob_spflow(root, head_rang)
    # true_pos = np.logical_and(res, head.op(data[head.var], head.threshold))
    totalrang = get_spn_range(body.merge(head), root)
    if local:
        total_sup = fn.prob_spflow(local, totalrang)
    else:
        total_sup = fn.prob_spflow(root, totalrang)
    conf = total_sup / body_sup

    for m in metrics:
        if m == 'sup':
            res.append(body_sup)
        elif m == 'conf':
            res.append(conf)
        elif 'F' == m:
            res.append((2 * conf * body_sup) / (conf + body_sup))
        elif m == 'head_sup':
            res.append(head_sup)
    return res

def rule_stats_df():
    #todo
    pass

def topdown_interesting_rules(spn, value_dict, metrics = ['sup', 'conf', 'head_sup', 'F'], labeled=True):
    #todo rule propagate as left: var = threshold right: var != threshold
    # @ rule merge (x = 5) (x != 3) -> (x = 5) automatic regulation
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
            rules.append([r, diff, weight])
    # rrules, rheads, rsup, rconf = [], [], [], []
    final_rules = []
    for lst in rules:
        #get confidence
        rule, head = lst[0]
        if len(rule) == 0 or len(head) == 0:
            continue
        stats = rule_stats(spn, rule, head, metrics=metrics)

        if stats[1] > 0.5 and stats[0] > 0.0001:
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
    def __init__(self, min_target_js = 0.2, min_local_js = 0.,
                 body_max_len = 4, head_max_len = 1, min_global_conf = 0.75, min_local_p = 0.):
        self.body_max_len = body_max_len
        self.head_max_len = head_max_len
        self.min_local_p = min_local_p
        self.min_global_conf = min_global_conf
        self.min_target_js = min_target_js
        self.min_local_js = min_local_js
        self.min_local_p = min_local_p

        self.prior_gen = prior_distributions_lazy()
        self.rules_yielded = {} # True: yielded False: not qualified

    def intra_rules_df(self, spn, target_vars, value_dict, max_candidates=1000, labels=False):
        itr = itertools.islice(self.rule_iterate(spn, target_vars, value_dict), max_candidates)
        # rules = list(itr)
        rules = []
        i=0
        for e in itr:
            if i >= max_candidates:
                break
            rules.append(e)
            i += 1
        if labels:
            for lst in rules:

                head, body = lst[0], lst[1]
                head, body = get_labeled_rule(head, body, value_dict)
                lst[0], lst[1] = head, body
                # body, head = str(body), str(head)
        cols = ['head', 'body', 'sup', 'conf', 'F', 'head_sup']
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
                    if max(var.p) < self.min_local_p:
                        continue
                    leaf = var.scope[0]
                elif isinstance(var, int):
                    varp = p_from_scope(node, var, value_dict)
                    if max(varp) < self.min_local_p:
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
            head = Condition(target, np.equal, np.argmax(targetp))
            # local rule quality check
            l=list(yield_rules(eligable_leaves,))
            for r in l:
                if (head, r) not in self.rules_yielded:
                    stats = rule_stats(root, r, head, metrics=['sup', 'conf', 'F', 'head_sup'])
                    if stats[1] >= self.min_global_conf:
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



