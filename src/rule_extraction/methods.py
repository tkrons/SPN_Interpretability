import numpy as np
from simple_spn import functions as fn
from spn.structure.Base import Leaf, Condition, Rule
# from scipy.stats import kstest, chisquare
import random
import math


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
            raise ValueError('Not implemented')
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
    :param local: use local subpopulation for the calculation of the probabilities
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
        head_sup = fn.prob_spflow(local, head_rang)
        totalrang = get_spn_range(body.merge(head), root)
        total_sup = fn.prob_spflow(local, totalrang)
    elif real_data is not None:
        body_bool = body.apply(real_data, value_dict=value_dict,)
        body_sup = body_bool.mean()
        head_bool = head.apply(real_data, value_dict=value_dict)
        head_sup = head_bool.mean()
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
        elif m == 'recall':
            rec = total_sup / head_sup
            if rec > 1:
                raise ValueError()
            res.append(rec)
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
        #     p_nanb, _ =  fn.marg_rang(root, fn.not_rang(totalrang, value_dict)) # P(!A!B)
        #     res.append(total_sup + p_nanb)
        elif m == 'jaccard':
            # P(A | B) / (P(A) + P(B)− P(AB))
            res.append((total_sup / head_sup) / (body_sup + head_sup - total_sup))
        else:
            raise ValueError('Unknown metric: {}'.format(m))
    return res



def rule_pprint(r, forlatex=False):
    if forlatex:
        s='\['
    else:
        s= '['
    if isinstance(r, str):
        raise ValueError()
    if isinstance(r, Condition):
        r = [r] # rule like
    for i,c in enumerate(r):
        s += c.var
        s +=  ' = '
        s += str(c.threshold)
        if i < len(r) -1:
            s += ' AND '
    if forlatex:
        s += '\]'
    else:
        s += ']'
    return s

def df_display(df, forlatex = False):
    df = df.copy(deep=True)
    # # df.loc[['head', 'body']] = df[['head', 'body']].applymap(str, )
    # df['body'] = df['body'].apply(str, )
    # df['head'] = df['head'].apply(str, )


    df['head'] = df['head'].apply(lambda x: rule_pprint(x, forlatex))
    df['body'] = df['body'].apply(lambda x: rule_pprint(x, forlatex))
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


