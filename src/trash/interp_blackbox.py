'''
Created on 17.06.2019

@author: Moritz
'''

import numpy as np

from spn.structure.Base import Context

from simple_spn import spn_util


def create_p_value_dataset():
    
    from data import R_wrapper_data
    from evaluation import evaluator
    from methods import R_wrapper
    from ml import dataset_creator, pre_processing
    from util import io
    
    tss = R_wrapper_data.get_noufaily_configuration(25, num_ts=10, num_weeks=624, k=5, random_seed=100)
    
    baseline = 7
    #Init 7 time points
    method_descriptions1 = [
        {"method": R_wrapper.get_EARS_score, "parameters" : {"method":"C1", "baseline":baseline, "alpha":0.005}},
        {"method": R_wrapper.get_EARS_score, "parameters" : {"method":"C2", "baseline":baseline, "alpha":0.005}},
        {"method": R_wrapper.get_EARS_score, "parameters" : {"method":"C3", "baseline":baseline, "alpha":0.005}},
        #{"method": R_wrapper.get_EARS_score, "parameters" : {"method":"C4", "baseline":6, "alpha":0.05}},
        {"method": R_wrapper.get_Bayes_score, "parameters" : {"b":0, "w":baseline, "actY":True, "alpha":0.005}},
        {"method": R_wrapper.get_RKI_score, "parameters" : {"b":0, "w":baseline, "actY":True}},
        ]
    method_results = evaluator.evaluate_method_results(tss, method_descriptions1)
    
    ds = None
    for ts in tss:
        if ds is None:
            ds = dataset_creator.pValue_dataset(ts, method_results, pre_process=pre_processing.peak)
        else:
            new_ds = dataset_creator.pValue_dataset(ts, method_results, pre_process=pre_processing.peak)
            ds.df = ds.df.append(new_ds.df)
    
    
    ds.df.drop(["ground_truth"], inplace=True, axis=1)
    
    
    ds.df.replace({"target" : "False"}, {"target" : 0}, inplace=True)
    ds.df.replace({"target" : "True"}, {"target" : 1}, inplace=True)
          
    io.print_pretty_table(ds.df.head(100))

    from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
    return ds.df.values, [Gaussian,Gaussian,Gaussian,Gaussian,Gaussian,Categorical]
    
    


def generate_dataset():
    
    a = np.r_[np.random.normal(10, 5, (300, 1)), np.random.normal(20, 10, (700, 1))]
    b = np.r_[np.random.normal(3, 2, (300, 1)), np.random.normal(50, 10, (700, 1))]
    c = np.r_[np.random.normal(20, 3, (1000, 1))]
    train_data = np.c_[a, b, c]
    return train_data

def generate_gender_age_data(num_instances, rand_seed):    
    '''
    Name:
    "gender-age"
    
    Features:
    1st column : gender : {male,female}
    2nd column : student: {yes,no}
    3rd column : age    : continuous (Gaussian distribution)
    
    Correlations:
    P(gender=male) = 50%
    P(gender=male) = 50%
    P(student=yes|gender=m) = 30%
    P(student=yes|gender=f) = 80%
    P(age) = N(mu=20, sigma=3)
    '''
    
    np.random.seed(rand_seed)

    data = [] 
    for _ in range(num_instances):
        inst = []
        if np.random.random() < 0.5:
            inst.append(0)
            if np.random.random() < 0.3:
                inst.append(1)
            else:
                inst.append(0)
            inst.append(int(np.random.normal(25, 1)))
            inst.append(int(np.random.normal(25, 4)))
            
        else:
            inst.append(1)
            if np.random.random() < 0.8:
                inst.append(1)
            else:
                inst.append(0)     
            inst.append(int(np.random.normal(20, 1)))
            inst.append(int(np.random.normal(20, 3)))
            
        #inst.append(int(np.random.normal(20, 3)))
        data.append(inst)
    
    from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
    return np.array(data), [Categorical, Categorical, Gaussian, Gaussian]
    
    

def learn_parametric_spn(data, parametric_types):
    
    from spn.algorithms.LearningWrappers import learn_parametric
    ds_context = Context(parametric_types=parametric_types).add_domains(data)
    ds_context.add_domains(data)
    spn = learn_parametric(data, ds_context, min_instances_slice=100, threshold=0.01)
    return spn


def marginalize(spn, keep):
    from spn.algorithms import Marginalization
    return Marginalization.marginalize(spn, keep)


def get_nodes_with_weight(node, feature_id):
    from spn.structure.Base import Sum, Product, Leaf
    
    if feature_id in node.scope:
        if isinstance(node, Leaf):
            return [(1.0, node)]
        elif isinstance(node, Sum):
            weighted_nodes = []
            for i, child in enumerate(node.children):
                weight = node.weights[i]
                for (r_weight, r_node) in get_nodes_with_weight(child, feature_id):
                    weighted_nodes.append((weight*r_weight, r_node))
            return weighted_nodes
            
        elif isinstance(node, Product):
            weighted_nodes = []
            for i, child in enumerate(node.children):
                if feature_id in child.scope:
                    weighted_nodes += get_nodes_with_weight(child, feature_id)      
            return weighted_nodes
        else:
            raise Exception("Invalide node: " + str(node))



def spn_for_evidence(spn, evidence_ranges, node_likelihood=None, distribution_update_ranges=None):
    from spn.structure.Base import Sum, Product, Leaf, assign_ids
    from spn.algorithms.TransformStructure import Prune
    from spn.algorithms.Validity import is_valid
    from copy import deepcopy
    
    def spn_for_evidence_recursive(node):
        
        if isinstance(node, Leaf):
            if len(node.scope) > 1:
                raise Exception("Leaf Node with |scope| > 1")
            
            if evidence_ranges[node.scope[0]] is not None:
                t_node = type(node)
                if t_node in node_likelihood:
                    ranges = np.array([evidence_ranges])
                    prob =  node_likelihood[t_node](node, ranges, node_likelihood=node_likelihood)[0][0]
                    if prob == 0:
                        newNode = deepcopy(node)
                    else:
                        newNode = deepcopy(node)
                        distribution_update_ranges[t_node](newNode, evidence_ranges[node.scope[0]])
                else:
                    raise Exception('No log-likelihood method specified for node type: ' + str(type(node)))
            else:
                prob = 1
                newNode = deepcopy(node)
                
            return prob, newNode
            

        newNode = node.__class__()
        newNode.scope = node.scope

        if isinstance(node, Sum):
            new_weights = []
            new_childs = []
        
            for i, c in enumerate(node.children):
                prob, new_child  = spn_for_evidence_recursive(c)
                new_prob = prob * node.weights[i]
                if new_prob > 0:
                    new_weights.append(new_prob)
                    new_childs.append(new_child)
            
            new_weights = np.array(new_weights)
            newNode.weights = new_weights / np.sum(new_weights)
            newNode.children = new_childs
            return np.sum(new_weights), newNode
        
        
        elif isinstance(node, Product):
            new_childs = []
            
            new_prob = 1.
            for i, c in enumerate(node.children):
                prob, new_child = spn_for_evidence_recursive(c)
                new_prob *= prob
                new_childs.append(new_child)
                
            newNode.children = new_childs
            return new_prob, newNode

    prob, newNode = spn_for_evidence_recursive(spn)
    assign_ids(newNode)
    newNode = Prune(newNode)
    valid, err = is_valid(newNode)
    assert valid, err

    return prob, newNode




def visualize_Density(spn):
    
    
    from spn.experiments.AQP.Ranges import NominalRange, NumericRange
    from spn.algorithms import Inference
    from simple_spn.InferenceRange import categorical_likelihood_range, gaussian_likelihood_range
    from spn.structure.Base import Sum, Product
    from spn.algorithms.Inference import sum_likelihood, prod_likelihood
    from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
    from simple_spn.UpdateRange import categorical_update_range
    
    inference_support_ranges = {Gaussian        : None, 
                                Categorical     : categorical_likelihood_range,
                                Sum             : sum_likelihood,
                                Product         : prod_likelihood}
    
    distribution_update_ranges = {Gaussian        : None, 
                                Categorical     : categorical_update_range}
    
    import matplotlib.pyplot as plt  
    _, axes = plt.subplots(1, 5, figsize=(15,10), squeeze=False, sharey=False, sharex=True)
    
    space_start = 0.00
    space_end = 1.0
    steps = 100
    max_y = 5
    
    for i in range(5):
        x_vals = np.linspace(space_start,space_end,num=steps)
        ranges = []
        for x_val in x_vals:
            r = [None] * i + [NumericRange([[x_val]])] + [None] *(5-i)
            ranges.append(r)
        
        
        ranges = np.array(ranges)
    
        inference_support_ranges = {Gaussian        : gaussian_likelihood_range, 
                                    Categorical     : categorical_likelihood_range,
                                    Sum             : sum_likelihood,
                                    Product         : prod_likelihood}
       
        y_vals = Inference.likelihood(spn, data=ranges, dtype=np.float64, node_likelihood=inference_support_ranges)[:,0]
        
        axes[0][i].plot(x_vals, y_vals)
        axes[0][i].set_title("Method " + str(i) + " All")
        axes[0][i].set_ylim([0, max_y])
    
    
    
    evidence = [None, None, None, None, None, NominalRange([0])]
    prob_no_alarm, spn_no_alarm = spn_for_evidence(spn, evidence, node_likelihood=inference_support_ranges, distribution_update_ranges=distribution_update_ranges)
    print(prob_no_alarm)
    
    for i in range(5):
        x_vals = np.linspace(space_start,space_end,num=steps)
        ranges = []
        for x_val in x_vals:
            r = [None] * i + [NumericRange([[x_val]])] + [None] *(5-i)
            ranges.append(r)
        
        
        ranges = np.array(ranges)
    
        inference_support_ranges = {Gaussian        : gaussian_likelihood_range, 
                                    Categorical     : categorical_likelihood_range,
                                    Sum             : sum_likelihood,
                                    Product         : prod_likelihood}
       
        y_vals = Inference.likelihood(spn_no_alarm, data=ranges, dtype=np.float64, node_likelihood=inference_support_ranges)[:,0]
        
        axes[0][i].plot(x_vals, y_vals, label="No Alarm", linestyle=":")

        
        
    
    evidence = [None, None, None, None, None, NominalRange([1])]
    prob_alarm, spn_alarm = spn_for_evidence(spn, evidence, node_likelihood=inference_support_ranges, distribution_update_ranges=distribution_update_ranges)
    print(prob_alarm)
        
    for i in range(5):
        x_vals = np.linspace(space_start,space_end,num=steps)
        ranges = []
        for x_val in x_vals:
            r = [None] * i + [NumericRange([[x_val]])] + [None] *(5-i)
            ranges.append(r)
        
        
        ranges = np.array(ranges)
    
        inference_support_ranges = {Gaussian        : gaussian_likelihood_range, 
                                    Categorical     : categorical_likelihood_range,
                                    Sum             : sum_likelihood,
                                    Product         : prod_likelihood}
       
        y_vals = Inference.likelihood(spn_alarm, data=ranges, dtype=np.float64, node_likelihood=inference_support_ranges)[:,0]
        
        axes[0][i].plot(x_vals, y_vals, label="Alarm")

    plt.legend()
    plt.tight_layout()
    
    plt.savefig("pdp.pdf")
    
    plt.show()    
    
    
    spn_util.plot_spn(spn, "pval.pdf")
    
    tmp = get_nodes_with_weight(spn, 5)
    
    for (weight, node) in tmp:
        print(str(round(node.p[1], 2)) + "\t" + str(weight))
    
    
    
    


def visualize_Density_2d(spn):
    
    
    from spn.experiments.AQP.Ranges import NominalRange, NumericRange
    from spn.algorithms import Inference
    from simple_spn.InferenceRange import categorical_likelihood_range, gaussian_likelihood_range
    from simple_spn.UpdateRange import categorical_update_range
    from spn.experiments.AQP.Ranges import NominalRange, NumericRange
    from spn.structure.Base import Sum, Product
    from spn.algorithms.Inference import sum_likelihood, prod_likelihood
    from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
    
    
    distribution_update_ranges = {Gaussian        : None, 
                                Categorical     : categorical_update_range}
    
    inference_support_ranges = {Gaussian        : gaussian_likelihood_range, 
                                    Categorical     : categorical_likelihood_range,
                                    Sum             : sum_likelihood,
                                    Product         : prod_likelihood}
    
    import matplotlib.pyplot as plt  
    _, axes = plt.subplots(1, 3, figsize=(15,10), squeeze=False, sharey=False, sharex=True)
    x_vals = np.linspace(0,1,num=50)
    y_vals = np.linspace(0,1,num=50)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    ranges = []
    vals = []
    for y_val in y_vals:
        print(y_val)
        ranges = []
        for x_val in x_vals:
            ranges.append([NumericRange([[x_val]]),NumericRange([[y_val]]), None, None, None, None])

        ranges = np.array(ranges)        
        densities = Inference.likelihood(spn, data=ranges, dtype=np.float64, node_likelihood=inference_support_ranges)[:,0]
        
        for i, d in enumerate(densities):
            if d > 5:
                densities[i] = 5
        
        vals.append(densities)
        
    vals = np.array(vals)
    axes[0][0].contour(X,Y,vals)
    axes[0][0].set_xlabel("Method1")
    axes[0][0].set_ylabel("Method2")
    axes[0][0].set_title("Overall")
    
    
    evidence = [None, None, None, None, None, NominalRange([0])]
    prob_no_alarm, spn_no_alarm = spn_for_evidence(spn, evidence, node_likelihood=inference_support_ranges, distribution_update_ranges=distribution_update_ranges)
    print(prob_no_alarm)
    
    ranges = []
    vals = []
    for y_val in y_vals:
        print(y_val)
        ranges = []
        for x_val in x_vals:
            ranges.append([NumericRange([[x_val]]),NumericRange([[y_val]]), None, None, None, None])

        ranges = np.array(ranges)        
        densities = Inference.likelihood(spn_no_alarm, data=ranges, dtype=np.float64, node_likelihood=inference_support_ranges)[:,0]
        
        for i, d in enumerate(densities):
            if d > 5:
                densities[i] = 5
        
        vals.append(densities)
        
    vals = np.array(vals)
    axes[0][1].contour(X,Y,vals)
    axes[0][1].set_xlabel("Method1")
    axes[0][1].set_ylabel("Method2")
    axes[0][1].set_title("Keine Epidemie")
    
    evidence = [None, None, None, None, None, NominalRange([1])]
    prob_alarm, spn_alarm = spn_for_evidence(spn, evidence, node_likelihood=inference_support_ranges, distribution_update_ranges=distribution_update_ranges)
    print(prob_alarm)
    
    ranges = []
    vals = []
    for y_val in y_vals:
        print(y_val)
        ranges = []
        for x_val in x_vals:
            ranges.append([NumericRange([[x_val]]),NumericRange([[y_val]]), None, None, None, None])

        ranges = np.array(ranges)        
        densities = Inference.likelihood(spn_alarm, data=ranges, dtype=np.float64, node_likelihood=inference_support_ranges)[:,0]
        
        for i, d in enumerate(densities):
            if d > 5:
                densities[i] = 5
        
        vals.append(densities)
        
    vals = np.array(vals)
    axes[0][2].contour(X,Y,vals)
    axes[0][2].set_xlabel("Method1")
    axes[0][2].set_ylabel("Method2")
    axes[0][2].set_title("Epidemie")
    
    
    
    plt.savefig("cdp.pdf")
    
    plt.show()   


def visualize_Gaussian(g_nodes):
    
    from scipy import stats
    
    x_vals = np.linspace(10,40,num=1000)
    y_vals = []
    
    for x_val in x_vals:
        mixture_density = 0
        for (weight, node) in g_nodes:
            gaus_mean = node.mean
            gaus_stdev = node.stdev
            density = stats.norm.pdf(x_val, gaus_mean, gaus_stdev)
            mixture_density += weight * density 
        y_vals.append(mixture_density)
    
    
    print(y_vals)
    
    import matplotlib.pyplot as plt
    
    plt.plot(x_vals, y_vals)
    plt.show()



def reduce_spn():
    from spn.experiments.AQP.Ranges import NominalRange, NumericRange
    from spn.structure.Base import Sum, Product
    from spn.algorithms.Inference import sum_likelihood, prod_likelihood
    
    from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
    from spn.structure.leaves.parametric.InferenceRange import categorical_likelihood_range
    from simple_spn.UpdateRange import categorical_update_range
    
    evidence = [NominalRange([0]), None, None, None]
    
    inference_support_ranges = {Gaussian        : None, 
                                Categorical     : categorical_likelihood_range,
                                Sum             : sum_likelihood,
                                Product         : prod_likelihood}
    
    distribution_update_ranges = {Gaussian        : None, 
                                Categorical     : categorical_update_range}
    
    
    #spn_util.plot_spn(spn, "old.pdf")
    
    prob, spn = spn_for_evidence(spn, evidence, node_likelihood=inference_support_ranges, distribution_update_ranges=distribution_update_ranges)
    print(prob)



if __name__ == '__main__':
    
    import os
    from util import io
    np.random.seed(123)
    path = os.path.dirname(os.path.abspath(__file__)) + "/"
    
    if os.path.exists(path + "spn.pkl"):
        spn = io.load_pickle(path + "spn.pkl")
    else:
        data, parametric_types = create_p_value_dataset()
        spn = learn_parametric_spn(data, parametric_types)
        io.dump_pickle(path, "spn.pkl", spn)

    print(spn)
    
    
    
    
    visualize_Density_2d(spn)
    #visualize_Density(spn)
    #g_nodes = get_nodes_with_weight(spn, 2)
    #visualize_Gaussian(g_nodes)
    #spn_util.plot_spn(spn, "new.pdf")
    
    