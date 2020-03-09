'''
Created on 27.06.2019

@author: Moritz
'''

import time
import numpy as np

from spn.structure.StatisticalTypes import MetaType
from spn.structure.Base import Context
from spn.algorithms.LearningWrappers import learn_parametric
from simple_spn import functions as fn
from util import io
import warnings






def learn_parametric_spn(data, parametric_types, rdc_threshold=0.3, min_instances_slice=0.05, clustering='kmeans'):
    
    ds_context = Context(parametric_types=parametric_types).add_domains(data)
    ds_context.add_domains(data)
    mis = int(len(data) * min_instances_slice)
    
    t0 = time.time()
    spn = learn_parametric(data, ds_context, threshold=rdc_threshold, min_instances_slice=mis, rows=clustering)
    const_time = time.time() - t0
    
    return spn, const_time


def create_parametric_spns(data, parametric_types, dataset_name, rdc_thresholds=[0.3], min_instances_slices=[0.05], value_dict=None, save=True,
                           clustering = 'kmeans', silence_warnings=False, nrows=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rdc_threshold in rdc_thresholds:
            for min_instances_slice in min_instances_slices:
                spn, const_time = learn_parametric_spn(data, parametric_types, rdc_threshold, min_instances_slice, clustering=clustering)
                if save:
                    save_spn(spn, const_time, dataset_name, rdc_threshold, min_instances_slice, value_dict, nrows,)
                else:
                    return spn, value_dict, parametric_types

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''



def exist_spn(dataset_name, rdc_threshold, min_instances_slice):
    return io.exist("rdc=" + str(rdc_threshold) + "_mis=" + str(min_instances_slice), dataset_name, "_spns")



def save_spn(spn, const_time, dataset_name, rdc_threshold, min_instances_slice, value_dict=None, nrows=None,):
    if value_dict is None: fn.generate_adhoc_value_dict(spn)
    name = "rdc=" + str(rdc_threshold) + "_mis=" + str(min_instances_slice)
    if nrows:
        name = name + "_n=" + np.format_float_scientific(nrows, precision=0, trim='-')
    io.save([spn, value_dict, const_time], name, dataset_name, loc="_spns")



def load_spn(dataset_name, rdc_threshold, min_instances_slice, nrows = None):
    fname = "rdc=" + str(rdc_threshold) + "_mis=" + str(min_instances_slice)
    if nrows:
        fname = fname + '_n=' + np.format_float_scientific(nrows, precision=0, trim='-')
    return io.load(fname, dataset_name, "_spns")



'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''



def generate_spn_parameters(rdc_threshold=0.3, cols="rdc", rows="rdc", min_instances_slice=50, ohe=False, prior_weight=0.00, identity_numeric=False):
    
    return {"rdc_threshold" : rdc_threshold,
            "cols" : cols,
            "rows" : rows,
            "min_instances_slice" : min_instances_slice,
            "ohe" : ohe,
            "prior_weight" : prior_weight,
            "identity_numeric" : identity_numeric}


def learn_AQP_spn(numpy_data, feature_types, spn_params, rand_gen):
    
    from spn.algorithms.StructureLearning import get_next_operation, learn_structure
    from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py

    from spn.structure.leaves.parametric.Parametric import Categorical
    from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
    from spn.experiments.AQP.leaves.identity.IdentityNumeric import create_identity_leaf


    #cast may not be necessary
    numpy_data = np.array(numpy_data, np.float64)
    
    #Generate meta_type array
    meta_types = []
    for feature_type in feature_types:
        if feature_type == "discrete":
            meta_types.append(MetaType.DISCRETE)
        elif feature_type == "continuous":
            meta_types.append(MetaType.REAL)
        else:
            raise Exception("Unknown feature type for SPN: " + feature_type)
    
    #Create information about the domains
    domains = []
    for col in range(numpy_data.shape[1]):
        feature_type = feature_types[col]
        if feature_type == 'continuous':
            domains.append([np.min(numpy_data[:, col]), np.max(numpy_data[:, col])])
        elif feature_type in {'discrete', 'categorical'}:
            domains.append(np.unique(numpy_data[:, col]))
    
    #Create context
    ds_context = Context(meta_types=meta_types, domains=domains)
        
    #Fixed parameters
    rdc_threshold = spn_params["rdc_threshold"]
    cols = spn_params["cols"]
    rows = spn_params["rows"]
    min_instances_slice = spn_params["min_instances_slice"]
    ohe = spn_params["ohe"]
    prior_weight = spn_params["prior_weight"]
    identity_numeric = spn_params["identity_numeric"]
    
    #Method to create leaves in the SPN
    def create_leaf(data, ds_context, scope):
        idx = scope[0]
        meta_type = ds_context.meta_types[idx]
        
        if meta_type == MetaType.REAL:
            if identity_numeric:
                return create_identity_leaf(data, scope)
        
            if prior_weight == 0.:
                return create_piecewise_leaf(data, ds_context, scope, prior_weight=None)
            else:
                return create_piecewise_leaf(data, ds_context, scope, prior_weight=prior_weight)
            

        elif meta_type == MetaType.DISCRETE:
            
            unique, counts = np.unique(data[:,0], return_counts=True)
            
            sorted_counts = np.zeros(len(ds_context.domains[idx]), dtype=np.float64)
            for i, x in enumerate(unique):
                sorted_counts[int(x)] = counts[i] 
            
            p = sorted_counts / data.shape[0]
            
            #Do regularization
            if prior_weight > 0.:
                p += prior_weight
            p = p/np.sum(p)
            
            return Categorical(p, scope)

        else:
            raise Exception("Mehtod learn_mspn_for_aqp(...) cannot create leaf for " + str(meta_type))
    
    #Set method to create leaves
    leaves = create_leaf
    
    #Set methods to cluster and to do the independence test
    if cols == "rdc":
        split_cols = get_split_cols_RDC_py(rdc_threshold, ohe=ohe, k=10, s=1 / 6,
                                           non_linearity=np.sin, n_jobs=1,
                                           rand_gen=rand_gen)
        
    if rows == "rdc":
        split_rows = get_split_rows_RDC_py(n_clusters=2, ohe=ohe, k=10, s=1 / 6,
                                           non_linearity=np.sin, n_jobs=1,
                                           rand_gen=rand_gen)
        
    #This choses which operation is performed
    nextop = get_next_operation(min_instances_slice)
    
    #Learn the SPN
    root_node = learn_structure(numpy_data, ds_context, split_rows, split_cols, leaves, nextop)
    
    return root_node

