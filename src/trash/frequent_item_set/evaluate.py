'''
Created on 26.08.2019

@author: Moritz
'''

import logging
import numpy as np

from util import io
from simple_spn import learn_SPN
from data import real_data
from trash.frequent_item_set import methods



def evaluate_baseline_method(method, item_dataset, min_supports, dataset_name):
    for min_support in min_supports:
        item_sets, exc_time = method(item_dataset, min_support)
        io.save([item_sets, exc_time], method.__name__ + "_minSup=" + str(min_support), "freq_sets/" + dataset_name, loc="_results")
        
    
def evaluate_spn1_method(rdc_thresholds, min_instances_slices, min_supports, dataset_name, binary_positive=True):
    

    for rdc_threshold in rdc_thresholds:
        for min_instances_slice in min_instances_slices:

            loc = "_spns"
            ident = "rdc=" + str(rdc_threshold) + "_mis=" + str(min_instances_slice)
            spn, const_time = io.load(ident, dataset_name, loc)
            
            for min_support in min_supports:
                item_sets, exc_time = methods.spn1(spn, min_support, binary_positive=binary_positive)
                io.save([item_sets, exc_time], "spn1_" + ident + "_minSup=" + str(min_support), "freq_sets/" + dataset_name, loc="_results")
            




if __name__ == '__main__':
    
    
    np.random.seed(123)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    
    
    ''' Data '''
    num_features = 100
    item_dataset, parametric_types = real_data.get_T10I4D(num_features, 100000)
    dataset_name = "T10I4D_100"
    
    
    
    
    
    
    '''
    Create SPNs if necessary
    '''
    rdc_thresholds = [0.2]
    min_instances_slices = [0.1, 0.01]
    
    #learn_SPN.create_parametric_spns(item_dataset, parametric_types, rdc_thresholds, min_instances_slices, folder=dataset_name)
    
    
    
    
    
    '''
    Evaluate
    '''
    min_supports = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    evaluate_baseline_method(methods.apriori1, item_dataset, min_supports, dataset_name)
    #evaluate_baseline_method(methods.fpgrowth1, item_dataset, min_supports, dataset_name)
    #evaluate_baseline_method(methods.fpgrowth2, item_dataset, min_supports, dataset_name)
    
    evaluate_spn1_method(rdc_thresholds, min_instances_slices, min_supports, dataset_name)
    
    
    
    