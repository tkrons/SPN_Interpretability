'''
Created on 23.10.2019

@author: Moritz
'''

from data import ed_data
from interpretability import visualizations as vz
from simple_spn import spn_handler
from simple_spn import functions as fn
from util import io

def explore_1():
    
    
    dataset_name = "rki_ed_1"
    rdc_threshold = 0.3
    min_instances_slice = 0.01
    if not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
        df, value_dict, parametric_types = ed_data.get_rki_ed_1()
        spn_handler.create_parametric_spns(df.values, parametric_types, dataset_name, [rdc_threshold], [min_instances_slice], value_dict)
    spn, value_dict, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
    
    spn = fn.marg(spn, keep=[0,2,3,4,5])
    
    
    
    
    fn.print_statistics(spn)
    
    p = io.get_path("_results/ed_data_explore")
    
    #vz.visualize_overall_distribution(spn, value_dict)
    
    from spn.experiments.AQP.Ranges import NominalRange
    
    target_conds = [{0 : NominalRange([5,6])}, {0 : NominalRange([0,1,2,3,4])}]
    #target_conds = [{0 : NominalRange([5,6]), 1 : NominalRange([0,1,2,3,4,5,6,7,8,9,10,11])}, {0 : NominalRange([0,1,2,3,4]), 1 : NominalRange([0,1,2,3,4,5,6,7,8,9,10,11])}]
    vz.visualize_target_based_conds_overall_distribution_compact(spn, target_conds, value_dict, target_names=["Wochenende", "Unter der Woche"], save_path=p+dataset_name+"_weekend_measures.pdf")
    
    #vz.visualize_target_based_overall_distribution_compact(spn, 1, value_dict)



def explore_2():
    
    
    dataset_name = "rki_ed_2"
    rdc_threshold = 0.3
    min_instances_slice = 0.01
    if not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
        df, value_dict, parametric_types = ed_data.get_rki_ed_2()
        spn_handler.create_parametric_spns(df.values, parametric_types, dataset_name, [rdc_threshold], [min_instances_slice], value_dict)
    spn, value_dict, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
    
    fn.print_statistics(spn)
    
    
    vz.visualize_overall_distribution(spn, value_dict)
    



    
def explore_3():
    
    dataset_name = "rki_ed_3"
    rdc_threshold = 0.3
    min_instances_slice = 0.01
    if not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
        df, value_dict, parametric_types = ed_data.get_rki_ed_3()
        spn_handler.create_parametric_spns(df.values, parametric_types, dataset_name, [rdc_threshold], [min_instances_slice], value_dict)
    spn, value_dict, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
    
    fn.print_statistics(spn)
    print(value_dict)
    
    p = io.get_path("_results/ed_data_explore")
    
    vz.visualize_likeliness_heatmap(spn, target_id_x=0, target_id_y=1, value_dict=value_dict, save_path=p+dataset_name+"_hour_dep.pdf")
    vz.visualize_likeliness_heatmap(spn, target_id_x=0, target_id_y=5, value_dict=value_dict, save_path=p+dataset_name+"_hour_day.pdf")  
    vz.visualize_likeliness_heatmap(spn, target_id_x=0, target_id_y=6, value_dict=value_dict, save_path=p+dataset_name+"_hour_month.pdf")  



if __name__ == '__main__':
    
    #explore_1()
    #explore_2()
    explore_3()
    
    
    
    