'''
Created on 22.10.2019

@author: Moritz
'''


def get_p_value_dataset():
    
    from data import R_wrapper_data
    from evaluation import evaluator
    from methods import R_wrapper
    from ml import dataset_creator, pre_processing

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

    return ds.df.values, [Gaussian,Gaussian,Gaussian,Gaussian,Gaussian,Categorical]


def get_p_value_test_value_dict():
    return {0 : ["numeric",  "EARS C1",    [0.0, 1.0]],
            1 : ["numeric",  "EARS C2",    [0.0, 1.0]],
            2 : ["numeric",  "EARS C3",    [0.0, 1.0]],
            3 : ["numeric",  "Bayes",      [0.0, 1.0]],
            4 : ["numeric",  "RKI",        [0.0, 1.0]],
            5 : ["discrete", "alarm",      {0: "True", 1: "False"}]
            }


if __name__ == '__main__':
    pass