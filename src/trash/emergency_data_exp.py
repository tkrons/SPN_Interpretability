'''
Created on 21.10.2019

@author: Moritz
'''


import os
from simple_spn import functions as fn
import pandas as pd
from simple_spn import learn_SPN

from util import io


def get_eseg_export():
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/eseg_export/example2019-09.json"
    
    df = pd.read_json(path)
    
    cols_circumstances = ["sys_date", "sys_hour", "sys_department", "sys_transport", 'sys_disposition', 'sys_ed', 'sys_lab', 'sys_labinfections']
    cols_patient = ["sys_gender", "sys_age16","sys_plz3"]
    cols_pain = ["sys_complaint", 'sys_diagnosis_icd4', "sys_triage", "sys_isolation", 'sys_ecg', 'sys_echo', 'sys_xraythorax']
    cols_measures = ["sys_heartrate", "sys_temperature", "sys_respiratoryrate",'sys_bloodpressuresystolic']
    
    io.print_pretty_table(df[cols_circumstances].head(10))
    io.print_pretty_table(df[cols_patient].head(10))
    io.print_pretty_table(df[cols_pain].head(10))
    io.print_pretty_table(df[cols_measures].head(10))
    
    
    
def get_rki_export():
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/eseg_epias_rki/epias_of_rki.2018-11.350000.json"
    
    df = pd.read_json(path)
        
    cols_circumstances = ["aufnahmezeitpunkt_datum", "aufnahmezeitpunkt_stunde", "behandlung_fachabteilung", "id_einrichtung", "zuweisungsart"]
    cols_patient = ["geschlecht", "altersklasse","plzbereich"]
    cols_pain = ["schmerz", 'diagnosen', "leitsymptom", "leitsymptom_gruppe", 'tetanus', 'triagesystem']
    cols_measures = ["untersuchung_bga", "untersuchung_echokardiographie", "untersuchung_ekg", 'untersuchung_roentgen_thorax', "vitalwerte"]
    
    io.print_pretty_table(df[cols_circumstances].head(10))
    io.print_pretty_table(df[cols_patient].head(10))
    io.print_pretty_table(df[cols_pain].head(10))
    io.print_pretty_table(df[cols_measures].head(10))
    



def test_rki_minimal():
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/eseg_epias_rki/"
    #df = pd.concat([pd.read_json(f) for f in [path + f for f in os.listdir(path)]], ignore_index = True)
    
    
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/eseg_epias_rki/epias_of_rki.2018-11.300000.json"
    df = pd.read_json(path)
    
    cols_measures = ["vitalwerte"]
    io.print_pretty_table(df[cols_measures].head(10))
    
    
    col_names = ["aufnahmezeitpunkt_stunde", "behandlung_fachabteilung", "zuweisungsart", "geschlecht", "altersklasse", "leitsymptom_gruppe"]
    my_df = df[col_names]
  
    io.print_pretty_table(my_df.head(100))
    
    for col_name in col_names:
        print(col_name + " : " + str(len(my_df[col_name].unique())))


def do_rki_minimal():
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/eseg_epias_rki/"
    df = pd.concat([pd.read_json(f) for f in [path + f for f in os.listdir(path)]], ignore_index = True, sort=False)

    col_names = ["aufnahmezeitpunkt_stunde", "behandlung_fachabteilung"]#, "zuweisungsart", "geschlecht"]
    df = df[col_names]
    
    
    
    trans_df, value_dict, param_types = fn.transform_dataset(df)
    
    io.print_pretty_table(trans_df.head(10))
    print(value_dict)
    print(param_types)
    
    
    
    spn, const_time = learn_SPN.learn_parametric_spn(trans_df.values, param_types, rdc_threshold=0.3, min_instances_slice=0.05*len(df))
    print(const_time)
    fn.print_statistics(spn)
    

    
    

if __name__ == '__main__':
    do_rki_minimal()
    
    
