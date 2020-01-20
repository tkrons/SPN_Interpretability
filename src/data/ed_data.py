'''
Created on 22.10.2019

@author: Moritz
'''

import os
import datetime 
import pandas as pd
import numpy as np
from util import io
from simple_spn import functions as fn


def get_rki_ed_data(column_names=None, file_names=None):
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/ed/rki_pre_process/"
    if file_names is None : file_names = os.listdir(path)
    df = pd.concat([pd.read_json(f) for f in [path + f for f in file_names]], ignore_index=True, sort=False)
    if column_names is not None: df = df[column_names]
    return df
    

def get_rki_ed_1():
    df = get_rki_ed_data(column_names=["aufnahmezeitpunkt_datum", "leitsymptom", "leitsymptom_gruppe", "vitalwerte", "diagnosen"])#, file_names=["epias_of_rki.2018-11.300000.json"])
    df["aufnahmezeitpunkt_wochentag"] = df["aufnahmezeitpunkt_datum"].apply(lambda date_str: datetime.datetime.strptime(date_str, '%Y-%m-%d').weekday())
    df["aufnahmezeitpunkt_monat"] = df["aufnahmezeitpunkt_datum"].apply(lambda date_str: datetime.datetime.strptime(date_str, '%Y-%m-%d').month)
    
    
    data = []  
    for val in df["aufnahmezeitpunkt_datum"].unique():
        day_df = df[df["aufnahmezeitpunkt_datum"] == val]
        if len(day_df) == 0: continue
        
        week_day = day_df["aufnahmezeitpunkt_wochentag"].iloc[0]
        month = day_df["aufnahmezeitpunkt_monat"].iloc[0]
        
        temps = []
        heart_rates = []
        breath_rates = []
        blood_pressures = []
        for vitals in day_df["vitalwerte"]:
            for vital in vitals:
                if 'blutdruck_systolisch' in vital: blood_pressures.append(vital['blutdruck_systolisch'])
                if 'atemfrequenz' in vital: breath_rates.append(vital['atemfrequenz'])
                if 'herzfrequenz' in vital: heart_rates.append(vital['herzfrequenz'])
                if 'temperatur' in vital: temps.append(vital['temperatur'])
            
        temps = np.array(temps, dtype=np.float64)
        heart_rates = np.array(heart_rates, dtype=np.float64)
        breath_rates = np.array(breath_rates, dtype=np.float64)
        blood_pressures = np.array(blood_pressures, dtype=np.float64)
        
        data.append([week_day, month, len(temps), len(heart_rates), len(breath_rates), len(blood_pressures)])#, np.mean(temps), np.mean(heart_rates), np.mean(breath_rates), np.mean(blood_pressures)])
    
    df = pd.DataFrame(data, columns=["wochentag",
                                "monat",
                                "count_temperatur",
                                "count_herzfrequenz",
                                "count_atemfrequenz",
                                "count_blutdruck",
                                #"avg_temperatur", "avg_herzfrequenz", "avg_atemfrequenz", "avg_blutdruck",
                                ])
    
    return fn.transform_dataset(df, feature_types=["discrete", "discrete", "numeric", "numeric", "numeric", "numeric"])



def get_rki_ed_2():
    df = get_rki_ed_data(column_names=["aufnahmezeitpunkt_datum", "leitsymptom", "leitsymptom_gruppe", "vitalwerte", "diagnosen"])#, file_names=["epias_of_rki.2018-11.300000.json"])
    df["aufnahmezeitpunkt_wochentag"] = df["aufnahmezeitpunkt_datum"].apply(lambda date_str: datetime.datetime.strptime(date_str, '%Y-%m-%d').weekday())
    df["aufnahmezeitpunkt_monat"] = df["aufnahmezeitpunkt_datum"].apply(lambda date_str: datetime.datetime.strptime(date_str, '%Y-%m-%d').month)
    
    
    data = []  
    for val in df["aufnahmezeitpunkt_datum"].unique():
        day_df = df[df["aufnahmezeitpunkt_datum"] == val]
        if len(day_df) == 0: continue
        
        week_day = day_df["aufnahmezeitpunkt_wochentag"].iloc[0]
        month = day_df["aufnahmezeitpunkt_monat"].iloc[0]
        
        temps = []
        heart_rates = []
        breath_rates = []
        blood_pressures = []
        for vitals in day_df["vitalwerte"]:
            for vital in vitals:
                if 'blutdruck_systolisch' in vital: blood_pressures.append(vital['blutdruck_systolisch'])
                if 'atemfrequenz' in vital: breath_rates.append(vital['atemfrequenz'])
                if 'herzfrequenz' in vital: heart_rates.append(vital['herzfrequenz'])
                if 'temperatur' in vital: temps.append(vital['temperatur'])
            
        temps = np.array(temps, dtype=np.float64)
        heart_rates = np.array(heart_rates, dtype=np.float64)
        breath_rates = np.array(breath_rates, dtype=np.float64)
        blood_pressures = np.array(blood_pressures, dtype=np.float64)
        
        if len(temps)==0 or len(heart_rates)==0 or len(breath_rates)==0 or len(blood_pressures) == 0:
            continue
        
        
        data.append([week_day, month, len(temps), len(heart_rates), len(breath_rates), len(blood_pressures), np.mean(temps), np.mean(heart_rates), np.mean(breath_rates), np.mean(blood_pressures)])
    
    df = pd.DataFrame(data, columns=["wochentag",
                                "monat",
                                "count_temperatur",
                                "count_herzfrequenz",
                                "count_atemfrequenz",
                                "count_blutdruck",
                                "avg_temperatur", "avg_herzfrequenz", "avg_atemfrequenz", "avg_blutdruck",
                                ])
    
    return fn.transform_dataset(df, feature_types=["discrete", "discrete", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric"])

#def get_rki_ed_2():
#    df = get_rki_ed_data(column_names=["aufnahmezeitpunkt_stunde", "behandlung_fachabteilung"])
#    return fn.transform_dataset(df)

def get_rki_ed_3():
    df = get_rki_ed_data(column_names=["aufnahmezeitpunkt_datum", "aufnahmezeitpunkt_stunde", "behandlung_fachabteilung", "geschlecht", "altersklasse", "zuweisungsart"], file_names=["epias_of_rki.2018-11.300000.json"])
    
    def weekday(date_str):
        week_day = datetime.datetime.strptime(date_str, '%Y-%m-%d').weekday()
        if week_day == 0: return "Mon"
        if week_day == 1: return "Tue"
        if week_day == 2: return "Wed"
        if week_day == 3: return "Thur"
        if week_day == 4: return "Fri"
        if week_day == 5: return "Sat"
        if week_day == 6: return "Sun"
    
    df["aufnahmezeitpunkt_wochentag"] = df["aufnahmezeitpunkt_datum"].apply(weekday)
    df["aufnahmezeitpunkt_monat"] = df["aufnahmezeitpunkt_datum"].apply(lambda date_str: datetime.datetime.strptime(date_str, '%Y-%m-%d').month)
    df.drop(columns=["aufnahmezeitpunkt_datum"], inplace=True)
    return fn.transform_dataset(df)




'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''



def get_eseg_export():
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/ed/eseg_export_sample/example2019-09.json"
    
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
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../_data/ed/rki_pre_process/epias_of_rki.2018-11.350000.json"
    
    df = pd.read_json(path)
        
    cols_circumstances = ["aufnahmezeitpunkt_datum", "aufnahmezeitpunkt_stunde", "behandlung_fachabteilung", "id_einrichtung", "zuweisungsart"]
    cols_patient = ["geschlecht", "altersklasse","plzbereich"]
    cols_pain = ["schmerz", 'diagnosen', "leitsymptom", "leitsymptom_gruppe", 'tetanus', 'triagesystem']
    cols_measures = ["untersuchung_bga", "untersuchung_echokardiographie", "untersuchung_ekg", 'untersuchung_roentgen_thorax', "vitalwerte"]
    
    io.print_pretty_table(df[cols_circumstances].head(10))
    io.print_pretty_table(df[cols_patient].head(10))
    io.print_pretty_table(df[cols_pain].head(10))
    io.print_pretty_table(df[cols_measures].head(10))





if __name__ == '__main__':
    get_eseg_export()
    get_rki_export()




