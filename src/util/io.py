'''
Created on Nov 26, 2018

@author: verwalter
'''

import os
import sys
import gzip
import json
import shutil
import pickle
import logging

import pandas as pd
from tabulate import tabulate


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger



def print_pretty_table(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))



def dump_pickle(path, file_name, data):
    if not os.path.isdir(path):
        os.makedirs(path)
    with gzip.open(path + "/" + file_name, mode="wb") as f:
        pickle.dump(data, f, protocol=-1)
       
       
        
def load_pickle(path):
    with gzip.open(path, mode='rb') as f:
        return pickle.load(f)
    


def dump_json(path, file_name, data):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path + "/" + file_name, mode="w") as f:
        json.dump(data, f)
        


def load_json(path):
    with open(path, mode='r') as f:
        return json.load(f)

 
def remove_directory(path, create=False):
    if os.path.isdir(path):
        shutil.rmtree(path)
    if create:
        os.makedirs(path)


def delete_file(path_to_file):
    os.remove(path_to_file)


def save_df_as_latex(name, df, output_folder, file_name, precision = '%.4f'):    
    #Generates a tex file for the dataframe
    
    #Parameters
    max_lines_per_page = 40
    
    #Set the format of the floats
    pd.set_option('display.float_format', lambda x: precision % x)
    
    #Header
    file_content = "\\documentclass[11pt]{article} \n\n\\usepackage[utf8]{inputenc} \n\\usepackage{booktabs}\n\\usepackage{rotating}\n\n\\begin{document}\n\n"
    file_content += "\\begin{sidewaystable}[!ht]\n\\centering\n\\tiny\n\n"
    
    new_page = True
    
    while new_page:
    
        #Create new table if there is not enough space
        if max_lines_per_page < len(df.index):
            printable = df[:max_lines_per_page].to_latex()
            df = df[max_lines_per_page:]
        else:
            new_page = False
            printable = df.to_latex(index=False)
            
        tmp = name.replace("_", "\_")
        file_content += tmp
        file_content += "\n\n"
        file_content += printable
        file_content += "\n\n"
        
        file_content += "\\vspace{1cm}"
        file_content += "\n\n"
        
        if new_page:
            file_content += "\\end{sidewaystable}"
            file_content += "\\begin{sidewaystable}[!ht]\n\\centering\n\\tiny\n\n"
        
    
    #End of file
    file_content += "\\end{sidewaystable}\n\n"
    file_content += "\\end{document}"
    

    #Save as tex-file
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    file = open(output_folder + file_name + ".tex", "w") 
    file.write(file_content)
    
    
    

def get_path(folder=None):
    return os.path.dirname(os.path.realpath(__file__)) + "/../../" + folder + "/"
    


def exist(name, folder=None, loc="_results"):
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../" + loc + "/"
    if folder is not None:
        save_path += folder + "/"
    save_path += name + ".pkl"
    return os.path.exists(save_path)



def remove(name, folder=None, loc="_results"):
    remove_path = os.path.dirname(os.path.realpath(__file__)) + "/../../" + loc + "/"
    if folder is not None:
        remove_path += folder + "/"
    remove_path += name + ".pkl"
    remove_directory(remove_path)



def save(data, name, folder=None, loc="_results"):
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../../" + loc + "/"
    if folder is not None:
        save_path += folder + "/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    dump_pickle(save_path, name + ".pkl", data)
    
    
    
def load(name, folder=None, loc="_results"):
    results_path = os.path.dirname(os.path.realpath(__file__)) + "/../../" + loc + "/"
    
    if folder is not None:
        results_path += folder + "/"
    
    load_path = results_path + name + ".pkl"
    return load_pickle(load_path)
    
