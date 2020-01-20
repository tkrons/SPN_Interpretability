'''
Created on 04.10.2019

@author: Moritz
'''


import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from simple_spn import learn_SPN
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from interpretability import subpopulations

def get_titanic():
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_data/titanic/train.csv"

    df = pd.read_csv(path)
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    #df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df["Age"].fillna(int(df["Age"].mean()), inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    #df['Embarked'] = df['Embarked'].map({'S': 0, 'Q': 1, 'C':2})
    #df['Pclass'] = df['Pclass'] - 1
    
    

    #Survived |   Pclass |   Sex |   Age |   SibSp |   Parch |     Fare |   Embarked 
    return df



def get_datatset(feature_names=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]):
    
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../../_data/titanic/train.csv"
    df = pd.read_csv(path)
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    df["Age"].fillna(int(df["Age"].mean()), inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'Q': 1, 'C':2})
    df['Pclass'] = df['Pclass'] - 1
    
    df2 = df[feature_names]
    for col_name in list(df2.columns):
        if col_name in ["Sex", "Pclass", "SibSp", "Parch", "Embarked"]:
            df2 = pd.get_dummies(df2, columns=[col_name])
    
    
    y = list(df["Survived"])
    df3 = df2.drop(columns=["Survived"])
    X = df3.values
    
    parametric_types = []
    value_dict = {}
    for i, col_name in enumerate(list(df2.columns)):
        
        value_entry = []
        if col_name in ["Age", "Fare"]:
            parametric_types.append(Gaussian)
            value_entry.append("numeric")
            value_entry.append(col_name)
            value_entry.append([np.min(df[col_name]), np.max(df[col_name])])
        else:
            parametric_types.append(Categorical)
            value_entry.append("discrete")
            value_entry.append(col_name)
            val_dict = {}
            for val in df2[col_name].unique():
                val_dict[val] = str(val)
            value_entry.append(val_dict)
            
        
        value_dict[i] = value_entry
    
    
    return X, y, df2, parametric_types, value_dict, "Survived"


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def evaluate_fi_logistic_regression(X, y):
    clf = LogisticRegression(random_state=0, solver='lbfgs')
    clf = clf.fit(X, y)
    fi = clf.coef_[0]
    fi = np.abs(fi)
    fi = fi/np.max(fi)
    return fi


def evaluate_fi_random_forest(X, y):
    clf = RandomForestClassifier(random_state=0, n_estimators=100)
    clf = clf.fit(X, y)
    fi = clf.feature_importances_ 
    fi = fi/np.max(fi)
    return fi

def evaluate_fi_spn(df, parametric_types, value_dict, target_name):
    
    data = df.values
    spn, const_time = learn_SPN.learn_parametric_spn(data, parametric_types, 0.3, 50)

    target_id = -1
    for i, col_name in enumerate(df.columns):
        if col_name == target_name:
            target_id = i
            break
    fi = subpopulations.feature_importance(spn, target_id, value_dict)
    fi = fi/np.max(fi)
    return fi
    




if __name__ == '__main__':
    
    print("Heleasdas")
    
    X, y, df, parametric_types, value_dict, target_name = get_datatset(["Survived", "Pclass", "Sex"])
    
    a = evaluate_fi_logistic_regression(X, y)
    b = evaluate_fi_random_forest(X, y)
    c = evaluate_fi_spn(df, parametric_types, value_dict, target_name)
    
    print(df)
    print(a)
    print(b)
    print(c)
    
    
    