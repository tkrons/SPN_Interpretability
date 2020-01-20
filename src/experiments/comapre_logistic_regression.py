'''
Created on 02.10.2019

@author: Moritz
'''


import os
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression



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



if __name__ == '__main__':
    
    
    df = get_titanic()
    y = list(df["Survived"])
    
    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass", "SibSp", "Parch"])
    df = df.drop(columns=["Survived"])
    X = df.values
    
    clf = LogisticRegression(random_state=0, solver='lbfgs')
    clf = clf.fit(X, y)
    
    
    for i in range(len(df.columns)):
        print(str(list(df.columns)[i]) + "\t\t " + str(round(list(clf.coef_[0])[i], 4)))
        
    
    
    exit()
    
    
    X, y = load_iris(return_X_y=True)
    
    print(X)
    print(y)
    
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf = clf.fit(X, y)
    preds = clf.predict(X[:2, :])
    
    print(preds)
    
    print(clf.coef_)
    
    
    
    
    
    
    
    