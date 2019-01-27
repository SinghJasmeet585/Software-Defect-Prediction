# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 00:19:45 2018

@author: Rituraj
"""

#Software Defect Prediction 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

#add dataset normalization and feature selection function

def my_sdp_preprocessor(datafilename_as_csv_inquotes):
    original_data = pd.read_csv(datafilename_as_csv_inquotes)
    original_data.isnull().values.any() #Gives false ie:No null value in dataset
    original_data = original_data.fillna(value=False)
    original_X = pd.DataFrame(original_data.drop(['defects'],axis=1))
    original_Y = original_data['defects']
    original_Y = pd.DataFrame(original_Y)
    x_train1, x_test, y_train1, y_test= train_test_split(original_X, original_Y, test_size = .1,
                                                              random_state=12)
    
    #now we resample, and from that we take training and validation sets
    
    sm = SMOTE(random_state=12, ratio = 1.0)
    x, y = sm.fit_sample(x_train1, y_train1)
    y_train2 = pd.DataFrame(y, columns=['defects'])
    x_train2 = pd.DataFrame(x, columns=original_X.columns)
    
    
    x_train, x_val, y_train, y_val= train_test_split(x_train2, y_train2, test_size = .1,
                                                              random_state=12)
    
    combined_training_data = x_train.copy()
    combined_training_data['defects'] = y_train
    
    
    
    import seaborn as sns
    corr = combined_training_data.corr()
    sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
    
    
    
    return original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val 

