# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:08:40 2018

@author: Rituraj
"""

import pandas as pd
import performance_check as ensembler
import preprocessingfile as preprocess
from sklearn.metrics import *
data = 'pc2.csv'
original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val = preprocess.my_sdp_preprocessor(data)
all_data = [original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val]

nn_clf, cnn_clf, svm_clf, rf_clf = ensembler.send_classifiers_to_LR_file()
log_reg_clf, new_test_set_x_matrix = ensembler.send_results_to_logistic_regression()

prediction = log_reg_clf.predict(new_test_set_x_matrix)
print('Accuracy:',accuracy_score(y_test.values,prediction))



#change 'data' variable in this file as well as performance_check.py
