# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 01:13:37 2018

@author: Rituraj
"""
import pandas as pd
import preprocessingfile as preprocess
#original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val = preprocess.my_sdp_preprocessor('pc2.csv')
#all_data = original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val 
def NN(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):   
    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = len(original_X.columns)))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting the ANN to the Training set
    classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)
    
    #Making the predictions and evaluating the model
    # Predicting the Test set results
    y_pred = classifier.predict(x_val)
    y_pred = (y_pred > 0.5)
    y_pred = pd.DataFrame(y_pred, columns=['defects'])
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, y_pred)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_val, y_pred)
    
    return classifier

    
def random_forest(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
    clf.fit(x_train, y_train)
    return clf

def svm(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):
    from sklearn.svm import SVC
    clf = SVC(gamma='auto')
    clf.fit(x_train, y_train)
    return clf
    

        


def cnn(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):
    from keras.models import Sequential
    from keras.layers import Dense,Dropout,Conv2D,Conv1D,Flatten,MaxPool2D
    #create model
    
    x_train_matrix = x_train.values
    x_val_matrix = x_val.values
    y_train_matrix = y_train.values
    y_val_matrix = y_val.values
    
    ytrainseries = y_train['defects']
    #y_train_onehot = pd.get_dummies(ytrainseries)
    yvalseries = y_val['defects']
    #y_val_onehot = pd.get_dummies(yvalseries)
    
    img_rows, img_cols = 1,len(original_X.columns)
    
    x_train1 = x_train_matrix.reshape(x_train_matrix.shape[0], img_rows, img_cols, 1)
    x_val1 = x_val_matrix.reshape(x_val_matrix.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    #add model layers
    model.add(Conv2D(64, kernel_size=1, activation='relu',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=1, activation='relu'))
    model.add(Conv2D(16, kernel_size=1, activation='relu'))
    
    
#   model.add(MaxPool2D(pool_size=(1,8)))
    #model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #train the model
    model.fit(x_train1, y_train_matrix, epochs=40)    
    y_pred = model.predict(x_val1)>0.5
    y_pred_df = pd.DataFrame(y_pred)
    
    return model         


    
#NN_clf = NN()
#rf_clf = random_forest()
#svm_clf = svm()
#cnn_clf = cnn()



