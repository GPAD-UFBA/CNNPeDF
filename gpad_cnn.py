import h5py
import numpy as np
def dataset_test(filename):
    whole_data_test = []
    file_to_load = 'h5/Data_' + str(filename) + '.hdf5'
    data = data_loader_DATA(file_to_load,filename)
    whole_data_test.append(data)
    X_test = whole_data_test[0]['X_test'][0]
    Y_test = whole_data_test[0]['Y_test'][0]

    return X_test,Y_test

def dataset_train(train):
    for i in train:
        whole_data_test = []
        filename=i
        file_to_load = 'h5/Data_' + str(filename) + '.hdf5'
        data = data_loader_DATA(file_to_load,filename)
        whole_data_test.append(data)
        X_train = whole_data_test[0]['X_train'][0]
        Y_train = whole_data_test[0]['Y_train'][0]
        X_val = whole_data_test[0]['X_val'][0]
        Y_val = whole_data_test[0]['Y_val'][0]
        Classes_sort = whole_data_test[0]['Classes_sort'][0]
        Rcr_classes = whole_data_test[0]['Rcr_classes'][0]
    
        if i==train[0]:
            X_train_0 = X_train
            Y_train_0 = Y_train
            X_val_0 = X_val
            Y_val_0 = Y_val
        elif i==train[1]:
            X_train_1 = X_train
            Y_train_1 = Y_train
            X_val_1 = X_val
            Y_val_1 = Y_val
        elif i==train[2]:
            X_train_2 = X_train
            Y_train_2 = Y_train
            X_val_2 = X_val
            Y_val_2 = Y_val
        if len(train)==4:
            if i==train[3]:
                X_train_3 = X_train
                Y_train_3 = Y_train
                X_val_3 = X_val
                Y_val_3 = Y_val
    
    if len(train)==4:
        X_train=np.concatenate((X_train_0,X_train_1,X_train_2,X_train_3))
        Y_train=np.concatenate((Y_train_0,Y_train_1,Y_train_2,Y_train_3))
        X_val=np.concatenate((X_val_0,X_val_1,X_val_2,X_val_3))
        Y_val=np.concatenate((Y_val_0,Y_val_1,Y_val_2,Y_val_3))
    else:
        X_train=np.concatenate((X_train_0,X_train_1,X_train_2))
        Y_train=np.concatenate((Y_train_0,Y_train_1,Y_train_2))
        X_val=np.concatenate((X_val_0,X_val_1,X_val_2))
        Y_val=np.concatenate((Y_val_0,Y_val_1,Y_val_2))

    return X_train,Y_train,X_val,Y_val

def data_loader_DATA(file_to_load,filename):
    

    # Loads the .hdf5 file
    hf = h5py.File(file_to_load,'r')
    

    # Cria lista para armarezar as variáveis
    if filename=='mtg' or filename=='eball' or filename=='extended_ballroom' or filename=='lmd' or filename=="gtzan":
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        Classes_sort = []
        Rcr_classes = []
    else:
        X_test = []
        Y_test = []

    # Extrai as variáveis 
    if filename=='mtg' or filename=='eball' or filename=='extended_ballroom' or filename=='lmd' or filename=='gtzan':
        X_train.append(np.array(hf.get('Xtrain').get('Xtrain')))
        Y_train.append(np.array(hf.get('Ytrain').get('Ytrain')))
        X_val.append(np.array(hf.get('Xval').get('Xval')))
        Y_val.append(np.array(hf.get('Yval').get('Yval')))
        Classes_sort.append(np.array(hf.get('Classessort').get('Classessort')))
        Rcr_classes.append(np.array(hf.get('Rcr').get('Rcr')))
    else:
        X_test.append(np.array(hf.get('Xtest').get('Xtest')))
        Y_test.append(np.array(hf.get('Ytest').get('Ytest')))




    # Saves in a dictionary
    if filename=='mtg' or filename=='eball' or filename=='extended_ballroom' or filename=='lmd' or filename=='gtzan':
        data = {    
               "X_train": X_train,
               "Y_train": Y_train,
               "X_val": X_val,
               "Y_val": Y_val,
               "Classes_sort": Classes_sort,
               "Rcr_classes": Rcr_classes
        }
    else:
        data = {    
               "X_test": X_test,
               "Y_test": Y_test,

        }

    
    hf.close()
    return data