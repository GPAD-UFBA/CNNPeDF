import h5py
import numpy as np

def data_loader_DATA(file_to_load):
    

    # Loads the .hdf5 file
    hf = h5py.File(file_to_load,'r')
    

    # Cria lista para armarezar as variáveis
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    X_test = []
    Y_test = []

    # Extrai as variáveis 
    
    X_train.append(np.array(hf.get('Xtrain').get('Xtrain')))
    Y_train.append(np.array(hf.get('Ytrain').get('Ytrain')))
    X_val.append(np.array(hf.get('Xval').get('Xval')))
    Y_val.append(np.array(hf.get('Yval').get('Yval')))
    X_test.append(np.array(hf.get('Xtest').get('Xtest')))
    Y_test.append(np.array(hf.get('Ytest').get('Ytest')))

    # Saves in a dictionary
    data = {    
           "X_train": X_train,
           "Y_train": Y_train,
           "X_val": X_val,
           "Y_val": Y_val,
           "X_test": X_test,
           "Y_test": Y_test
    }
    
    
    hf.close()
    return data