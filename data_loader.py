import h5py
import numpy as np

def data_loader(file_to_load):

    # Loads the .hdf5 file
    hf = h5py.File(file_to_load,'r')
    
    # Finds the number of ODFs stored in the .hdf5
    numero_de_odf_salvas = len(hf.get('ODF').items())

    # Cria lista para armarezar ODFs e PeDFs
    ODF_SET = []
    PeDF_FULL_SET = []
    PeDF_PARTIAL_SET = []
    #coeffs = []

    # Extrai as ODFs, PeDFs, coeffs
    for i in range(0,numero_de_odf_salvas):
        ODF_SET.append(np.array(hf.get('ODF').get('ODF' + str(i))))
        PeDF_FULL_SET.append(np.array(hf.get('PeDFfull').get('PeDFfull' + str(i))))
        PeDF_PARTIAL_SET.append(np.array(hf.get('PeDFpartial').get('PeDFpartial' + str(i))))
        #coeffs.append(np.array(hf.get('coeffswav').get('coeffswav' + str(i))))

    bpm = np.array(hf.get('bpm').get('bpm'))

    # Saves in a dictionary
    data = {
          "ODF_SET": ODF_SET,
          "PeDF_FULL_SET": PeDF_FULL_SET,
          "PeDF_PARTIAL_SET": PeDF_PARTIAL_SET,
          "bpm": int(bpm)
    }
    
    hf.close()
    return data