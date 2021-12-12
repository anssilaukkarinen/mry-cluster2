# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:51:24 2021

@author: laukkara

This script is run first to fetch results data from university's network drive

"""
import os
import pickle


input_folder_for_Delphin_data = r'S:\91202_Rakfys_Mallinnus\RAMI\simulations'

output_folder = os.path.join(r'C:\Local\laukkara\Data\github\mry-cluster2\input')
    
output_pickle_file_name = 'S_RAMI.pickle'



## Preparations
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_pickle_file_path = os.path.join(output_folder,
                                       output_pickle_file_name)

## Read in results data from pickle files
cases = {}
data = {}

cases = os.listdir(input_folder_for_Delphin_data)
cases.remove('olds')
cases.remove('RAMI_simulated_cases.xlsx')

data = {}
for case in cases:
    print('Reading:', case)
    fname = os.path.join(input_folder_for_Delphin_data, case, 'd.pickle')
    
    with open(fname, 'rb') as f:
        try:
            df = pickle.load(f)
            
            if df.shape[0] == 1200:
                data[case] = df
            else:
                print('ERROR AT:', case)
                
        except:
            print('Error when reading case:', case)

print(data[cases[0]].columns)



with open(output_pickle_file_path, 'wb') as f:
    pickle.dump(data, f)







