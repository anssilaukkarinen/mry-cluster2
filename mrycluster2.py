# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 18:57:42 2021

@author: laukkara
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import mry_helper


from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform




root_folder = r'C:\Local\laukkara\Data\github\mry-cluster2'

input_folder = os.path.join(root_folder, 'input')
output_folder = os.path.join(root_folder, 'output')


# Load input data

fname = os.path.join(input_folder,
                     'S_RAMI.pickle')
with open(fname, 'rb') as f:
    data = pickle.load(f)

# remove wood 626 from data
data_new = {}

for key in data:
    if '626' in key:
        pass
    
    elif 'UST_' in key:
        # remove interior data points from UST
        
        cols_to_include = [x for x in data[key].columns if 'stud_i' not in x and 'ins_i' not in x]
        
        data_new[key] = data[key].loc[:, cols_to_include]
        
    else:
        data_new[key] = data[key]

data = data_new





################

starts_all = [x.split('_')[0] for x in data.keys()]
starts_unique = set(starts_all)
print('Unique main case types:')
print(starts_unique)

# Add case+mp counts to legends

#starts_unique = ['UST']

# 'M_', 'RH_'
# 'moverhygr_', 'mmass_', 'T_' & 'max' OR 'min'


# M
# y_yes_filters = ['M_']
# y_not_filters = ['rank']

# moverhygr
y_yes_filters = ['moverhygr_']
y_not_filters = ['rank']

# RH
# y_yes_filters = ['RH_']
# y_not_filters = ['rank', 'TRH']



ml_args_agglomerative = {'name': 'agglomerative',
                         'n_clusters': 2,
                        'affinity': 'euclidean',
                        'linkage': 'ward'}

# ml_args = ml_args_agglomerative
# affinity: {“euclidean”, “l1”, “l2”, “manhattan”, “cosine”, “precomputed”}, default euclidean
# linkage: {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
ml_args_a_e_w = {'name': 'agglomerative',
            'n_clusters': 2,
            'affinity': 'euclidean',
            'linkage': 'ward'}

ml_args_a_e_a = {'name': 'agglomerative',
            'n_clusters': 2,
            'affinity': 'euclidean',
            'linkage': 'average'}

ml_args_a_m_a = {'name': 'agglomerative',
            'n_clusters': 2,
            'affinity': 'manhattan',
            'linkage': 'average'}

ml_args_k = {'name': 'kmeans',
          'n_clusters': 2,
          'n_init': 100,
          'max_iter': 1000,
          'tol': 1e-5}

ml_args_d_10 = {'name': 'dbscan',
            'eps': 10,
            'min_samples': 1,
            'leaf_size': 30}

ml_args_d_30 = {'name': 'dbscan',
            'eps': 30,
            'min_samples': 1,
            'leaf_size': 30}

ml_args_d_50 = {'name': 'dbscan',
            'eps': 50,
            'min_samples': 1,
            'leaf_size': 30}

# spectral clustering gives "Graph is not fully connected" warnings
#ml_args = {'name': 'spectralclustering',
#           'n_clusters': 2}


# ml_args_all = [ml_args_a_e_w,
#                ml_args_a_e_a,
#                ml_args_a_m_a,
#                ml_args_k,
#                ml_args_d_10,
#                ml_args_d_30,
#                ml_args_d_50]
ml_args_all = [ml_args_a_e_w,
                ml_args_a_e_a,
                ml_args_a_m_a,
                ml_args_k]

# ml_args_all = [ml_args_k]



n_clusters_max = 4

for ml_args in ml_args_all:
    print(ml_args['name'])

    # one value of n_clusters
    for start in starts_unique:
        print(' Start:', start, flush=True)
        
        if 'YP' in start:
            subcategories = ['_']
        else:
            subcategories = ['north', 'south']
        
        
        for subcategory in subcategories:
            print('  Subcategory:', subcategory)
        
            case_filters = [start, subcategory]
            df_X = mry_helper.create_X(data, case_filters, y_yes_filters, y_not_filters)
            
            fname_str = '{}_{}_not_{}'.format(''.join(case_filters),
                                              ''.join(y_yes_filters),
                                              ''.join(y_not_filters))
            
            # Plot filtered data to dendrogram (uses always all data of the subcategory)
            model = mry_helper.plot_dendrogram(df_X,
                                                ml_args_agglomerative,
                                                output_folder,
                                                fname_str)
            
            # Create multiple clusters from the filtered data
            for n_clusters in range(1, n_clusters_max+1):
                
                ml_args['n_clusters'] = n_clusters
                df_list = mry_helper.cluster_and_export(df_X,
                                                          ml_args,
                                                          output_folder,
                                                          fname_str)
    














