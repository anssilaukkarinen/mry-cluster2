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

import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

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


################





def cluster_and_export(df, ml_args, output_folder, fname_str):
    
    df_local = df.copy()
    
    # Drop non-numerical columns
    idxs_col = ['location', 'climate', 'year']
    ids = df_local[idxs_col].copy()
    df_local.drop(columns=idxs_col, inplace=True)
    #print('cluster_and_export df.shape, this contains all case+mp pairs:', df_local.shape)
    
    if ml_args['n_clusters'] > df_local.shape[1]:
        ml_args['n_clusters'] = df_local.shape[1]
    
    #fit model
    if ml_args['name'] == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=ml_args['n_clusters'],
                                        affinity=ml_args['affinity'],
                                        linkage=ml_args['linkage'])
        model.fit(df_local.T)
        output_dir = os.path.join(output_folder,
                                  ml_args['name'] + '_' + ml_args['affinity'] + '_' + ml_args['linkage'],
                                  'n_clusters_max{}'.format(ml_args['n_clusters']))
    
    elif ml_args['name'] == 'kmeans':
        model = KMeans(n_clusters=ml_args['n_clusters'],
                       n_init=ml_args['n_init'],
                       max_iter=ml_args['max_iter'],
                       tol=ml_args['tol'],
                       verbose=0)
        model.fit(df_local.T)
        output_dir = os.path.join(output_folder,
                                  'kmeans',
                                  'n_clusters_max{}'.format(ml_args['n_clusters']))
    
    
    elif ml_args['name'] == 'spectralclustering':
        model = SpectralClustering(n_clusters=ml_args['n_clusters'])
        model.fit(df_local.T)
        output_dir = os.path.join(output_folder,
                                  'spectralclustering',
                                  'n_clusters_max{}'.format(ml_args['n_clusters']))
    
    elif ml_args['name'] == 'dbscan':
        model = DBSCAN(eps=ml_args['eps'],
                           min_samples=ml_args['min_samples'],
                           leaf_size=ml_args['leaf_size'])
        model.fit(df_local.T)
        output_dir = os.path.join(output_folder,
                                  'dbscan',
                                  'eps_{:.2f}'.format(ml_args['eps']))
        
        
        
        
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # export column names (and values) to excel
    xlsx_file_path = os.path.join(output_dir, fname_str + '_cases1200.xlsx')
    writer = pd.ExcelWriter(xlsx_file_path, engine='xlsxwriter')
    list_df = []
    
    for label_idx in range(np.max(model.labels_) + 1):
        
        cols_for_idx = df_local.columns[model.labels_ == label_idx]
        df_for_idx = df_local.loc[:, cols_for_idx].copy()
        df_for_idx = pd.concat([ids, df_for_idx], axis=1)
        df_for_idx.T.to_excel(writer, sheet_name='df{}'.format(label_idx))
        list_df.append(df_for_idx)
    
    writer.save()
    
    # generate 30-year plots
    mry_helper.generate_30year_xlsx(list_df, output_dir, fname_str)
    mry_helper.generate_30year_plots(list_df, output_dir, fname_str)
    
    return(list_df)




def plot_dendrogram(df, ml_args, output_folder, fname_str):
    
    df_local = df.copy()
    
    # Drop non-numerical columns
    idxs_col = ['location', 'climate', 'year']
    ids = df_local[idxs_col].copy()
    df_local.drop(columns=idxs_col, inplace=True)
    #print('plot_dendrogram df_local.shape, this contains all case+mp pairs:', df_local.shape)
    
    
    # Fit model and create linkage matrix (from scikit-learn.org)
    model = AgglomerativeClustering(n_clusters=None,
                                    affinity=ml_args['affinity'],
                                    linkage=ml_args['linkage'],
                                    distance_threshold=0.0)
    model.fit(df_local.T)
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for idx, merge in enumerate(model.children_):
        current_count = 0
        for idx_child in merge:
            if idx_child < n_samples:
                current_count += 1 # Leaf node
            else:
                current_count += counts[idx_child - n_samples]
        counts[idx] = current_count
    linkage_matrix = np.column_stack([model.children_,
                                     model.distances_,
                                     counts]).astype(float)
    
    # Plot to file
    output_dir = os.path.join(output_folder,
                              ml_args['name'] + '_' + ml_args['affinity'] + '_' + ml_args['linkage'])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    fig, ax = plt.subplots()
    sch.dendrogram(linkage_matrix,
                   ax=ax,
                   truncate_mode='none',
                   p=3,
                   no_labels=True)
    fname = os.path.join(output_dir,
                         fname_str+'_dendrogram_all.png')
    fig.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)

    # return
    return(model)






################

## What happens, if we select 
# one year without filtering?
# one year for each case and north/south orientation?

# allow automatic clustering from all case+mp pairs?
# allow automatic clustering from filtered case+mp pairs?

starts_all = [x.split('_')[0] for x in data.keys()]
starts_unique = set(starts_all)
print('Unique main case types:')
print(starts_unique)

# Add case+mp counts to legends

#starts_unique = ['UST']

# 'M_', 'RH_'
# 'moverhygr_', 'mmass_', 'T_' & 'max' OR 'min'
y_yes_filters = ['RH_']
y_not_filters = ['rank', 'TRH']

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


ml_args_all = [ml_args_a_e_w,
               ml_args_a_e_a,
               ml_args_a_m_a,
               ml_args_k,
               ml_args_d_10,
               ml_args_d_30,
               ml_args_d_50]




n_clusters_max = 3

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
            model = plot_dendrogram(df_X,
                                    ml_args_agglomerative,
                                    output_folder,
                                    fname_str)
            
            # Create multiple clusters from the filtered data
            for n_clusters in range(1, n_clusters_max+1):
                
                ml_args['n_clusters'] = n_clusters
                df_list = cluster_and_export(df_X,
                                              ml_args,
                                              output_folder,
                                              fname_str)
    














