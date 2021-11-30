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
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering




root_folder = r'C:\Local\laukkara\Data\github\mry-cluster2'

input_folder = os.path.join(root_folder, 'input')
output_folder = os.path.join(root_folder, 'output')


# Load input data

fname = os.path.join(input_folder,
                     'S_RAMI.pickle')
with open(fname, 'rb') as f:
    data = pickle.load(f)


################

def create_X(data, case_filters, y, y_not):

    list_values = []
    list_names = []
    
    for idx_case, case in enumerate(data):
        
        # Check that location, climate and year columns are identical
        if idx_case == 0:
            ids1 = data[case].loc[:, ['location', 'climate', 'year']].copy()
        else:
            ids2 = data[case].loc[:, ['location', 'climate', 'year']].copy()
            
            if not ids1.equals(ids2):
                print('NOT EQUAL:', case)
            
        
        # continue
        for idx_column, column in enumerate(data[case].columns):
            
            cond1 = all(x in case for x in case_filters)
            cond2 = y in column and y_not not in column
            if cond1 and cond2:
                
                column_str = '{}__{}'.format(case, column)
                
                list_values.append(data[case].loc[:, column])
                list_names.append(column_str)
    
    df_X = pd.concat(list_values, axis=1, keys=list_names)
    df_X = pd.concat([df_X, ids1], axis=1)
    
    return(df_X)





def generate_30year_plots(df, output_folder):
    
    labels_for_median = ['1989-2018', 'RCP45-2050','RCP45-2080','RCP85-2050','RCP85-2080']
    
    idxs_col = ['location', 'climate', 'year']
    ids = df[idxs_col].copy()
    df = df.drop(columns=idxs_col).copy()
    
    locations = ids['location'].unique()
    
    
    
    ### UPDATE medians.xlsx file name
    
    xlsx_file_path = os.path.join(output_folder, 'medians.xlsx')
    writer = pd.ExcelWriter(xlsx_file_path, engine='xlsxwriter')
    
    for location in locations:
        
        print('location:', location)
        
        # calculate original means and ranked values
        idxs_rows = ids.loc[:, 'location'] == location
        
        y = df.loc[idxs_rows, :].mean(axis=1).values.reshape((30, -1), order='F')
        x = ids.loc[idxs_rows, 'year'].values[0:30].reshape(-1)
        climates = ids.loc[idxs_rows, 'climate'].unique()
        
        y_df = pd.DataFrame(y, index=x, columns=climates)
        y_df_ranked = y_df.rank(axis=0)
        
        y_df.loc['mean'] = y_df.mean(axis=0)
        y_df.round(2).to_excel(writer, sheet_name=location)
        
        y_df_ranked.loc[:, 'median'] = y_df_ranked.loc[:, labels_for_median].median(axis=1)
        y_df_ranked.to_excel(writer, sheet_name=location+'_ranked')
        
        
        # make plots
        # ranks of different years 1989-2018
        # scatter plots of original values as a function of year
        
    writer.save()
        




def cluster_and_export_agglomerative(df, output_folder, fname_str):
    
    # Drop non-numerical columns
    idxs_col = ['location', 'climate', 'year']
    ids = df[idxs_col].copy()
    df.drop(columns=idxs_col, inplace=True)    
    
    
    #fit model
    model = AgglomerativeClustering(n_clusters=2,
                                    affinity='manhattan',
                                    linkage='average')    
    model.fit(df.T)
    
    # export column names (and values) to excel
    cols0 = df.columns[model.labels_ == 0]
    cols1 = df.columns[model.labels_ == 1]
    
    df0 = df.loc[:, cols0].copy()
    df1 = df.loc[:, cols1].copy()
    
    df0 = pd.concat([ids, df0], axis=1)
    df1 = pd.concat([ids, df1], axis=1)
    
    output_dir = os.path.join(output_folder, 'Agglomerative')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    xlsx_file_path = os.path.join(output_dir, fname_str + '.xlsx')
    
    writer = pd.ExcelWriter(xlsx_file_path, engine='xlsxwriter')
    df0.T.to_excel(writer, sheet_name='df0')
    df1.T.to_excel(writer, sheet_name='df1')
    writer.save()
    
    # generate 30-year plots
    generate_30year_plots(df0, output_folder)
    generate_30year_plots(df1, output_folder)
    
    
    
    return(df0, df1)




def plot_dendrogram(df, fname):
    
    print('AgglomerativeClustering plus dendrograms...')
    # affinity: {“euclidean”, “l1”, “l2”, “manhattan”, “cosine”, “precomputed”}, default euclidean
    # linkage: {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
    
    
    # Fit model
    model = AgglomerativeClustering(n_clusters=None,
                                    affinity='manhattan',
                                    linkage='average',
                                    distance_threshold=0.0)
    
    model.fit(df.T)
    
    # Create linkage matrix (from scikit-learn.org)
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
    fig, ax = plt.subplots()
    sch.dendrogram(linkage_matrix,
                   ax=ax,
                   truncate_mode='none',
                   p=3,
                   no_labels=True)
    fig.savefig(fname, dpi=100, bbox_inches='tight')

    # return
    return(model)



################



# create X

case_filters = ['USP']
y = 'M_'
y_not = 'rank'

df_X = create_X(data, case_filters, y, y_not)

fname_str = '{}_{}_not{}'.format(''.join(case_filters), y, y_not)
fname = os.path.join(output_folder,
                     fname_str+'.png')

df0, df1 = cluster_and_export_agglomerative(df_X,
                                            output_folder,
                                            fname_str)

model = plot_dendrogram(df_X,
                        fname)


# model = AgglomerativeClustering(n_clusters=2,
#                                 affinity='manhattan',
#                                 linkage='average')
# model.fit(df_USP_south_M_not_rank.T)














# def plot_heatmap_corr(df, fname):
#     # cmaps: YlGnBu, RdYlGn
    
#     fig, ax = plt.subplots()
#     sns.heatmap(data=df.corr(),
#                 vmin=-1.0,
#                 vmax=1.0,
#                 cmap="RdYlGn",
#                 xticklabels=False,
#                 yticklabels=False,
#                 cbar_kws={'shrink': 0.8,
#                           'aspect': 30},
#                 ax=ax)
#     fig.savefig(fname, dpi=200, bbox_inches='tight')






# fname = os.path.join(output_folder,
#                      'heatmap_corr_{}_{}_{}.png'.format(''.join(case_filters), y, y_not))
# plot_heatmap_corr(df_USP_south_M_not_rank, fname)















