# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 09:37:30 2021

@author: laukkara

This is run after mrycluster2.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mry_helper

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=np.inf)


root_dir = r'C:\Local\laukkara\Data\github\mry-cluster2'

locations = ['Jok', 'Van', 'Jyv', 'Sod']



##
ML_folders = ['agglomerative_euclidean_average',
              'agglomerative_euclidean_ward',
              'agglomerative_manhattan_average',
              'kmeans']
y_yess = ['_M_', '_RH_', '_moverhygr_']
n_clusters_maxs = [1, 2, 3, 4]


##
# ML_folders = ['agglomerative_euclidean_average']
# y_yess = ['_M_', '_RH_']
# n_clusters_maxs = [3, 4]

##
# ML_folders = ['kmeans']
# y_yess = ['_M_']
# n_clusters_maxs = [3]





dpi_val = 200

##################
output_folder = os.path.join(root_dir,
                             'output',
                             'rank_years')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)



## Read and export
# Read in medians from ClimateMedians.xlsx files
# Export to combined_location files
groupers = {}

for ML_folder in ML_folders:
    print(ML_folder)
    
    groupers[ML_folder] = {}
    
    for y_yes in y_yess:
        print('  ', y_yes)
        
        groupers[ML_folder][y_yes] = {}

        for n_clusters_max in n_clusters_maxs:
            print(' ', n_clusters_max)

            data, grouper = mry_helper.func_rank_years(root_dir,
                                                       ML_folder,
                                                       n_clusters_max,
                                                       y_yes)
            
            #mry_helper.plot_hist_normplot(data, root_dir, ML_folder, y_yes, n_clusters_max)
            #mry_helper.plot_pointplot_ranks(data, root_dir, ML_folder, y_yes, n_clusters_max)

            groupers[ML_folder][y_yes][n_clusters_max] = grouper




## read combined_location, export to step1

for ML_folder in ML_folders:
    
    for n_clusters_max in n_clusters_maxs:
        
        for y_yes in y_yess:
            
            for location in locations:
                
                # read in data
                fname = os.path.join(root_dir,
                                     'output',
                                     ML_folder,
                                     'n_clusters_max{}'.format(n_clusters_max),
                                     'combined_location{}.xlsx'.format(y_yes))
            
                df = pd.read_excel(fname, sheet_name=location, index_col=0)
                
                ## north
                # find suitable rows
                rows_rank_north = [x for x in df.index if '___rank' in x and 'north' in x]
                rows_values_north = [x for x in df.index if '___rank' not in x and 'north' in x]
                
                # create new dataframes
                df_north_ranks = df.loc[df.index.isin(rows_rank_north)].copy()
                df_north_values = df.loc[df.index.isin(rows_values_north)].copy()
                
                # create median rank row
                df_north_ranks.loc['median_rank'] = df_north_ranks.median(axis=0)
                
                # sort df_rank according to median_rank
                df_north_ranks.sort_values(by='median_rank', axis=1, inplace=True)
                
                # sort df_values according to column order of df_rank
                df_north_values = df_north_values.reindex(columns=df_north_ranks.columns[::-1]).copy()
                
                # plot
                x_labels = mry_helper.get_new_names(df_north_values.index)
                
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                df_north_values.iloc[:, 0:5].plot(ax=ax,
                                                  grid=True,
                                                  style='.-')
                ax.set_xticks(range(0, len(df_north_values.index)))
                ax.set_xticklabels(x_labels, rotation=90)
                if y_yes == '_M_':
                    ax.set_ylabel('M, -')
                    ax.set_ylim( (0,6) )
                elif y_yes == '_RH_':
                    ax.set_ylabel('RH, %')
                    ax.set_ylim( (50, 100) )
                elif y_yes == '_moverhygr_':
                    ax.set_ylabel('$\Delta m$, kg/m$^2$')
                else:
                    print('other ylabel!')
                
                fname = os.path.join(output_folder,
                                      '{} {} {} {} {}'.format('north',
                                                              ML_folder,
                                                              n_clusters_max,
                                                              y_yes,
                                                              location))
                fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
                plt.close(fig)
                
                
                
                ## south
                rows_rank_south = [x for x in df.index if '___rank' in x and 'south' in x]
                rows_values_south = [x for x in df.index if '___rank' not in x and 'south' in x]
                
                df_south_ranks = df.loc[df.index.isin(rows_rank_south)].copy()
                df_south_values = df.loc[df.index.isin(rows_values_south)].copy()
                
                df_south_ranks.loc['median_rank'] = df_south_ranks.median(axis=0)
                
                df_south_ranks.sort_values(by='median_rank', axis=1, inplace=True)
                
                df_south_values = df_south_values.reindex(columns=df_south_ranks.columns[::-1]).copy()
                
                # plot
                x_labels = mry_helper.get_new_names(df_south_values.index)
                
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                df_south_values.iloc[:, 0:5].plot(ax=ax,
                                                  grid=True,
                                                  style='.-')
                ax.set_xticks(range(0, len(df_south_values.index)))
                ax.set_xticklabels(x_labels, rotation=90)
                if y_yes == '_M_':
                    ax.set_ylabel('M, -')
                    ax.set_ylim( (0,6) )
                elif y_yes == '_RH_':
                    ax.set_ylabel('RH, %')
                    ax.set_ylim( (50, 100) )
                elif y_yes == '_moverhygr_':
                    ax.set_ylabel('$\Delta m$, kg/m$^2$')
                else:
                    print('other ylabel!')
                
                fname = os.path.join(output_folder,
                                      '{} {} {} {} {}'.format('south',
                                                              ML_folder,
                                                              n_clusters_max,
                                                              y_yes,
                                                              location))
                fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
                plt.close(fig)
                
    
                ## YP
                rows_rank_YP = [x for x in df.index if '___rank' in x and 'YP' in x]
                rows_values_YP = [x for x in df.index if '___rank' not in x and 'YP' in x]
                
                df_YP_ranks = df.loc[df.index.isin(rows_rank_YP)].copy()
                df_YP_values = df.loc[df.index.isin(rows_values_YP)].copy()
                
                df_YP_ranks.loc['median_rank'] = df_YP_ranks.median(axis=0)
                
                df_YP_ranks.sort_values(by='median_rank', axis=1, inplace=True)
                
                df_YP_values = df_YP_values.reindex(columns=df_YP_ranks.columns[::-1]).copy()
                
                # plot
                x_labels = mry_helper.get_new_names(df_YP_values.index)
                
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                df_YP_values.iloc[:, 0:5].plot(ax=ax,
                                                  grid=True,
                                                  style='.-')
                ax.set_xticks(range(0, len(df_YP_values.index)))
                ax.set_xticklabels(x_labels, rotation=90)
                if y_yes == '_M_':
                    ax.set_ylabel('M, -')
                    ax.set_ylim( (0,6) )
                elif y_yes == '_RH_':
                    ax.set_ylabel('RH, %')
                    ax.set_ylim( (50, 100) )
                elif y_yes == '_moverhygr_':
                    ax.set_ylabel('$\Delta m$, kg/m$^2$')
                else:
                    print('other ylabel!')
                
                fname = os.path.join(output_folder,
                                     '{} {} {} {} {}'.format('YP',
                                                             ML_folder,
                                                              n_clusters_max,
                                                              y_yes,
                                                              location))
                fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
                plt.close(fig)
                
    









