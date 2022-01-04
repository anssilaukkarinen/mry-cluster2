# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:25:44 2021

@author: laukkara
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

import scipy.stats as ss



#############################################

def create_X(data, case_filters, y_yes_filters, y_not_filters):
    # This function filters spesific case+mp pairs
    

    list_values = []
    list_names = []
    
    for idx_case, case in enumerate(data):
        # loop through cases
        
        # Check that location, climate and year columns are identical
        if idx_case == 0:
            ids1 = data[case].loc[:, ['location', 'climate', 'year']].copy()
        else:
            ids2 = data[case].loc[:, ['location', 'climate', 'year']].copy()
            
            if not ids1.equals(ids2):
                print('NOT EQUAL:', case)
            
        
        for idx_column, column in enumerate(data[case].columns):
            # loop through columns
            
            cond_case_names = all(x in case for x in case_filters)
            
            cond_yes_column_names = all(x in column for x in y_yes_filters)
            
            cond_not_column_names = all(x not in column for x in y_not_filters)
            
            cond_all = cond_case_names and cond_yes_column_names and cond_not_column_names
            
            if cond_all:
                
                column_str = '{}__{}'.format(case, column)
                
                list_values.append(data[case].loc[:, column])
                list_names.append(column_str)
    
    df_X = pd.concat(list_values, axis=1, keys=list_names)
    df_X = pd.concat([df_X, ids1], axis=1)
    
    print('  df_X.shape', df_X.shape)
    
    return(df_X)






############################################

def generate_30year_xlsx(list_df, output_dir, filename_str):
    # list_df = [df0, df1] from clustering algorithm
    
    climates_for_median = ['1989-2018', 'RCP45-2050','RCP45-2080','RCP85-2050','RCP85-2080']
    
    ## Excel files
    for idx_df, df in enumerate(list_df):
        idxs_col = ['location', 'climate', 'year']
        ids = df[idxs_col].copy()
        df = df.drop(columns=idxs_col).copy()
        #print('list_df, df.shape', df.shape) # df.shape == (1200, 2...166), kun USP
        
        locations = ids['location'].unique()
        
        xlsx_file_path = os.path.join(output_dir,
                                      '{}_df{}_n{}_yearClimateMedians.xlsx' \
                                          .format(filename_str, idx_df, df.shape[1]))
        writer = pd.ExcelWriter(xlsx_file_path, engine='xlsxwriter')
        
        for location in locations:
            
            # calculate original medians and ranked values
            idxs_rows = ids.loc[:, 'location'] == location
            
            y = df.loc[idxs_rows, :].median(axis=1).values.reshape((30, -1), order='F')
            x = ids.loc[idxs_rows, 'year'].values[0:30].reshape(-1)
            climates = ids.loc[idxs_rows, 'climate'].unique()
            
            y_df = pd.DataFrame(y, index=x, columns=climates)
            y_df_ranked = y_df.rank(axis=0)
            
            y_df.loc['mean'] = y_df.mean(axis=0)
            y_df.loc[:, 'median'] = y_df.loc[:, climates_for_median].median(axis=1)
            y_df.round(2).to_excel(writer, sheet_name=location)
            
            y_df_ranked.loc['mean'] = y_df_ranked.mean(axis=0)
            y_df_ranked.loc[:, 'median'] = y_df_ranked.loc[:, climates_for_median].median(axis=1)
            y_df_ranked.to_excel(writer, sheet_name=location+'_ranked')
            
        writer.save()
    





############################################

def generate_30year_plots(list_df, output_dir, filename_str):
    
    climates_for_median = ['1989-2018', 'RCP45-2050','RCP45-2080','RCP85-2050','RCP85-2080']
    
    locs = {'Jok': 'Jokioinen', 'Jyv': 'Jyväskylä',
            'Sod': 'Sodankylä', 'Van': 'Vantaa'}
    
    ## Plots
    for location in locs.keys():
        
        # lineplot
        fig_lp, ax_lp = plt.subplots()
        legends = []
        
        for idx_df, df in enumerate(list_df):
            
            # gather data and calculate ranks and medians
            idxs_col = ['location', 'climate', 'year']
            ids = df[idxs_col].copy()
            df = df.drop(columns=idxs_col).copy()
            n_case_plus_mp = df.shape[1]
            #print('n_case_plus_mp:', n_case_plus_mp)
            #print('local df.shape', df.shape) # df.shape == (1200, xxx), specific case+mp pairs
            
            idxs_rows = ids.loc[:, 'location'] == location
            
            # plot, boxplots
            # here we don't calculate medians, but just reshape data for 30 years
            dummy1 = df.loc[idxs_rows, :].values.reshape( (30, -1) , order='F')
            dummy2 = pd.DataFrame(dummy1, index=np.arange(1989, 2019))
            #print('  dummy2', location, idx_df, dummy2.shape) # (30, xxx), all case+mp pairs for certain location
            
            fig_bp, ax_bp = plt.subplots(figsize=(5.0, 0.75*5.0))
            dummy2.T.boxplot(rot=90, ax=ax_bp,
                             flierprops={'marker':'.', 'markersize':2})
            my_labels = [str(x) for x in np.arange(1989,2019)]
            ax_bp.set_xticklabels(my_labels)
            ax_bp.set_title(locs[location])
            if 'M_' in filename_str and 'not_rank' in filename_str:
                ax_bp.set_ylim( (0, 6) )
            elif '_RH_' in filename_str and 'not_rank' in filename_str:
                ax_bp.set_ylim((50, 100))
            fname_bp = os.path.join(output_dir,
                                    '{}_boxplot_{}_df{}_n{}.png' \
                                        .format(filename_str, location, idx_df, n_case_plus_mp))
            fig_bp.savefig(fname_bp, dpi=100, bbox_inches='tight')
            plt.close(fig_bp)
            
            
            # plot, pointplots
            # fig_pp, ax_pp = plt.subplots(figsize=(5, 0.75*5))
            # sns.pointplot(data=dummy2.T, ci='sd',
            #               join=False, scale=0.5, errwidth=0.7,
            #               ax=ax_pp)
            # ax_pp.set_xticklabels([str(x) for x in np.arange(1989, 2019)], rotation=90)
            # ax_pp.set_title(locs[location])
            # if 'M_' in filename_str and 'not_rank' in filename_str:
            #     ax_pp.set_ylim( (0, 6) )
            # elif '_RH_' in filename_str and 'not_rank' in filename_str:
            #     ax_pp.set_ylim( (50, 100) )
            # fname_pp = os.path.join(output_dir,
            #                         '{}_pointplot_{}_df{}_n{}.png' \
            #                             .format(filename_str, location, idx_df, n_case_plus_mp))
            # fig_pp.savefig(fname_pp, dpi=100, bbox_inches='tight')
            # plt.close(fig_pp)
            
            
            
            
            ## Add medians and means
            
            # this part calculates means of all case+mp combinations
            y = df.loc[idxs_rows, :].median(axis=1).values.reshape((30, -1), order='F')
            x = ids.loc[idxs_rows, 'year'].values[0:30].reshape(-1)
            climates = ids.loc[idxs_rows, 'climate'].unique()
            
            y_df = pd.DataFrame(y, index=x, columns=climates)
            # y_df.T.shape == (10, 30) # 10 climates, 30 years
            y_df_ranked = y_df.rank(axis=0)
            
            
            
            y_df.loc[:, 'median'] = y_df.loc[:, climates_for_median].median(axis=1)
            y_df_ranked.loc[:, 'median'] = y_df_ranked.loc[:, climates_for_median].median(axis=1)
            
            y_df.plot(y='median', use_index=True, grid=True, style='-o', ax=ax_lp)
            legends.append('n = {}'.format(n_case_plus_mp)) # n is the number of (1200x1) vectors
            

            
        # lineplots
        if 'M_' in filename_str and 'not_rank' in filename_str:
            ax_lp.set_ylim( (0,6) )
        elif '_RH_' in filename_str and 'not_rank' in filename_str:
            ax_lp.set_ylim( (50.0, 101.0) )
        ax_lp.legend(legends)
        
        output_dir_lineplots = os.path.join(output_dir,
                                            'lineplots')
        if not os.path.exists(output_dir_lineplots):
            os.makedirs(output_dir_lineplots)
        
        fname_lp = os.path.join(output_dir_lineplots,
                              filename_str+'_lineplot_medians_'+location+'.png')
        fig_lp.savefig(fname_lp, dpi=100, bbox_inches='tight')
        plt.close(fig_lp)
    







##################################

def func_parallel_scatter_plots(data, case, output_folder):
    dpi_val = 100
    s_val = 10
    markers = ['o', '^', 'd', 'x', 'v', '<', '>', '2', 's', '*']
    
    print('Scatter plots:', case)
    
    rank_names = [x for x in data[case].columns if 'rank' in x]
    print('case, len(rank_names)', case, len(rank_names))
    
    for idx1, val1 in enumerate(rank_names):
        
        for idx2 in range(idx1+1, len(rank_names)):
            
            # plot folder
            print(idx1, idx2)
            
            figures_folder = os.path.join(output_folder, 'correlations_scatter', case)
            if not os.path.exists(figures_folder):
                os.makedirs(figures_folder)
            
            
            # rank against rank, all points at one go
            x1 = data[case].loc[:, rank_names[idx1]].values
            x2 = data[case].loc[:, rank_names[idx2]].values
            
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(x1, x2, '.')
            ax.grid()
            ax.set_xlabel(rank_names[idx1])
            ax.set_ylabel(rank_names[idx2])
            fname = os.path.join(figures_folder, \
                                rank_names[idx1] + '_' \
                                + rank_names[idx2] + '_all.png')
            fig.savefig(fname, bbox_inches='tight', dpi=dpi_val)
            plt.close(fig)
            
            
            
            # rank against rank, labelling different groups
            c_loc, u_loc = data[case].loc[:, 'location'].factorize()
            
            fig, ax = plt.subplots(figsize=(6,4))
            for idx, val in enumerate(u_loc):
                idxs = data[case].loc[:, 'location'] == val
                ax.scatter(x1[idxs], x2[idxs],
                         marker=markers[idx], s=s_val)
            ax.grid()
            ax.set_xlabel(rank_names[idx1])
            ax.set_ylabel(rank_names[idx2])
            ax.legend(u_loc, bbox_to_anchor=(1.0, 0.5), loc='center left')
            fname = os.path.join(figures_folder, \
                                rank_names[idx1] + '_' \
                                + rank_names[idx2] + '_groups.png')
            fig.savefig(fname, bbox_inches='tight', dpi=dpi_val)
            plt.close(fig)
            

            # value against value, all points at one go
            x1_name = rank_names[idx1].replace('_rank', '')
            x2_name = rank_names[idx2].replace('_rank', '')
            
            x1 = data[case].loc[:, x1_name].values
            x2 = data[case].loc[:, x2_name].values
            
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(x1, x2, '.')
            ax.grid()
            ax.set_xlabel(x1_name)
            ax.set_ylabel(x2_name)
            fname = os.path.join(figures_folder, \
                                x1_name + '_' \
                                + x2_name + '_all.png')
            fig.savefig(fname, bbox_inches='tight', dpi=dpi_val)
            plt.close(fig)
            
            
            
            # value against value, labelling different groups
            c_loc, u_loc = data[case].loc[:, 'location'].factorize()
            
            fig, ax = plt.subplots(figsize=(6,4))            
            for idx, val in enumerate(u_loc):
                idxs = data[case].loc[:, 'location'] == val
                ax.scatter(x1[idxs], x2[idxs],
                            marker=markers[idx], s=s_val)
            ax.grid()
            ax.set_axisbelow(True)
            ax.set_xlabel(x1_name)
            ax.set_ylabel(x2_name)
            ax.legend(u_loc, bbox_to_anchor=(1.0, 0.5), loc='center left')
            fname = os.path.join(figures_folder, \
                                x1_name + '_' \
                                + x2_name + '_groups.png')
            fig.savefig(fname, bbox_inches='tight', dpi=dpi_val)
            plt.close(fig)








################################################

def func_parallel_30year_plots(data, case, output_folder):
    
    dpi_val = 100
    labels_for_median = ['1989-2018', 'RCP45-2050','RCP45-2080','RCP85-2050','RCP85-2080']
    
    # plot folder
    print('30 year plots:', case)
    rank_names = [x for x in data[case].columns if 'rank' in x]
    
    figures_folder_30a = os.path.join(output_folder, 'figures_30a', case)
    if not os.path.exists(figures_folder_30a):
        os.makedirs(figures_folder_30a)
    
    
    
    for idx_rank, rank_name in enumerate(rank_names):
        
        locations = data[case].loc[:, 'location'].unique()
        
        for location in locations:
            
            idxs = data[case].loc[:, 'location'] == location
            
            y_name = rank_name.replace('_rank', '')
            y = data[case].loc[ idxs , y_name].values.reshape((30, -1), order='F')
            x = data[case].loc[ idxs, 'year'].values[0:30].reshape(-1) # reshape((30, 1))
            climates = data[case].loc[idxs, 'climate'].unique()
            
            if len(climates) < 10:
                print('ERROR ERROR ERROR')
                print(case, rank_name, location)
                break
            
            
            
            # Multiple time series for each indicator
            fig, ax = plt.subplots(figsize=(6,4))
            print('shapes:', x.shape, y.shape, np.fliplr(y).shape)
            ax.plot(x, np.fliplr(y), '.-')
            if 'M_' in rank_name:
                ax.set_ylim((0, 6))
            ax.grid()
            ax.set_xlabel('Vuosi')
            ax.set_ylabel(y_name)
            ax.set_title(location)
            ax.legend(climates[::-1], bbox_to_anchor=(1.0, 0.5), loc='center left')
            fname = os.path.join(figures_folder_30a,
                                 y_name + '_' \
                                 + location + '_linegraph.png')
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)
            
            
            # boxplots of absolute values
            fig, ax = plt.subplots(figsize=(6,4))
            _ = plt.boxplot(y, labels=climates)
            ax.grid()
            if 'M_' in rank_name:
                ax.set_ylim((0, 6))
            fig.autofmt_xdate()
            ax.set_ylabel(y_name)
            ax.set_title(location)
            fname = os.path.join(figures_folder_30a,
                                 y_name + '_' \
                                 + location + '_boxplot_abs.png')
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)
            
            
            # boxplots of deviations from median rank
            y_df = pd.DataFrame(y, index=x, columns=climates)
            median_ranks = y_df.rank(axis=0).loc[:, labels_for_median].median(axis=1)
            d_ydf = y_df.rank(axis=0).subtract(median_ranks, axis=0)
            
            fig, ax = plt.subplots(figsize=(6,4))
            _ = plt.boxplot(d_ydf, labels=climates)
            ax.grid()
            fig.autofmt_xdate()
            ax.set_ylabel('$\Delta R$ suureelle ' + rank_name + ', 30 a')
            ax.set_title(location)
            fname = os.path.join(figures_folder_30a,
                                  y_name + '_' \
                                  + location + '_boxplot_dev.png')
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)






############################################

def func_rank_years(root_dir,
                    ML_folder,
                    n_clusters_max,
                    y_yes):
    
    ## fetch median data from xlsx files
    ## 30 values for each filter + location + cluster
    input_dir = os.path.join(root_dir,
                             'output',
                             ML_folder,
                             'n_clusters_max{}'.format(n_clusters_max))
    
    locations = ['Jok', 'Jyv', 'Sod', 'Van']
    
    xlsx_median_files = [x for x in os.listdir(input_dir) \
                             if x.endswith('ClimateMedians.xlsx') and y_yes in x]
    
    results = {}
    
    for location in locations:
        
        results[location] = []
        
        for file in xlsx_median_files:
            
            fname = os.path.join(input_dir,
                                 file)
            # read original values
            dummy = pd.read_excel(fname,
                                  sheet_name=location,
                                  header=0,
                                  index_col=0,
                                  usecols="A,L",
                                  nrows=30)
            
            col_name = file.replace('_yearClimateMedians.xlsx', '')
            dummy.rename(columns={'median': col_name},
                         inplace=True)
            dummy.index.name = location
            
            # add ranks
            for col in dummy.columns:
                dummy[col + '___rank'] = dummy[col].rank()
            
            
            results[location].append(dummy)
    
    ## reorder cases according to original/rank and north/south
    data = {}
    for location in locations:
        
        # concat data to a single dataframe
        data[location] = pd.concat(results[location], axis=1)
        
        # reorder columns
        list_all = data[location].columns
        list_yes_north = []
        list_yes_south = []
        list_no_north = []
        list_no_south = []
        for item in list_all:
            if '___rank' in item:
                
                if 'north' in item or 'YP' in item:
                    list_yes_north.append(item)
                else:
                    list_yes_south.append(item)
                
            else:
                if 'north' in item or 'YP' in item:
                    list_no_north.append(item)
                else:
                    list_no_south.append(item)
        
        list_appended = list_no_north + list_no_south + list_yes_north + list_yes_south
        
        data[location] = data[location].reindex(columns=list_appended)
        
    
    
    
    
    ## group by location and export (this)
    fname = os.path.join(input_dir,
                         'combined_location{}.xlsx'.format(y_yes))
    with pd.ExcelWriter(fname) as writer:
        
        for idx_location, val_location in enumerate(data.keys()):        
            
            data[val_location].T.to_excel(writer,
                                        sheet_name=val_location)
            
            # Formatting
            # (first_row, first_col, last_row, last_col)
            workbook  = writer.book
            worksheet = writer.sheets[val_location]
            
            # top part
            last_row_top = int(data[val_location].T.shape[0] / 2)
            format_xlsx_originals = workbook.add_format({'bg_color': '#FFEB9C',
                                                          'font_color': '#9C5700'})
            
            worksheet.conditional_format(1, 1, last_row_top, 30,
                                          {'type': 'formula',
                                          'criteria':  '=B2>=LARGE($B2:$AE2, 6)',
                                          'format': format_xlsx_originals})
            
            # format bottom part
            format_xlsx_cell = workbook.add_format({'bg_color': '#FFC7CE',
                                                    'font_color': '#9C0006'})
            cell_format = {'type': 'cell',
                           'criteria': 'between',
                           'minimum': 26,
                           'maximum': 30,
                           'format': format_xlsx_cell}
            worksheet.conditional_format(last_row_top+1, 1, last_row_top+1+last_row_top, 30,
                                          cell_format)
            
        
            
        
    # group by case and export
    fname = os.path.join(input_dir,
                         'combined_case{}.xlsx'.format(y_yes))
    
    fname_txt = os.path.join(input_dir,
                             'txt{}.txt'.format(y_yes))
    fhandle_txt = open(fname_txt, 'w')
    
    grouper = {}
    
    with pd.ExcelWriter(fname) as writer:
        
        for idx_location, val_location in enumerate(locations):
            
            if val_location not in grouper.keys():
                grouper[val_location] = {}
            
            ## grouped by case
            unique_starts = set([x.split('_')[0] for x in data[val_location].columns])
            
            for unique_start in unique_starts:
                
                # write to excel
                cols = [x for x in data[val_location].columns if unique_start in x]
                
                data[val_location].loc[:, cols].to_excel(writer,
                                                        sheet_name=unique_start,
                                                        startcol=idx_location*(n_clusters_max*2+1))
                
                # Formatting
                workbook  = writer.book
                worksheet = writer.sheets[unique_start]
                
                # bottom part
                format_xlsx_cell = workbook.add_format({'bg_color': '#FFC7CE',
                                                        'font_color': '#9C0006'})
                cell_format = {'type': 'cell',
                               'criteria': 'between',
                               'minimum': 25,
                               'maximum': 30,
                               'format': format_xlsx_cell}
                worksheet.conditional_format(1, 1, 30, 4*(n_clusters_max*2+1),
                                             cell_format)
                
                # identify candidate years (not used)
                rank_cols = [x for x in cols if '___rank' in x]
                
                median_ranks = data[val_location].loc[:, rank_cols].median(axis=1)
                df_last = data[val_location].loc[median_ranks >= 25, cols].copy()
                
                df_last = df_last.reindex(df_last.loc[:, rank_cols]\
                                          .median(axis=1).sort_values(ascending=False).index)
                
                print(val_location, unique_start,'\n',
                      df_last.index.values, '\n',
                      df_last.values,
                      file=fhandle_txt)
                
                grouper[val_location][unique_start] = df_last
            
    fhandle_txt.close()
    
    return(data, grouper)







############################################

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
    generate_30year_xlsx(list_df, output_dir, fname_str)
    generate_30year_plots(list_df, output_dir, fname_str)
    
    return(list_df)








############################################

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








############################################

def plot_hist_normplot(data, root_dir, ML_folder, y_yes, n_clusters_max):
    
    for location in data.keys():
        
        for key in data[location]:
            
            if '___rank' not in key:
                
                # make folder
                plot_folder = os.path.join(root_dir,
                                     'output',
                                     ML_folder,
                                     'n_clusters_max{}'.format(n_clusters_max),
                                     'hist_normplot_{}'.format(location))
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                
                if 'M' in y_yes:
                    temp_x_label = 'M, -'
                    temp_x_lim = (0.0, 6.0)
                    temp_n_bins = 12
                if 'RH' in y_yes:
                    temp_x_label = 'RH, %'
                    temp_x_lim = (50.0, 100.0)
                    temp_n_bins = 10
                elif 'moverhygr' in y_yes:
                    temp_x_label = 'dw(> 95 % RH)'
                    temp_x_lim = None
                    temp_n_bins = 10
                
                
                # histograms
                fig, ax = plt.subplots(figsize=(6,4))
                data[location][key].plot.hist(bins=temp_n_bins, ax=ax)
                ax.set_xlabel(temp_x_label)
                ax.set_ylabel('Lukumäärä, kpl')
                ax.set_xlim(temp_x_lim)
                fname = os.path.join(plot_folder,
                                     'hist_' + key + '.png')
                fig.savefig(fname, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                
                # normal probability plot (quantile-quantile plot)
                fig, ax = plt.subplots(figsize=(6,4))
                ss.probplot(data[location][key],
                            plot=ax)
                fname = os.path.join(plot_folder,
                                     'normplot_' + key + '.png')
                fig.savefig(fname, dpi=100, bbox_inches='tight')
                plt.close(fig)







#####################

def plot_pointplot_ranks(data, root_dir, ML_folder, y_yes, n_clusters_max):
    
    for location in data.keys():
        
        for key in data[location]:
            
            if '___rank' not in key:
                
                # make folder
                plot_folder = os.path.join(root_dir,
                                     'output',
                                     ML_folder,
                                     'n_clusters_max{}'.format(n_clusters_max),
                                     'scatter_ranks_{}'.format(location))
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                
                # plot
                fig, ax = plt.subplots()
                ax.plot(data[location][key + '___rank'],
                         data[location][key], 'o')
                ax.grid(True)
                ax.set_xlim( (0, 30) )
                
                if 'M' in y_yes:
                    ax.set_ylim( (0, 6) )
                    ax.set_ylabel('M, -')
                elif 'RH' in y_yes:
                    ax.set_ylim( (50, 100) )
                    ax.set_ylabel('RH, %')
                elif 'moverhygr' in y_yes:
                    ax.set_ylabel('dw(> 95 % RH)')
                
                fname = os.path.join(plot_folder,
                                     'scatterrank_' + key + '.png')
                fig.savefig(fname, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                
                
    


















