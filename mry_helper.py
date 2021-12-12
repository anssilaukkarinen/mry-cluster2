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
        fname_lp = os.path.join(output_dir,
                              filename_str+'_lineplot_medians_'+location+'.png')
        fig_lp.savefig(fname_lp, dpi=100, bbox_inches='tight')
        plt.close(fig_lp)
    







##################################

def func_parallel_scatter_plots(data, case, output_folder):
    dpi_val = 100
    s_val = 10
    markers = ['o', '^', 'd', 'x', 'v', '<', '>', '2', 's', '*']
    
    #print('Scatter plots:', idx_case, case)
    print('Scatter plots:', case)
    
    #rank_names = [x for x in data[case].columns if 'rank' in x and 'M_' in x]
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
        
        #print(rank_name)
        
        locations = data[case].loc[:, 'location'].unique()
        
        for location in locations:
            
            #print(location)
            
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






