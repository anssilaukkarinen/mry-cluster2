# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:58:12 2021

@author: laukkara
"""

if __name__ == '__main__':
    # the above line is the new part here

    import os
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.cluster.hierarchy as sch
    
    import multiprocessing
    import itertools
    import sys
    path_to_helper_module = r'C:\Local\laukkara\Data\github\mry-cluster2'
    sys.path.insert(0, path_to_helper_module)
    
    import mry_helper
    

    input_pickle_file_path = r'C:\Local\laukkara\Data\github\mry-cluster2\input\S_RAMI.pickle'
    
    output_folder = r'C:\Local\laukkara\Data\github\mry-cluster2\output\basic_plots'
    
    
    labels_for_median = ['1989-2018', 'RCP45-2050','RCP45-2080','RCP85-2050','RCP85-2080']
    
    dpi_val = 200
    
    locs = ['Sod', 'Jyv', 'Jok', 'Van']
    loc_full_name = ['Sodankylä', 'Jyväskylä', 'Jokioinen', 'Vantaa']
    
    #var_list = ['T_se', 'RH_', 'moverhygr_', 'TRH_580', 'M_']
    var_list = ['moverhygr_', 'M_']
    
    
    with open(input_pickle_file_path, 'rb') as f:
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
    
    
    
    
    
    
    
    ################################
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    def plot_heatmap_corr(df, fname):
        # Correlation coefficient
        # https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb
        
        # cmaps: YlGnBu, RdYlGn
        X = df.corr().values
        X = X[~np.isnan(X).all(axis=1), :] # only non-nan rows
        X = X[:, ~np.isnan(X).all(axis=0)] # only non-nan columns
        d = sch.distance.pdist(1-X)# distances in condensed format
        L = sch.linkage(d, method='complete')# hierarchical clustering defined by the linkage matrix
        t = 0.5*d.max()
        ind = sch.fcluster(L, t, criterion='distance')# form flat clusters from hierarchical clusters
        cols_heatmap = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
        df_sorted = df.reindex(cols_heatmap, axis=1)# reindex_axis is deprecated, use reindex instead
    
        fig, ax = plt.subplots()
        sns.heatmap(data=df_sorted.corr(),
                    vmin=-1.0,
                    vmax=1.0,
                    cmap="RdYlGn",
                    xticklabels=False,
                    yticklabels=False,
                    cbar_kws={'shrink': 0.8,
                              'aspect': 30},
                    ax=ax)
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        
    
    
    
    ########################
    ## Calculate number of case+measurement point (mp) pairs
    # Mmax vectors
    counter = 0
    for case in data:
        Ms = [x for x in data[case].columns if 'M_' in x and '_rank' in x ]
        
        for M in Ms:
            counter += 1
    
    print('Total number of Mmax-cases is:', counter)
    
    # mass of overhygroscopic moisture content
    counter = 0
    for case in data:
        mos = [x for x in data[case].columns if 'moverhygr_' in x and '_rank' in x ]
        
        for mo in mos:
            counter += 1
    
    print('Total number of moverhygr-cases is:', counter)
    
    
    
    ###########################
    
    ## Comparisons of different indicators within each structure
    
    # Scatter plots for indicator vs indicator
    
    # Original format, where the function was inside the for loop:
    # for idx_case, case in enumerate(data):
    #     func_parallel_scatter_plots(data, idx_case, case, output_folder)
    
    # Using multiprocessing, i.e. new version:
    # [x for x in data.keys() if 'YP' in x]
    with multiprocessing.Pool(processes=7) as a_pool:
        a_pool.starmap(mry_helper.func_parallel_scatter_plots,
                        zip(itertools.repeat(data), 
                            data.keys(),
                            itertools.repeat(output_folder)))
    
    
    
    
    
    
    
    
    
    ###############################
    
    # 30a time series and 10 pieces of boxplots, old version:
    # for idx_case, case in enumerate(data):
    #     mry_helper.func_parallel_30year_plots(data, case, output_folder)
        
    # new version:
    # [x for x in data.keys() if 'YP' in x]
    with multiprocessing.Pool(processes=7) as a_pool:
        a_pool.starmap(mry_helper.func_parallel_30year_plots,
                        zip(itertools.repeat(data), 
                            data.keys(),
                            itertools.repeat(output_folder)))
    
    
    
    
    ##############################
    ## plot heatmap of spearman correlations coefficients
    
    
    output_folder_heatmaps = os.path.join(output_folder,
                                          'heatmaps')
    if not os.path.exists(output_folder_heatmaps):
        os.makedirs(output_folder_heatmaps)
    
    for var in var_list:
        df_dict = {}
        for case in data:
            for col in data[case].columns:
                if var in col and 'rank' not in col:
                    col_name = '{}__{}'.format(case, col)
                    df_dict[col_name] = data[case].loc[:, col]
        
        # originals
        df_all = pd.concat(df_dict, axis=1)
        fname = os.path.join(output_folder_heatmaps, 'heatmap_{}_originals.png'.format(var))
        plot_heatmap_corr(df_all, fname)
        
        # ranked
        df_all_ranked = df_all.rank()
        fname = os.path.join(output_folder_heatmaps, 'heatmap_{}_ranked'.format(var))
        plot_heatmap_corr(df_all_ranked, fname)
    
    
    
        
    
    
    
    
    
    
    
    
    
    #############################
    ## medians to excel file
    
    # Calculations
    
    print('Median values to file...')
    
    output_folder_medians = os.path.join(output_folder,
                                          'median_values')
    if not os.path.exists(output_folder_medians):
        os.makedirs(output_folder_medians)
    
    cases_list = []
    values_names_list = []
    locations_list = []
    xs_list = []
    
    values_key = 'M_'
    
    for idx_case, case in enumerate(data):
        
        # plot folder
        #print(case)
        y_names = [x for x in data[case].columns if values_key in x and 'rank' not in x]
        
        for idx_name, y_name in enumerate(y_names):
            
            #print(rank_name)
            
            locations = data[case].loc[:, 'location'].unique()
            
            for location in locations:
                
                #print(location)
                
                idxs = data[case].loc[:, 'location'] == location
                y = data[case].loc[ idxs , y_name].values.reshape((30, -1), order='F')
                x = data[case].loc[ idxs, 'year'].values[0:30].reshape(-1) # reshape((30, 1))
                climates = data[case].loc[idxs, 'climate'].unique()
                
                if len(climates) < 10:
                    print('ERROR ERROR ERROR')
                    print(case, y_name, location)
                    break
                
                y_df = pd.DataFrame(y, index=x, columns=climates)
                
                median_values = y_df.loc[:, labels_for_median].median(axis=1)
                
                cases_list.append(case)
                values_names_list.append(y_name)
                locations_list.append(location)
                xs_list.append(median_values.values.reshape((1,-1)))
                
                
    X = np.concatenate(xs_list, axis=0)
    
    columns_list = [str(x) for x in range(1989, 2019)]
    medians_df = pd.DataFrame(data=X, columns=columns_list)
    medians_df.insert(0, 'case', cases_list)
    medians_df.insert(1, 'mp', values_names_list)
    medians_df.insert(2, 'location', locations_list)
    
    # To excel
    xlsx_file_path = os.path.join(output_folder_medians, 'median_values_{}.xlsx'.format(values_key))
    with pd.ExcelWriter(xlsx_file_path) as excel_writer:
        for loc in locs:
            medians_df.loc[medians_df['location']==loc, :].to_excel(excel_writer,
                                                                    sheet_name=loc)
    
    
    # Plot
    for idx, loc in enumerate(locs):
            
        # Plot, median of medians
        # medians are used, because we are primarily interested of the order
        # between various years
        fig, ax = plt.subplots()
        medians_df.loc[medians_df['location']==loc, columns_list].median(axis=0).plot(title=loc_full_name[idx],
                                                                                    grid=True,
                                                                                    style='-o',
                                                                                    ax=ax)
        ax.set_ylim((0,6))
        ax.set_ylabel('Mediaanien mediaani')
        fname = os.path.join(output_folder_medians,
                             'median_of_medians_{}_{}.png'.format(loc, values_key))
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
        
        
        # Plot, boxplots
        flierprops = {'marker': 'o',
                      'markersize': 0.6}
        print('medians_df.shape:', medians_df.shape)
        fig, ax = plt.subplots()
        medians_df.loc[medians_df['location']==loc, columns_list].boxplot(ax=ax,
                                                                          rot=90,
                                                                          notch=True,
                                                                          flierprops=flierprops)
        _ = ax.set_xticklabels(range(1989, 2019))
        fig.suptitle(loc_full_name[idx])
        ax.set_ylabel('Mediaanien jakauma')
        #ax.grid(b=True, axis='y')
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        fname = os.path.join(output_folder_medians,
                             'boxplot_{}_{}.png'.format(loc, values_key))
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
        
        
        # Plot, pointplots, the ci is for the estimated value, not the whole dataset
        print('medians_df.shape:', medians_df.shape)   
        fig, ax = plt.subplots()
        sns.pointplot(data=medians_df.loc[medians_df['location']==loc, columns_list],
                      ci=95,
                      join=False,
                      scale=0.5,
                      errwidth=0.7,
                      ax=ax)
        ax.set_xticklabels([str(x) for x in np.arange(1989, 2019)], rotation=90)
        fig.suptitle(loc_full_name[idx])
        ax.set_ylabel('Mediaanien jakauma')
        if 'M_' in values_key:
            ax.set_ylim( (0, 6) )
        elif '_RH_' in values_key:
            ax.set_ylim( (50, 100) )
        ax.grid(b=True, axis='y') # lines become unclear, if grid is included
        fname = os.path.join(output_folder_medians,
                             'pointplot_of_mean_{}_{}.png'.format(loc, values_key))
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
    
    
    
    
    
    
    
    #########################
    # Comparisons for variables from other structures
    
    print('Individual comparisons...')
    
    output_folder_comparisons = os.path.join(output_folder,
                                             'comparisons')
    if not os.path.exists(output_folder_comparisons):
        os.makedirs(output_folder_comparisons)
    
    # stud vs wb
    # north vs south
    # expco0336 vs expco169
    # Ti2125viISO13788-1 vs Ti2125viISO13788-5
    
    comps = [{'headers': ['BSWE_EPS100S220_asol25_expco224_north',
                         'BSWE_EPS100S220_asol25_expco224_south'],
             'varnames': ['M_ins_e_max_rank',
                         'M_ins_e_max_rank']},
            {'headers': ['BSWE_EPS100S220_asol25_expco224_south',
                         'BSWE_EPS100S220_asol90_expco224_south'],
             'varnames': ['M_ins_e_max_rank',
                          'M_ins_e_max_rank']},
            {'headers': ['USP_GB_MW250_asol25_expco169_north',
                         'USP_GB_MW250_asol25_expco169_north'],
             'varnames': ['M_stud_e_up_max',
                          'M_wb_i_up_max']},
            {'headers': ['USP_GB_MW250_asol25_expco169_north',
                         'USP_GB_MW250_asol25_expco169_south'],
             'varnames': ['M_stud_e_up_max',
                          'M_stud_e_up_max']},
            {'headers': ['USP_GB_MW250_asol25_expco0336_north',
                         'USP_GB_MW250_asol25_expco169_north'],
             'varnames': ['M_stud_e_up_max',
                          'M_stud_e_up_max']},
            {'headers': ['USP_GB_MW250_asol25_expco169_north_Ti2125viISO13788-1',
                         'USP_GB_MW250_asol25_expco169_north_Ti2125viISO13788-5'],
             'varnames': ['M_stud_e_up_max',
                          'M_stud_e_up_max']}]
    
    # Tähän ylle voi lisätä muita pareja vertailuun...
    
    
    
    
    for idx, vals in enumerate(comps):
        
        x1_str = vals['headers'][0] + '__' + vals['varnames'][0]
        x2_str = vals['headers'][1] + '__' + vals['varnames'][1]
        
        cols = [data[vals['headers'][0]].loc[:, vals['varnames'][0] ],
                data[vals['headers'][1]].loc[:, vals['varnames'][1] ] ]
        df = pd.concat(cols, axis=1,
                       keys=[x1_str,
                             x2_str ])
        
        # time series plot
        ax = df.plot()
        fig = ax.get_figure()
        ax.legend(loc='upper center',
                  bbox_to_anchor=(0.5, -0.1))
        fname = os.path.join(output_folder_comparisons,
                             str(idx) + '_time.png')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
        
        
        # scatter plot, this is not needed anymore, but still kept
        x1_dummy = x1_str.replace('__','\n')
        if 'M_' in x1_dummy:
            x1_label = 'M, -'
        else:
            x1_label = x1_dummy
            
        x2_dummy = x2_str.replace('__','\n')
        if 'M_' in x2_dummy:
            x2_label = 'M, -'
        else:
            x2_label = x2_dummy
            
        
        ax = df.plot.scatter(x=x1_str,
                             y=x2_str)
        ax.set_xlabel(x1_label)
        ax.set_ylabel(x2_label)
        fig = ax.get_figure()
        ax.set_title('Corr: {:.2f}'.format(df.corr().iloc[0, -1]))
        fname = os.path.join(output_folder_comparisons,
                             '{} {} {}_corr.png'.format(str(idx),
                                                  x1_str,
                                                  x2_str))
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
    
    
    
    
    
    
    
    #########################################
    # Boxplots and medians for filtered data, where
    # the filtering is done for every case and north/south orientation
    
    output_folder_medians = os.path.join(output_folder,
                                         'median_values',
                                          'medians_filtered')
    if not os.path.exists(output_folder_medians):
        os.makedirs(output_folder_medians)
    
    starts_all = [x.split('_')[0] for x in data.keys()]
    starts_unique = set(starts_all)
    print('Unique main case types:')
    print(starts_unique)
    
    y_yes_filters = ['M_']
    y_not_filters = ['rank']
    
    for start in starts_unique:
        print('Start:', start, flush=True)
        
        if 'YP' in start:
            subcategories = ['_']
        else:
            subcategories = ['north', 'south']
        
        
        for subcategory in subcategories:
            print(' subcategory:', subcategory)
        
            case_filters = [start, subcategory]
            df_X = mry_helper.create_X(data, case_filters, y_yes_filters, y_not_filters)
            
            fname_str = '{}_{}_not_{}'.format(''.join(case_filters),
                                              ''.join(y_yes_filters),
                                              ''.join(y_not_filters))
            
            mry_helper.generate_30year_xlsx([df_X], output_folder_medians, fname_str)
            mry_helper.generate_30year_plots([df_X], output_folder_medians, fname_str)
            
    
