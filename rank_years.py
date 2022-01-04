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



## Export to file, version 1, this is not used

for ML_folder in groupers:
    print(ML_folder)
    
    for y_yes in groupers[ML_folder]:
        print(' ', y_yes)
        
        # Create file and close
        for location1 in locations:
            
            fname = os.path.join(root_dir,
                                 'output',
                                 ML_folder,
                                 '{}_y_yes{}.txt'.format(location1, y_yes))
            fhandle_groups = open(fname, 'w')
            fhandle_groups.close()
        
        
        for n_clusters_max in groupers[ML_folder][y_yes]:
            
            for location in groupers[ML_folder][y_yes][n_clusters_max]:
                print('  ', location)
                
                starts_uniques = groupers[ML_folder][y_yes][n_clusters_max][location].keys()
                
                for starts_unique in sorted(starts_uniques):
                    #print(, file=fhandle_groups)
                    
                    idxs = groupers[ML_folder][y_yes][n_clusters_max][location][starts_unique].index.values
                    
                    # write to file
                    fname = os.path.join(root_dir,
                                         'output',
                                         ML_folder,
                                         '{}_y_yes{}.txt'.format(location, y_yes))
                    fhandle_groups = open(fname, 'a')
                    print(starts_unique, n_clusters_max, idxs, file=fhandle_groups)
                    fhandle_groups.close()
                    
                        
                        
    

## read combined_location, export to step1

fname = os.path.join(root_dir,
                     'output',
                     'combine_step1_north.txt')
fhandle_txt_north = open(fname, 'w')
s = 'ML_folder n_clusters_max criteria location year26 year27 year28 year29 year30'
fhandle_txt_north.write(s+'\n')

fname = os.path.join(root_dir,
                     'output',
                     'combine_step1_south.txt')
fhandle_txt_south = open(fname, 'w')
fhandle_txt_south.write(s+'\n')

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
                
                # north
                rows_rank_north = [x for x in df.index if '___rank' in x and 'north' in x]
                
                meds = df.loc[df.index.isin(rows_rank_north)] \
                        .median(axis=0).sort_values(ascending=True).tail(5).index.values
                
                print(ML_folder, n_clusters_max, y_yes, location,
                      str(meds).replace('[','').replace(']',''),
                      file=fhandle_txt_north)
                
                
                # south
                rows_rank_south = [x for x in df.index if '___rank' in x and 'south' in x]
                
                meds = df.loc[df.index.isin(rows_rank_south)] \
                        .median(axis=0).sort_values(ascending=True).tail(5).index.values
                
                print(ML_folder, n_clusters_max, y_yes, location,
                      str(meds).replace('[','').replace(']',''),
                      file=fhandle_txt_south)

fhandle_txt_north.close()
fhandle_txt_south.close()



## read step1, export to step2
# north
fname = os.path.join(root_dir,
                     'output',
                     'combine_step1_north.txt')
df_north = pd.read_csv(fname, sep='\s+')
fname = os.path.join(root_dir,
                     'output',
                     'combine_step2_north.txt')
fhandle_txt_north = open(fname, 'w')
s = 'ML_folder criteria location 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018'
fhandle_txt_north.write(s+'\n')

# south
fname = os.path.join(root_dir,
                     'output',
                     'combine_step1_south.txt')
df_south = pd.read_csv(fname, sep='\s+')
fname = os.path.join(root_dir,
                     'output',
                     'combine_step2_south.txt')
fhandle_txt_south = open(fname, 'w')
fhandle_txt_south.write(s+'\n')


changer = {'_M_': 'M',
           '_moverhygr_': 'dw(>95%RH)',
           '_RH_': 'RH'}

for ML_folder in ML_folders:
    for location in locations:
        for y_yes in y_yess:
            
            # north
            idxs = (df_north['ML_folder'] == ML_folder) \
                    & (df_north['location'] == location) \
                    & (df_north['criteria'] == y_yes)
            
            years = df_north[idxs].iloc[:,-5:].values
            
            cumsum = np.zeros(30)
            
            # this part loops through all the n_clusters_max values
            for row in range(years.shape[0]):
                for col in range(years.shape[1]):
                    cumsum[ int(years[row, col]) - 1989] += col+1
            
            print(ML_folder, changer[y_yes], location,
                  str(cumsum).replace('[','').replace(']',''),
                  file=fhandle_txt_north)
            
            
            # south
            idxs = (df_south['ML_folder'] == ML_folder) \
                    & (df_south['location'] == location) \
                    & (df_south['criteria'] == y_yes)
            
            years = df_south[idxs].iloc[:,-5:].values
            
            cumsum = np.zeros(30)
            
            for row in range(years.shape[0]):
                for col in range(years.shape[1]):
                    cumsum[ int(years[row, col]) - 1989] += col+1
            
            print(ML_folder, changer[y_yes], location,
                  str(cumsum).replace('[','').replace(']',''),
                  file=fhandle_txt_south)


fhandle_txt_north.close()
fhandle_txt_south.close()
                    


## step 3
## read in step 2 and calculate sums from ML_folders

fname_out_step3 = os.path.join(root_dir,
                         'output',
                         'combine_step3.xlsx')

fname_txtout_step3 = os.path.join(root_dir,
                                  'output',
                                  'step3.txt')
fhandle_txt_step3 = open(fname_txtout_step3, 'w')

orientations = ['north', 'south']

with pd.ExcelWriter(fname_out_step3) as writer:
    
    workbook  = writer.book
    format_xlsx_cell = workbook.add_format({'bg_color': '#FFC7CE',
                                            'font_color': '#9C0006'})
    cell_format = {'type': 'cell',
                   'criteria': '>',
                   'value': 0,
                   'format': format_xlsx_cell}
    
    for orientation in orientations:
        
        print('Orientation:', orientation,
              file=fhandle_txt_step3)

        fname_in = os.path.join(root_dir,
                                    'output',
                                    'combine_step2_{}.txt'.format(orientation))
        df = pd.read_csv(fname_in,
                         sep='\s+')
        
        df_step3 = df.groupby(by=['location', 'criteria']) \
            .sum().copy()
        df_step3.to_excel(writer, sheet_name=orientation)
            
        # Formatting
        worksheet = writer.sheets[orientation]
        
        worksheet.conditional_format(1, 2, 12, 31,
                                     cell_format)
        
        # print years for highest scores
        for idx_multi in df_step3.index:
            dummy = df_step3.loc[idx_multi, :][df_step3.loc[idx_multi, :]>0].sort_values()
            print(str(idx_multi).replace('(','').replace(')','').replace("'",""),
                  str(dummy.tail(3)[::-1].index.values).replace('[','').replace(']','').replace("'", ""),
                  file=fhandle_txt_step3)



# step 4














