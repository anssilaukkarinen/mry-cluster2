# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:56:31 2022

@author: laukkara

This code is intended to test the criticality of selected
moisture design years (MDY) against structure type - orientation groups.
The previous design years are shown as reference.
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# The following code is run twice, change "condition" in between
# condition = 'M_'
condition = 'moverhygr_'








input_file = os.path.join(r'C:\Local\laukkara\Data\github\mry-cluster2\input',
                          'S_RAMI.pickle')

with open(input_file, 'rb') as f:
    data = pickle.load(f)


output_folder = os.path.join(r'C:\Local\laukkara\Data\github\mry-cluster2\output',
                             'post_test')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_folder_rcp = os.path.join(r'C:\Local\laukkara\Data\github\mry-cluster2\output',
                             'post_test', 'RCP')
if not os.path.exists(output_folder_rcp):
    os.makedirs(output_folder_rcp)



climate_names = data['UST_WFB25_MW250_asol25_expco169_south_brick512'].loc[:,'climate'].unique()
climate_names = list(climate_names)

locs = ['Jok', 'Van', 'Jyv', 'Sod']
years = range(1989, 2019)

climates_for_median = ['1989-2018', 'RCP45-2050', 'RCP45-2080','RCP85-2050', 'RCP85-2080']

figseiz = (5.5, 3.5)
dpi_val = 200

converter_struc = {'USP': 'PRP',
                  'UST': 'PRT',
                  'BSWE': 'BSW',
                  'PICB': 'EHR',
                  'TRC': 'BSR',
                  'SW': 'PVP',
                  'USH': 'USH',
                  'YP': 'YP'}

converter_orient = {'north': 'pohjoinen',
                    'south': 'etelä',
                    'other': 'vaakapinta'}


converter_loc = {'Jok': 'Jokioinen',
                 'Jyv': 'Jyväskylä',
                 'Sod': 'Sodankylä',
                 'Van': 'Vantaa'}


converter_condition = {'M_': 'Mmax',
                       'moverhygr_': 'dm95'}


########################################
## This section looks only at certain years


res = {}


for case in data:
    
    structure = case.split('_')[0]
    if 'YP' in structure:
        structure = 'YP'
    
    
    if 'north' in case:
        orientation = 'north'
    elif 'south' in case:
        orientation = 'south'
    else:
        orientation = 'other'
    
    struc_orient = '{}_{}'.format(structure, orientation)
    
    if struc_orient not in res:
        
        res[struc_orient] = {}
        
        res[struc_orient]['jok2004'] = {}
        res[struc_orient]['van2007'] = {}
        
        res[struc_orient]['jok2011'] = {}
        res[struc_orient]['jok2017'] = {}
        res[struc_orient]['van2017'] = {}
        res[struc_orient]['van2004'] = {}
        
    
    
    
    idxs_jok2004 = (data[case].loc[:, 'location'] == 'Jok') \
                 & (data[case].loc[:, 'year'] == 2004)
             
    idxs_van2007 = (data[case].loc[:, 'location'] == 'Van') \
                 & (data[case].loc[:, 'year'] == 2007)
    
    
    idxs_jok2011 = (data[case].loc[:, 'location'] == 'Jok') \
                 & (data[case].loc[:, 'year'] == 2011)
                 
    idxs_jok2017 = (data[case].loc[:, 'location'] == 'Jok') \
                 & (data[case].loc[:, 'year'] == 2017)
    
    idxs_van2004 = (data[case].loc[:, 'location'] == 'Van') \
                 & (data[case].loc[:, 'year'] == 2004)
                 
    idxs_van2017 = (data[case].loc[:, 'location'] == 'Van') \
                 & (data[case].loc[:, 'year'] == 2017)
         
    cols = [x for x in data[case].columns if 'M_' in x and '_rank' not in x]
    
    for col in cols:
        
        key = '{}__{}'.format(case, col)
        
        
        y_jok2004 = data[case].loc[idxs_jok2004, col]
        y_van2007 = data[case].loc[idxs_van2007, col]
        
        y_jok2011 = data[case].loc[idxs_jok2011, col]
        y_jok2017 = data[case].loc[idxs_jok2017, col]
        y_van2004 = data[case].loc[idxs_van2004, col]
        y_van2017 = data[case].loc[idxs_van2017, col]


        res[struc_orient]['jok2004'][key] = y_jok2004
        res[struc_orient]['van2007'][key] = y_van2007
        
        res[struc_orient]['jok2011'][key] = y_jok2011
        res[struc_orient]['jok2017'][key] = y_jok2017
        res[struc_orient]['van2004'][key] = y_van2004
        res[struc_orient]['van2017'][key] = y_van2017
        
        

res_med = {}

for struc_orient in res:
    
    jok2004_med = pd.DataFrame(res[struc_orient]['jok2004']).median(axis=1)
    van2007_med = pd.DataFrame(res[struc_orient]['van2007']).median(axis=1)
    
    jok2011_med = pd.DataFrame(res[struc_orient]['jok2011']).median(axis=1)
    jok2017_med = pd.DataFrame(res[struc_orient]['jok2017']).median(axis=1)
    van2004_med = pd.DataFrame(res[struc_orient]['van2004']).median(axis=1)
    van2017_med = pd.DataFrame(res[struc_orient]['van2017']).median(axis=1)
    
    Y = pd.concat( (jok2011_med.reset_index(drop=True),
                    jok2017_med.reset_index(drop=True),
                    van2004_med.reset_index(drop=True),
                    van2017_med.reset_index(drop=True),
                    van2007_med.reset_index(drop=True),
                    jok2004_med.reset_index(drop=True)),
                  axis=1,
                  keys=['Jokioinen 2011',
                        'Jokioinen 2017',
                        'Vantaa 2004',
                        'Vantaa 2017',
                        'Vantaa 2007',
                        'Jokioinen 2004'])
    Y.index = climate_names
    
    res_med[struc_orient] = Y


for struc_orient in res_med:
    fig, ax = plt.subplots(figsize=figseiz)
    res_med[struc_orient].plot(rot=90, ax=ax)
    #ax.set_ylim(bottom=-0.01)
    title_struc = converter_struc[struc_orient.split('_')[0]]
    title_orient = converter_orient[struc_orient.split('_')[1]]
    dummy = '{} {}'.format(title_struc, title_orient)
    ax.set_title(dummy)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax.set_xticks(range(10))
    ax.set_xticklabels(climate_names)
    
    ax.grid(True)
    
    fname = os.path.join(output_folder_rcp,
                         'RCPs {}.png'.format(struc_orient))
    fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)
    
    
    
    


#############################################
## This section looks at all the years, Jokioinen and Vantaa





res2 = {}


output_folder_jok_van = os.path.join(output_folder,
                                     'Jok_Van')
if not os.path.exists(output_folder_jok_van):
    os.makedirs(output_folder_jok_van)


for case in data:
    
    structure = case.split('_')[0]
    if 'YP' in structure:
        structure = 'YP'
    
    if 'north' in case:
        orientation = 'north'
    elif 'south' in case:
        orientation = 'south'
    else:
        orientation = 'other'
    
    struc_orient = '{}_{}'.format(structure, orientation)
    
    if struc_orient not in res2:
        res2[struc_orient] = {}
    
    cols = [x for x in data[case].columns if condition in x and '_rank' not in x]
    
    for loc in locs:
        
        for year in years:
            
            loc_year = '{}_{}'.format(loc, year)
            
            if loc_year not in res2[struc_orient]:
                res2[struc_orient][loc_year] = {}
            
                        
            idxs = (data[case].loc[:, 'location'] == loc) \
                 & (data[case].loc[:, 'year'] == year)
            
            for col in cols:
                
                case_col = '{}__{}'.format(case, col)
                
                df = data[case].loc[idxs, [col]]
                df.reset_index(drop=True, inplace=True)
                df.index = climate_names
                
                
                res2[struc_orient][loc_year][case_col] = df


res3 = {}

for struc_orient in res2:
    
    res3[struc_orient] = {}
    
    dummy = {}
    
    for loc_year in res2[struc_orient]:
        
        df = pd.concat(res2[struc_orient][loc_year], axis=1)
        df.columns = res2[struc_orient][loc_year]
        
        dummy[loc_year] = df.median(axis=1)
    
    res3[struc_orient] = pd.concat(dummy, axis=1)
        
        

res4 = {}

for struc_orient in res3:
    
    dummy = {}
    
    for loc in locs:
        cols = [x for x in res3[struc_orient].columns if loc in x]
        df_med = res3[struc_orient].loc[climates_for_median, cols] \
                      .median(axis=0).sort_values(ascending=False)
        
        dummy[loc] = df_med
    res4[struc_orient] = pd.concat(dummy, axis=1)
    

# plot
for struc_orient in res4:
    
    title_struc = converter_struc[struc_orient.split('_')[0]]
    title_orient = converter_orient[struc_orient.split('_')[1]]
    dummy = '{} {}'.format(title_struc, title_orient)
    
    fig, ax = plt.subplots(figsize=figseiz)
    res4[struc_orient].plot(title=dummy,
                            style='-o', ms=2, ax=ax)
    
    
    
    res4[struc_orient].loc['Jok_2011'].plot(style='^', markerfacecolor='none', label='Jok 2011', ax=ax)
    res4[struc_orient].loc['Jok_2017'].plot(style='v', markerfacecolor='none', label='Jok 2017', ax=ax)
    res4[struc_orient].loc['Van_2017'].plot(style='s', markerfacecolor='none', label='Van 2017', ax=ax)
    res4[struc_orient].loc['Van_2004'].plot(style='d', markerfacecolor='none', label='Van 2004', ax=ax)
    res4[struc_orient].loc['Van_2007'].plot(style='+', markerfacecolor='none', label='Van 2007', ax=ax)
    res4[struc_orient].loc['Jok_2004'].plot(style='o', markerfacecolor='none', label='Jok 2004', ax=ax)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    #ax.set_ylim(bottom=-0.01)
    ax.grid(True)
    
    ax.set_xticklabels("")
    
    fname = os.path.join(output_folder_jok_van,
                         'locs {} {}.png'.format(condition, struc_orient))
    fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)
    
    
    









# Jyväskylä, Sodankylä

res2 = {}


output_folder_jyv_sod = os.path.join(output_folder,
                                     'Jyv_Sod')
if not os.path.exists(output_folder_jyv_sod):
    os.makedirs(output_folder_jyv_sod)


for case in data:
    
    structure = case.split('_')[0]
    if 'YP' in structure:
        structure = 'YP'
    
    if 'north' in case:
        orientation = 'north'
    elif 'south' in case:
        orientation = 'south'
    else:
        orientation = 'other'
    
    struc_orient = '{}_{}'.format(structure, orientation)
    
    if struc_orient not in res2:
        res2[struc_orient] = {}
    
    cols = [x for x in data[case].columns if condition in x and '_rank' not in x]
    
    for loc in locs:
        
        for year in years:
            
            loc_year = '{}_{}'.format(loc, year)
            
            if loc_year not in res2[struc_orient]:
                res2[struc_orient][loc_year] = {}
            
                        
            idxs = (data[case].loc[:, 'location'] == loc) \
                 & (data[case].loc[:, 'year'] == year)
            
            for col in cols:
                
                case_col = '{}__{}'.format(case, col)
                
                df = data[case].loc[idxs, [col]]
                df.reset_index(drop=True, inplace=True)
                df.index = climate_names
                
                
                res2[struc_orient][loc_year][case_col] = df


res3 = {}

for struc_orient in res2:
    
    res3[struc_orient] = {}
    
    dummy = {}
    
    for loc_year in res2[struc_orient]:
        
        df = pd.concat(res2[struc_orient][loc_year], axis=1)
        df.columns = res2[struc_orient][loc_year]
        
        dummy[loc_year] = df.median(axis=1)
    
    res3[struc_orient] = pd.concat(dummy, axis=1)
        
        

res4 = {}

for struc_orient in res3:
    
    dummy = {}
    
    for loc in locs:
        cols = [x for x in res3[struc_orient].columns if loc in x]
        df_med = res3[struc_orient].loc[climates_for_median, cols] \
                      .median(axis=0).sort_values(ascending=False)
        
        dummy[loc] = df_med
    res4[struc_orient] = pd.concat(dummy, axis=1)
    

# plot
for struc_orient in res4:
    
    title_struc = converter_struc[struc_orient.split('_')[0]]
    title_orient = converter_orient[struc_orient.split('_')[1]]
    dummy = '{} {}'.format(title_struc, title_orient)
    
    fig, ax = plt.subplots(figsize=figseiz)
    res4[struc_orient].plot(title=dummy,
                            style='-o', ms=2, ax=ax)
    
    
    
    res4[struc_orient].loc['Jyv_2011'].plot(style='^', markerfacecolor='none', label='Jyv 2011', ax=ax)
    res4[struc_orient].loc['Jyv_1996'].plot(style='v', markerfacecolor='none', label='Jyv 1996', ax=ax)
    res4[struc_orient].loc['Sod_2015'].plot(style='s', markerfacecolor='none', label='Sod 2015', ax=ax)
    res4[struc_orient].loc['Sod_2014'].plot(style='d', markerfacecolor='none', label='Sod 2014', ax=ax)
    
    res4[struc_orient].loc['Van_2007'].plot(style='+', markerfacecolor='none', label='Van 2007', ax=ax)
    res4[struc_orient].loc['Jok_2004'].plot(style='o', markerfacecolor='none', label='Jok 2004', ax=ax)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    #ax.set_ylim(bottom=-0.01)
    ax.grid(True)
    
    ax.set_xticklabels("")
    
    fname = os.path.join(output_folder_jyv_sod,
                         'locs {} {}.png'.format(condition, struc_orient))
    fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)
    
    
    















###################################
# boxplots of climate change


locs5 = data['UST_WFB25_MW250_asol25_expco169_south_brick512'].loc[:, 'location']
climates5 = data['UST_WFB25_MW250_asol25_expco169_south_brick512'].loc[:, 'climate']

res5 = {}


for case in data:
    
    if 'USH_200' in case or 'USH_300' in case:
        continue
    
    structure = case.split('_')[0]
    if 'YP' in structure:
        structure = 'YP'
    
    if 'north' in case:
        orientation = 'north'
    elif 'south' in case:
        orientation = 'south'
    else:
        orientation = 'other'
    
    struc_orient = '{}_{}'.format(structure, orientation)
    
    if struc_orient not in res5:
        res5[struc_orient] = []
    
    cols = [x for x in data[case].columns if condition in x \
                                            and '_rank' not in x \
                                            and 'ins_i' not in x \
                                            and 'stud_i' not in x]
    
    df = data[case].loc[:, cols]
    res5[struc_orient].append(df)



# create one large dataframe
res6 = []

for struc_orient in res5:
    
    df_struc_orient = pd.concat(res5[struc_orient], axis=1)
    
    for loc in locs:
        
        for climate in climate_names:
            
            idxs = (locs5 == loc) \
                 & (climates5 == climate)
            
            vals = df_struc_orient.loc[idxs, :].mean(axis=0).values
            
            for val in vals:
                struc_name = converter_struc[struc_orient.split('_')[0]]
                orient_name = converter_orient[struc_orient.split('_')[1]]
                                
                dummy = {'struc_orient': '{} {}'.format(struc_name, orient_name),
                         'Paikkakunta': converter_loc[loc],
                         '30-vuotisjakso': climate,
                         converter_condition[condition]: val}
                res6.append(dummy)

df = pd.DataFrame(res6)



# plot

output_folder_mry = os.path.join(output_folder,
                                 'boxplot_mry')

if not os.path.exists(output_folder_mry):
    os.makedirs(output_folder_mry)


flierprops = dict(marker='.', markersize=2)


for struc_orient in df['struc_orient'].unique():
    
    fig, ax = plt.subplots()
    
    idxs = (df['struc_orient'] == struc_orient)
    
    sns.boxplot(data=df.loc[idxs],
                x='30-vuotisjakso',
                y=converter_condition[condition],
                hue='Paikkakunta',
                flierprops=flierprops,
                ax=ax)
    ax.set_title('{} (n = {:.0f})'.format(struc_orient, idxs.sum()/(4*10)  ))
    ax.grid(True)
    ax.set_axisbelow(True)
    if struc_orient != 'EHR pohjoinen' \
        and struc_orient != 'PVP etelä' \
        and struc_orient != 'PVP pohjoinen':
        ax.set_ylim(bottom=-0.01)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    xtl = [x.get_text() for x in ax.get_xticklabels()]
    xtl_new = []
    for xt in xtl:
        if 'RCP26' in xt:
            begin = xt.split('-')[0].replace('RCP26', 'RCP2.6')
            end = xt.split('-')[1]
            xtl_new.append(begin+'-'+end)
        elif 'RCP45' in xt:
            begin = xt.split('-')[0].replace('RCP45', 'RCP4.5')
            end = xt.split('-')[1]
            xtl_new.append(begin+'-'+end)
        elif 'RCP85' in xt:
            begin = xt.split('-')[0].replace('RCP85', 'RCP8.5')
            end = xt.split('-')[1]
            xtl_new.append(begin+'-'+end)
        else:
            xtl_new.append(xt)
    
    ax.set_xticklabels(xtl_new, rotation=90)
    
    fname = os.path.join(output_folder_mry,
                         '{} {}.png'.format(converter_condition[condition], struc_orient))
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


    
    

            



    
        
        
        

    
            
            









