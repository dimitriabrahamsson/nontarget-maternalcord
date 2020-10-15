#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:41:04 2020

@author: dabrahamsson
"""

import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('R01_posneg_database_search.csv')
df = df.fillna(0)
df = df.drop('Unnamed: 0', axis=1)
df['iso_id_esi'] = df['iso_id_esi'].str.replace(' ', '')
df.head()

df1 = pd.read_csv('level2.csv')
df['MSMSconfirmed'] = np.where(df['Formula_s1'].isin(df1['Formula']), 1, 0)
df['MSMSconfirmed'].sum()

df['Adduct'] = np.where(((df['Sodium_Adduct'] == 1)|(df['Potasium_Adduct'] == 1)|
                        (df['Ammonium_Adduct'] == 1)|(df['ACN_Adduct'] == 1)|
                        (df['Formate_Adduct'] == 1)|(df['H2O_Adduct'] == 1)|
                        (df['CO2_Adduct'] == 1)), 1, 0)

df['Adduct'].sum()

df['Adduct_filter'] = np.where(((df['Adduct']==1) & (df['MSMSconfirmed']==0)), 1, 0) 
df['Adduct_filter'].sum()

df = df[(df['Sodium_Adduct'] == 0) & (df['Adduct_filter']== 0)]
df = df[(df['Potasium_Adduct'] == 0) & (df['Adduct_filter'] == 0)]
df = df[(df['Ammonium_Adduct'] == 0) & (df['Adduct_filter'] == 0)]
df = df[(df['ACN_Adduct'] == 0) & (df['Adduct_filter'] == 0)]
df = df[(df['Formate_Adduct'] == 0) & (df['Adduct_filter'] == 0)]
df = df[(df['H2O_Adduct'] == 0) & (df['Adduct_filter'] == 0)]
df = df[(df['CO2_Adduct'] == 0) & (df['Adduct_filter'] == 0)]
df = df[df['pos_neg_dup'] == 0]

df.to_csv('R01_posneg_filtered.csv')

#df = df.set_index('iso_id_esi')

dfH = df.loc[:,'M331':'845C']
dfH = np.log10(dfH)
dfHM = dfH.loc[:, dfH.columns.str.contains('M')]
dfHC = dfH.loc[:, dfH.columns.str.contains('C')]

dfH['meanlogA'] = dfH.mean(axis=1)
dfH['meanMA'] = dfHM.mean(axis=1)
dfH['stdMA'] = dfHM.std(axis=1)
dfH['meanCA'] = dfHC.mean(axis=1)
dfH['stdCA'] = dfHC.std(axis=1)

dfH['Retention Time'] = df['Retention Time_av']
dfH['Molecular Mass'] = df['Mass_av']
dfH['iso_id_esi'] = df['iso_id_esi']
dfH['esi'] = np.where(dfH['iso_id_esi'].str.contains('pos'), 'pos', 'neg')

x = dfH['Retention Time']
y = dfH['Molecular Mass']
z = dfH['meanlogA']
fig = sns.scatterplot(x, y, size = z, 
                sizes=(20, 200), alpha=0.4, hue=dfH['esi'])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.xlabel('Retention Time (min)')
plt.ylabel('Molecular Mass (g/mol)')
fig.figure.savefig('bubble_plot.png', dpi=400, bbox_inches='tight')
plt.show()

x = dfH['meanMA']
y = dfH['meanCA']
xerror = dfH['stdMA']
yerror = dfH['stdCA']

sns.set(font_scale=1)
sns.set_style('white')
fig = sns.jointplot('meanMA','meanCA', data=dfH, color='darkcyan', kind='reg')
fig.annotate(stats.pearsonr)
plt.xlabel('mean logA maternal')
plt.ylabel('mean logA cord')
plt.errorbar(x, y, yerr=yerror, fmt=' ', barsabove=False, ecolor='darkcyan', elinewidth=0.3)
plt.show()
fig.savefig('meanlogMAMCyerrors.png', dpi=400)

plt.errorbar(x, y, yerr=yerror, fmt='o', barsabove=False)
plt.show()


# Frequency calculations

dfH = df.loc[:,'M331':'845C']
dfH = np.log10(dfH)
dfHM = dfH.loc[:, dfH.columns.str.contains('M')]
dfHC = dfH.loc[:, dfH.columns.str.contains('C')]

dfHF = dfH
dfHF[dfHF < np.log10(5000)] = np.nan
dfHMF = dfHF.loc[:, dfHF.columns.str.contains('M')]
dfHCF = dfHF.loc[:, dfHF.columns.str.contains('C')]

dfHMF['freq_m'] =  dfHMF.count(axis='columns')
dfHMF['freq(%)_m'] = dfHMF['freq_m']/dfHMF['freq_m'].max()*100

dfHCF['freq_c'] =  dfHCF.count(axis='columns')
dfHCF['freq(%)_c'] = dfHCF['freq_c']/dfHCF['freq_c'].max()*100

dfHMF.to_csv('dfHMFrequency.csv')
dfHCF.to_csv('dfHCFrequency.csv')

sns.set(font_scale=1)
sns.set_style('white')
fig = sns.distplot(dfHMF['freq(%)_m'], color='royalblue')
plt.ylabel('Kernel density estimate')
plt.xlabel('Frequency (%)')
plt.show()
fig.figure.savefig('freqM.png', dpi=400)

sns.set(font_scale=1)
sns.set_style('white')
fig = sns.distplot(dfHCF['freq(%)_c'], color='royalblue')
plt.ylabel('Kernel density estimate')
plt.xlabel('Frequency (%)')
plt.show()
fig.figure.savefig('freqC.png', dpi=400)

dfHMF['Met_Presence'] = df['Met_Presence']
dfHMF.to_csv('dfHMFrequency.csv')

dfHMFendo = dfHMF.loc[dfHMF['Met_Presence'] == 1] 
dfHMFexo = dfHMF.loc[dfHMF['Met_Presence'] == 0] 

fig = sns.distplot(dfHMFendo['freq(%)_m'] , color="red", kde=False)
plt.ylabel('N Features')
plt.xlabel('Frequency (%)')
plt.show()
fig.figure.savefig('freqM_endo.png', dpi=400)

fig = sns.distplot(dfHMFexo['freq(%)_m'], color="gray", kde=False)
plt.ylabel('N Features')
plt.xlabel('Frequency (%)')
plt.show()
fig.figure.savefig('freqM_exo.png', dpi=400)


dfHCF['Met_Presence'] = df['Met_Presence']
dfHCF.to_csv('dfHCFrequency.csv')

dfHCFendo = dfHCF.loc[dfHCF['Met_Presence'] == 1] 
dfHCFexo = dfHCF.loc[dfHCF['Met_Presence'] == 0] 

fig = sns.distplot(dfHCFendo['freq(%)_c'] , color="red", kde=False)
plt.ylabel('N Features')
plt.xlabel('Frequency (%)')
plt.show()
fig.figure.savefig('freqC_endo.png', dpi=400)

fig = sns.distplot(dfHCFexo['freq(%)_c'], color="gray", kde=False)
plt.ylabel('N Features')
plt.xlabel('Frequency (%)')
plt.show()
fig.figure.savefig('freqC_exo.png', dpi=400)

# Frequency calculations per sample

dfH = df.loc[:,'M331':'845C']
dfH = np.log10(dfH)
dfHM = dfH.loc[:, dfH.columns.str.contains('M')]
dfHC = dfH.loc[:, dfH.columns.str.contains('C')]

dfHF = dfH.T
dfHF[dfHF < np.log10(5000)] = np.nan

dfHF['freq'] =  dfHF.count(axis='columns')
dfHF['freq(%)'] = dfHF['freq']/dfHF['freq'].max()*100

dfHF.to_csv('dfHFrequency_sample.csv')



# Save a copy of the complete dataset before melting
df_complete = df

ids = df.columns.values.tolist()
ids = ids[1:589]
ids
df = pd.melt(df, id_vars=['iso_id_esi', 'Formula_s1', 'Mass_av', 'Retention Time_av'], value_vars=ids, value_name='Abundance')

df.columns = df.columns.str.replace('variable', 'enrollment_id_s')
df['enrollment_id'] = df['enrollment_id_s']
df['enrollment_id'] = df['enrollment_id'].str.replace('M', '')
df['enrollment_id'] = df['enrollment_id'].str.replace('C', '')

df['logA'] = np.log10(df['Abundance'])
df['MvsC'] = np.where(df['enrollment_id_s'].str.contains('M'), 1, 0)

# Save a copy of the melted df with both M and C samples
dfMC = df

# Finding chemicals that are differentially expressed in maternal and cord samples
loc1 = df.loc[:,'logA']
loc2 = df.loc[:,'iso_id_esi']
loc3 = df.loc[:,'enrollment_id_s']
loc4 = df.loc[:,'MvsC']
dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)

import scipy
from scipy import stats
from scipy.stats import linregress

axisvalues = dft.columns.values

def get_element(my_list, position):
    return my_list[position]


dft['MvsC'] = dft['MvsC'].astype(str)
dft['enrollment_id_s'] = dft['enrollment_id_s'].astype(str)
dft['identifier'] = dft['MvsC'] + '_' + dft['enrollment_id_s']
dft = dft.drop(['MvsC','enrollment_id_s'], axis=1)

dft = dft.pivot_table('logA','iso_id_esi','identifier')

axisvalues = dft.columns.values
axisvalues_df = pd.DataFrame({'identifier': axisvalues})
axisvalues_ = axisvalues_df.identifier.str.split('_').apply(lambda x: x[0])
axisvalues_ = axisvalues_.astype(float)

dft = dft.astype(float)
def calc_slope(row):
    a = scipy.stats.linregress(axisvalues_ , row)
    return pd.Series(a._asdict())

print (dft.apply(calc_slope,axis=1))

dft = dft.join(dft.apply(calc_slope,axis=1))
dft

#negative slope
dft['neg_slope'] = np.where(dft['slope'] < 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ns = dft.loc[(dft['neg_slope']=='1') & (dft['sig_pvalue']=='1')]

#positive slope
dft['pos_slope'] = np.where(dft['slope'] > 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ps = dft.loc[(dft['pos_slope']=='1') & (dft['sig_pvalue']=='1')]

dft_ns = dft_ns.reset_index(drop=False)
dft_ns = dft_ns.loc[:, 'iso_id_esi':'slope']
dft_ns = dft_ns.drop('slope', axis=1)
dft_ns = dft_ns.set_index('iso_id_esi')

dft_ps = dft_ps.reset_index(drop=False)
dft_ps = dft_ps.loc[:, 'iso_id_esi':'slope']
dft_ps = dft_ps.drop('slope', axis=1)
dft_ps = dft_ps.set_index('iso_id_esi')

htm = sns.heatmap(dft_ns, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

htm = sns.heatmap(dft_ps, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()


#Benjamini-Hochberg filtering p-values
dft = dft.sort_values(by='pvalue')
dft['rank'] = dft.reset_index().index + 1
dft['(I/m)Q'] = (dft['rank']/len(dft))*0.05
dft['(I/m)Q - p'] = dft['(I/m)Q'] - dft['pvalue']
dft['BH_sig'] = np.where(dft['(I/m)Q - p'] < 0, '0', '1')

dft_ns = dft_ns.reset_index()
dft_ps = dft_ps.reset_index()
dft = dft.reset_index()

dft_bh_1 = dft.loc[:,'iso_id_esi']
dft_bh_2 = dft.loc[:,'BH_sig']
dft_bh = pd.concat([dft_bh_1, dft_bh_2], axis=1)
dft_bh.columns = dft_bh.columns.str.replace('BH_sig','BH_sig_mvc')

dft_ns_id = dft_ns.loc[:, 'iso_id_esi':'0_114C']
dft_ns_id.columns = dft_ns_id.columns.str.replace('0_114C', 'sig_mvc_neg')
dft_ns_id['sig_mvc_neg'] = dft_ns_id['sig_mvc_neg'].notnull().astype(int)
df_s = pd.merge(df_complete, dft_ns_id, on='iso_id_esi', how='left')

dft_ps_id = dft_ps.loc[:, 'iso_id_esi':'0_114C']
dft_ps_id.columns = dft_ps_id.columns.str.replace('0_114C', 'sig_mvc_pos')
dft_ps_id['sig_mvc_pos'] = dft_ps_id['sig_mvc_pos'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ps_id, on='iso_id_esi', how='left')

df_s = pd.merge(df_s, dft_bh, on='iso_id_esi', how='left')
print(df_s)
print(df_s.columns.values)

df_s['BH_sig_mvc'].astype(float).sum()

# Finding chemicals that are differentially expressed with Gestational Age
# Gestational Age Numeric
# Maternal samples

df = df.loc[df['enrollment_id_s'].str.contains('M')]

df_m = pd.read_csv('medicalrecabstraction.csv', sep=',')
df_g = pd.read_csv('gestational_age_imputed_v2.csv')
df_g['ga_days_mr_imputed'] = df_g['ga_tdays_mr'].round(0)
df_g = df_g.loc[df_g['ga_days_mr_imputed'] == df_g['ga_days_mr_imputed']]
df_g['enrollment_id'] = df_g['ppt_id'].astype(str)
df_g = df_g.loc[:, ['enrollment_id', 'ga_days_mr_imputed']]

df = pd.merge(df, df_g, on='enrollment_id')

loc1 = df.loc[:,'logA']
loc2 = df.loc[:,'iso_id_esi']
loc3 = df.loc[:,'enrollment_id']
loc4 = df.loc[:,'ga_days_mr_imputed']
dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)

dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)
dft['ga_days_mr_imputed'] = dft['ga_days_mr_imputed'].astype(str)
dft['enrollment_id'] = dft['enrollment_id'].astype(str)
dft['ga_days_id'] = dft['ga_days_mr_imputed'] + '_' + dft['enrollment_id']
dft = dft.drop(['ga_days_mr_imputed','enrollment_id'], axis=1)

dft = dft.pivot_table('logA','iso_id_esi','ga_days_id')

axisvalues = dft.columns.values
axisvalues_df = pd.DataFrame({'ga_days_id': axisvalues})
axisvalues_ = axisvalues_df.ga_days_id.str.split('_').apply(lambda x: x[0])
axisvalues_ = axisvalues_.astype(float)

dft = dft.astype(float)
print (dft.apply(calc_slope,axis=1))
dft = dft.join(dft.apply(calc_slope,axis=1))
dft.columns.values

#negative slope
dft['neg_slope'] = np.where(dft['slope'] < 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ns = dft.loc[(dft['neg_slope']=='1') & (dft['sig_pvalue']=='1')]

#positive slope
dft['pos_slope'] = np.where(dft['slope'] > 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ps = dft.loc[(dft['pos_slope']=='1') & (dft['sig_pvalue']=='1')]

dft_ns = dft_ns.reset_index(drop=False)
dft_ns = dft_ns.loc[:, 'iso_id_esi':'slope']
dft_ns = dft_ns.drop('slope', axis=1)
dft_ns = dft_ns.set_index('iso_id_esi')

dft_ps = dft_ps.reset_index(drop=False)
dft_ps = dft_ps.loc[:, 'iso_id_esi':'slope']
dft_ps = dft_ps.drop('slope', axis=1)
dft_ps = dft_ps.set_index('iso_id_esi')

htm = sns.heatmap(dft_ns, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

htm = sns.heatmap(dft_ps, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

dft_ds = pd.concat([dft_ns, dft_ps], axis=0)
ctm = sns.clustermap(dft_ds, cmap='coolwarm', col_cluster = False)
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

#Benjamini-Hochberg filtering p-values
dft = dft.sort_values(by='pvalue')
dft['rank'] = dft.reset_index().index + 1
dft['(I/m)Q'] = (dft['rank']/len(dft))*0.05
dft['(I/m)Q - p'] = dft['(I/m)Q'] - dft['pvalue']
dft['BH_sig'] = np.where(dft['(I/m)Q - p'] < 0, '0', '1')

dft_ns = dft_ns.reset_index()
dft_ps = dft_ps.reset_index()
dft_ns.columns.values
dft = dft.reset_index()

dft_bh_1 = dft.loc[:,'iso_id_esi']
dft_bh_2 = dft.loc[:,'BH_sig']
dft_bh = pd.concat([dft_bh_1, dft_bh_2], axis=1)
dft_bh.columns = dft_bh.columns.str.replace('BH_sig','BH_sig_gestM')

dft_ns_id = dft_ns.iloc[:, 0:2]
dft_ns_id.columns = ('iso_id_esi','sig_gestM_neg')
dft_ns_id['sig_gestM_neg'] = dft_ns_id['sig_gestM_neg'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ns_id, on='iso_id_esi', how='left')

dft_ps_id = dft_ps.iloc[:, 0:2]
dft_ps_id.columns = ('iso_id_esi','sig_gestM_pos')
dft_ps_id['sig_gestM_pos'] = dft_ps_id['sig_gestM_pos'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ps_id, on='iso_id_esi', how='left')

df_s = pd.merge(df_s, dft_bh, on='iso_id_esi', how='left')
print(df_s)
print(df_s.columns.values)
print(df_s['BH_sig_gestM'].astype(float).sum())



# Finding chemicals that are differentially expressed with Gestational Age
# Gestational Age Categorical
# Maternal samples

df['enrollment_id_s']

df['ga_cat_1'] = np.where(df['ga_days_mr_imputed'] < 260, '1', '')
df['ga_cat_2'] = np.where(((df['ga_days_mr_imputed'] >= 260) & (df['ga_days_mr_imputed'] < 273)), '2', '')
df['ga_cat_3'] = np.where(((df['ga_days_mr_imputed'] >= 273) & (df['ga_days_mr_imputed'] < 287)), '3', '')
df['ga_cat_4'] = np.where(df['ga_days_mr_imputed'] >= 287, '4', '')

df['ga_cat'] = df['ga_cat_1']+df['ga_cat_2']+df['ga_cat_3']+df['ga_cat_4']
df = df.drop(['ga_cat_1', 'ga_cat_2', 'ga_cat_3', 'ga_cat_4'], axis=1)

loc1 = df.loc[:,'logA']
loc2 = df.loc[:,'iso_id_esi']
loc3 = df.loc[:,'enrollment_id']
loc4 = df.loc[:,'ga_cat']
dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)

dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)
dft['ga_cat'] = dft['ga_cat'].astype(str)
dft['enrollment_id'] = dft['enrollment_id'].astype(str)
dft['ga_days_id'] = dft['ga_cat'] + '_' + dft['enrollment_id']
dft = dft.drop(['ga_cat','enrollment_id'], axis=1)

dft = dft.pivot_table('logA','iso_id_esi','ga_days_id')

axisvalues = dft.columns.values
axisvalues_df = pd.DataFrame({'ga_days_id': axisvalues})
axisvalues_ = axisvalues_df.ga_days_id.str.split('_').apply(lambda x: x[0])
axisvalues_ = axisvalues_.astype(float)

dft = dft.astype(float)
print (dft.apply(calc_slope,axis=1))
dft = dft.join(dft.apply(calc_slope,axis=1))
dft.columns.values

#negative slope
dft['neg_slope'] = np.where(dft['slope'] < 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ns = dft.loc[(dft['neg_slope']=='1') & (dft['sig_pvalue']=='1')]

#positive slope
dft['pos_slope'] = np.where(dft['slope'] > 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ps = dft.loc[(dft['pos_slope']=='1') & (dft['sig_pvalue']=='1')]

dft_ns = dft_ns.reset_index(drop=False)
dft_ns = dft_ns.loc[:, 'iso_id_esi':'slope']
dft_ns = dft_ns.drop('slope', axis=1)
dft_ns = dft_ns.set_index('iso_id_esi')

dft_ps = dft_ps.reset_index(drop=False)
dft_ps = dft_ps.loc[:, 'iso_id_esi':'slope']
dft_ps = dft_ps.drop('slope', axis=1)
dft_ps = dft_ps.set_index('iso_id_esi')

htm = sns.heatmap(dft_ns, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

htm = sns.heatmap(dft_ps, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

dft_ds = pd.concat([dft_ns, dft_ps], axis=0)
ctm = sns.clustermap(dft_ds, cmap='coolwarm', col_cluster = False)
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

#Benjamini-Hochberg filtering p-values
dft = dft.sort_values(by='pvalue')
dft['rank'] = dft.reset_index().index + 1
dft['(I/m)Q'] = (dft['rank']/len(dft))*0.05
dft['(I/m)Q - p'] = dft['(I/m)Q'] - dft['pvalue']
dft['BH_sig'] = np.where(dft['(I/m)Q - p'] < 0, '0', '1')

dft_ns = dft_ns.reset_index()
dft_ps = dft_ps.reset_index()
dft_ns.columns.values
dft = dft.reset_index()

dft_bh_1 = dft.loc[:,'iso_id_esi']
dft_bh_2 = dft.loc[:,'BH_sig']
dft_bh = pd.concat([dft_bh_1, dft_bh_2], axis=1)
dft_bh.columns = dft_bh.columns.str.replace('BH_sig','BH_sig_gestcatM')


dft_ns_id = dft_ns.iloc[:, 0:2]
dft_ns_id.columns = ('iso_id_esi','sig_gestcatM_neg')
dft_ns_id['sig_gestcatM_neg'] = dft_ns_id['sig_gestcatM_neg'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ns_id, on='iso_id_esi', how='left')

dft_ps_id = dft_ps.iloc[:, 0:2]
dft_ps_id.columns = ('iso_id_esi','sig_gestcatM_pos')
dft_ps_id['sig_gestcatM_pos'] = dft_ps_id['sig_gestcatM_pos'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ps_id, on='iso_id_esi', how='left')

df_s = pd.merge(df_s, dft_bh, on='iso_id_esi', how='left')
print(df_s)
print(df_s.columns.values)
print(df_s['BH_sig_gestcatM'].astype(float).sum())


# Finding chemicals that are differentially expressed with Gestational Age
# Gestational Age Categorical 2 categories pre/early term vs full/post term
# Maternal samples

df['ga_cat_1'] = np.where(df['ga_days_mr_imputed'] < 260, '1', '')
df['ga_cat_2'] = np.where(df['ga_days_mr_imputed'] >= 260, '2', '')

df['ga_catV2'] = df['ga_cat_1']+df['ga_cat_2']
df = df.drop(['ga_cat_1', 'ga_cat_2'], axis=1)

loc1 = df.loc[:,'logA']
loc2 = df.loc[:,'iso_id_esi']
loc3 = df.loc[:,'enrollment_id']
loc4 = df.loc[:,'ga_catV2']
dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)

dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)
dft['ga_catV2'] = dft['ga_catV2'].astype(str)
dft['enrollment_id'] = dft['enrollment_id'].astype(str)
dft['ga_days_id'] = dft['ga_catV2'] + '_' + dft['enrollment_id']
dft = dft.drop(['ga_catV2','enrollment_id'], axis=1)

dft = dft.pivot_table('logA','iso_id_esi','ga_days_id')

axisvalues = dft.columns.values
axisvalues_df = pd.DataFrame({'ga_days_id': axisvalues})
axisvalues_ = axisvalues_df.ga_days_id.str.split('_').apply(lambda x: x[0])
axisvalues_ = axisvalues_.astype(float)

dft = dft.astype(float)
print (dft.apply(calc_slope,axis=1))
dft = dft.join(dft.apply(calc_slope,axis=1))
dft.columns.values

#negative slope
dft['neg_slope'] = np.where(dft['slope'] < 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ns = dft.loc[(dft['neg_slope']=='1') & (dft['sig_pvalue']=='1')]

#positive slope
dft['pos_slope'] = np.where(dft['slope'] > 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ps = dft.loc[(dft['pos_slope']=='1') & (dft['sig_pvalue']=='1')]

dft_ns = dft_ns.reset_index(drop=False)
dft_ns = dft_ns.loc[:, 'iso_id_esi':'slope']
dft_ns = dft_ns.drop('slope', axis=1)
dft_ns = dft_ns.set_index('iso_id_esi')

dft_ps = dft_ps.reset_index(drop=False)
dft_ps = dft_ps.loc[:, 'iso_id_esi':'slope']
dft_ps = dft_ps.drop('slope', axis=1)
dft_ps = dft_ps.set_index('iso_id_esi')

htm = sns.heatmap(dft_ns, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

htm = sns.heatmap(dft_ps, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

dft_ds = pd.concat([dft_ns, dft_ps], axis=0)
ctm = sns.clustermap(dft_ds, cmap='coolwarm', col_cluster = False)
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

#Benjamini-Hochberg filtering p-values
dft = dft.sort_values(by='pvalue')
dft['rank'] = dft.reset_index().index + 1
dft['(I/m)Q'] = (dft['rank']/len(dft))*0.05
dft['(I/m)Q - p'] = dft['(I/m)Q'] - dft['pvalue']
dft['BH_sig'] = np.where(dft['(I/m)Q - p'] < 0, '0', '1')

dft_ns = dft_ns.reset_index()
dft_ps = dft_ps.reset_index()
dft_ns.columns.values
dft = dft.reset_index()

dft_bh_1 = dft.loc[:,'iso_id_esi']
dft_bh_2 = dft.loc[:,'BH_sig']
dft_bh = pd.concat([dft_bh_1, dft_bh_2], axis=1)
dft_bh.columns = dft_bh.columns.str.replace('BH_sig','BH_sig_gestcatV2M')

dft_ns_id = dft_ns.iloc[:, 0:2]
dft_ns_id.columns = ('iso_id_esi','sig_gestcatV2M_neg')
dft_ns_id['sig_gestcatV2M_neg'] = dft_ns_id['sig_gestcatV2M_neg'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ns_id, on='iso_id_esi', how='left')

dft_ps_id = dft_ps.iloc[:, 0:2]
dft_ps_id.columns = ('iso_id_esi','sig_gestcatV2M_pos')
dft_ps_id['sig_gestcatV2M_pos'] = dft_ps_id['sig_gestcatV2M_pos'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ps_id, on='iso_id_esi', how='left')

df_s = pd.merge(df_s, dft_bh, on='iso_id_esi', how='left')
print(df_s)
print(df_s.columns.values)
print(df_s['BH_sig_gestcatV2M'].astype(float).sum())


# Finding chemicals that are differentially expressed with Gestational Age
# Gestational Age Numeric
# Cord samples

df = dfMC.loc[dfMC['enrollment_id_s'].str.contains('C')]
df = pd.merge(df, df_g, on='enrollment_id')

loc1 = df.loc[:,'logA']
loc2 = df.loc[:,'iso_id_esi']
loc3 = df.loc[:,'enrollment_id']
loc4 = df.loc[:,'ga_days_mr_imputed']
dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)

dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)
dft['ga_days_mr_imputed'] = dft['ga_days_mr_imputed'].astype(str)
dft['enrollment_id'] = dft['enrollment_id'].astype(str)
dft['ga_days_id'] = dft['ga_days_mr_imputed'] + '_' + dft['enrollment_id']
dft = dft.drop(['ga_days_mr_imputed','enrollment_id'], axis=1)

dft = dft.pivot_table('logA','iso_id_esi','ga_days_id')

axisvalues = dft.columns.values
axisvalues_df = pd.DataFrame({'ga_days_id': axisvalues})
axisvalues_ = axisvalues_df.ga_days_id.str.split('_').apply(lambda x: x[0])
axisvalues_ = axisvalues_.astype(float)

dft = dft.astype(float)
print (dft.apply(calc_slope,axis=1))
dft = dft.join(dft.apply(calc_slope,axis=1))
dft.columns.values

#negative slope
dft['neg_slope'] = np.where(dft['slope'] < 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ns = dft.loc[(dft['neg_slope']=='1') & (dft['sig_pvalue']=='1')]

#positive slope
dft['pos_slope'] = np.where(dft['slope'] > 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ps = dft.loc[(dft['pos_slope']=='1') & (dft['sig_pvalue']=='1')]

dft_ns = dft_ns.reset_index(drop=False)
dft_ns = dft_ns.loc[:, 'iso_id_esi':'slope']
dft_ns = dft_ns.drop('slope', axis=1)
dft_ns = dft_ns.set_index('iso_id_esi')

dft_ps = dft_ps.reset_index(drop=False)
dft_ps = dft_ps.loc[:, 'iso_id_esi':'slope']
dft_ps = dft_ps.drop('slope', axis=1)
dft_ps = dft_ps.set_index('iso_id_esi')

htm = sns.heatmap(dft_ns, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

htm = sns.heatmap(dft_ps, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

dft_ds = pd.concat([dft_ns, dft_ps], axis=0)
ctm = sns.clustermap(dft_ds, cmap='coolwarm', col_cluster = False)
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

#Benjamini-Hochberg filtering p-values
dft = dft.sort_values(by='pvalue')
dft['rank'] = dft.reset_index().index + 1
dft['(I/m)Q'] = (dft['rank']/len(dft))*0.05
dft['(I/m)Q - p'] = dft['(I/m)Q'] - dft['pvalue']
dft['BH_sig'] = np.where(dft['(I/m)Q - p'] < 0, '0', '1')

dft_ns = dft_ns.reset_index()
dft_ps = dft_ps.reset_index()
dft_ns.columns.values
dft = dft.reset_index()

dft_bh_1 = dft.loc[:,'iso_id_esi']
dft_bh_2 = dft.loc[:,'BH_sig']
dft_bh = pd.concat([dft_bh_1, dft_bh_2], axis=1)
dft_bh.columns = dft_bh.columns.str.replace('BH_sig','BH_sig_gestC')

dft_ns_id = dft_ns.iloc[:, 0:2]
dft_ns_id.columns = ('iso_id_esi','sig_gestC_neg')
dft_ns_id['sig_gestC_neg'] = dft_ns_id['sig_gestC_neg'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ns_id, on='iso_id_esi', how='left')

dft_ps_id = dft_ps.iloc[:, 0:2]
dft_ps_id.columns = ('iso_id_esi','sig_gestC_pos')
dft_ps_id['sig_gestC_pos'] = dft_ps_id['sig_gestC_pos'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ps_id, on='iso_id_esi', how='left')

df_s = pd.merge(df_s, dft_bh, on='iso_id_esi', how='left')
print(df_s)
print(df_s.columns.values)
print(df_s['BH_sig_gestC'].astype(float).sum())



# Finding chemicals that are differentially expressed with Gestational Age
# Gestational Age Categorical
# Cord samples

df['enrollment_id_s']

df['ga_cat_1'] = np.where(df['ga_days_mr_imputed'] < 260, '1', '')
df['ga_cat_2'] = np.where(((df['ga_days_mr_imputed'] >= 260) & (df['ga_days_mr_imputed'] < 273)), '2', '')
df['ga_cat_3'] = np.where(((df['ga_days_mr_imputed'] >= 273) & (df['ga_days_mr_imputed'] < 287)), '3', '')
df['ga_cat_4'] = np.where(df['ga_days_mr_imputed'] >= 287, '4', '')

df['ga_cat'] = df['ga_cat_1']+df['ga_cat_2']+df['ga_cat_3']+df['ga_cat_4']
df = df.drop(['ga_cat_1', 'ga_cat_2', 'ga_cat_3', 'ga_cat_4'], axis=1)

loc1 = df.loc[:,'logA']
loc2 = df.loc[:,'iso_id_esi']
loc3 = df.loc[:,'enrollment_id']
loc4 = df.loc[:,'ga_cat']
dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)

dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)
dft['ga_cat'] = dft['ga_cat'].astype(str)
dft['enrollment_id'] = dft['enrollment_id'].astype(str)
dft['ga_days_id'] = dft['ga_cat'] + '_' + dft['enrollment_id']
dft = dft.drop(['ga_cat','enrollment_id'], axis=1)

dft = dft.pivot_table('logA','iso_id_esi','ga_days_id')

axisvalues = dft.columns.values
axisvalues_df = pd.DataFrame({'ga_days_id': axisvalues})
axisvalues_ = axisvalues_df.ga_days_id.str.split('_').apply(lambda x: x[0])
axisvalues_ = axisvalues_.astype(float)

dft = dft.astype(float)
print (dft.apply(calc_slope,axis=1))
dft = dft.join(dft.apply(calc_slope,axis=1))
dft.columns.values


#negative slope
dft['neg_slope'] = np.where(dft['slope'] < 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ns = dft.loc[(dft['neg_slope']=='1') & (dft['sig_pvalue']=='1')]

#positive slope
dft['pos_slope'] = np.where(dft['slope'] > 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ps = dft.loc[(dft['pos_slope']=='1') & (dft['sig_pvalue']=='1')]

dft_ns = dft_ns.reset_index(drop=False)
dft_ns = dft_ns.loc[:, 'iso_id_esi':'slope']
dft_ns = dft_ns.drop('slope', axis=1)
dft_ns = dft_ns.set_index('iso_id_esi')

dft_ps = dft_ps.reset_index(drop=False)
dft_ps = dft_ps.loc[:, 'iso_id_esi':'slope']
dft_ps = dft_ps.drop('slope', axis=1)
dft_ps = dft_ps.set_index('iso_id_esi')

htm = sns.heatmap(dft_ns, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

htm = sns.heatmap(dft_ps, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

dft_ds = pd.concat([dft_ns, dft_ps], axis=0)
ctm = sns.clustermap(dft_ds, cmap='coolwarm', col_cluster = False)
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

#Benjamini-Hochberg filtering p-values
dft = dft.sort_values(by='pvalue')
dft['rank'] = dft.reset_index().index + 1
dft['(I/m)Q'] = (dft['rank']/len(dft))*0.05
dft['(I/m)Q - p'] = dft['(I/m)Q'] - dft['pvalue']
dft['BH_sig'] = np.where(dft['(I/m)Q - p'] < 0, '0', '1')

dft_ns = dft_ns.reset_index()
dft_ps = dft_ps.reset_index()
dft_ns.columns.values
dft = dft.reset_index()

dft_bh_1 = dft.loc[:,'iso_id_esi']
dft_bh_2 = dft.loc[:,'BH_sig']
dft_bh = pd.concat([dft_bh_1, dft_bh_2], axis=1)
dft_bh.columns = dft_bh.columns.str.replace('BH_sig','BH_sig_gestcatC')

dft_ns_id = dft_ns.iloc[:, 0:2]
dft_ns_id.columns = ('iso_id_esi','sig_gestcatC_neg')
dft_ns_id['sig_gestcatC_neg'] = dft_ns_id['sig_gestcatC_neg'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ns_id, on='iso_id_esi', how='left')

dft_ps_id = dft_ps.iloc[:, 0:2]
dft_ps_id.columns = ('iso_id_esi','sig_gestcatC_pos')
dft_ps_id['sig_gestcatC_pos'] = dft_ps_id['sig_gestcatC_pos'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ps_id, on='iso_id_esi', how='left')

df_s = pd.merge(df_s, dft_bh, on='iso_id_esi', how='left')
print(df_s)
print(df_s.columns.values)
print(df_s['BH_sig_gestcatC'].astype(float).sum())

# Finding chemicals that are differentially expressed with Gestational Age
# Gestational Age Categorical 2 categories pre/early term vs full/post term
# Cord samples

df['ga_cat_1'] = np.where(df['ga_days_mr_imputed'] < 260, '1', '')
df['ga_cat_2'] = np.where(df['ga_days_mr_imputed'] >= 260, '2', '')

df['ga_catV2'] = df['ga_cat_1']+df['ga_cat_2']
df = df.drop(['ga_cat_1', 'ga_cat_2'], axis=1)

loc1 = df.loc[:,'logA']
loc2 = df.loc[:,'iso_id_esi']
loc3 = df.loc[:,'enrollment_id']
loc4 = df.loc[:,'ga_catV2']
dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)

dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)
dft['ga_catV2'] = dft['ga_catV2'].astype(str)
dft['enrollment_id'] = dft['enrollment_id'].astype(str)
dft['ga_days_id'] = dft['ga_catV2'] + '_' + dft['enrollment_id']
dft = dft.drop(['ga_catV2','enrollment_id'], axis=1)

dft = dft.pivot_table('logA','iso_id_esi','ga_days_id')

axisvalues = dft.columns.values
axisvalues_df = pd.DataFrame({'ga_days_id': axisvalues})
axisvalues_ = axisvalues_df.ga_days_id.str.split('_').apply(lambda x: x[0])
axisvalues_ = axisvalues_.astype(float)

dft = dft.astype(float)
print (dft.apply(calc_slope,axis=1))
dft = dft.join(dft.apply(calc_slope,axis=1))
dft.columns.values

#negative slope
dft['neg_slope'] = np.where(dft['slope'] < 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ns = dft.loc[(dft['neg_slope']=='1') & (dft['sig_pvalue']=='1')]

#positive slope
dft['pos_slope'] = np.where(dft['slope'] > 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ps = dft.loc[(dft['pos_slope']=='1') & (dft['sig_pvalue']=='1')]

dft_ns = dft_ns.reset_index(drop=False)
dft_ns = dft_ns.loc[:, 'iso_id_esi':'slope']
dft_ns = dft_ns.drop('slope', axis=1)
dft_ns = dft_ns.set_index('iso_id_esi')

dft_ps = dft_ps.reset_index(drop=False)
dft_ps = dft_ps.loc[:, 'iso_id_esi':'slope']
dft_ps = dft_ps.drop('slope', axis=1)
dft_ps = dft_ps.set_index('iso_id_esi')

htm = sns.heatmap(dft_ns, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

htm = sns.heatmap(dft_ps, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

dft_ds = pd.concat([dft_ns, dft_ps], axis=0)
ctm = sns.clustermap(dft_ds, cmap='coolwarm', col_cluster = False)
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

#Benjamini-Hochberg filtering p-values
dft = dft.sort_values(by='pvalue')
dft['rank'] = dft.reset_index().index + 1
dft['(I/m)Q'] = (dft['rank']/len(dft))*0.05
dft['(I/m)Q - p'] = dft['(I/m)Q'] - dft['pvalue']
dft['BH_sig'] = np.where(dft['(I/m)Q - p'] < 0, '0', '1')

dft_ns = dft_ns.reset_index()
dft_ps = dft_ps.reset_index()
dft_ns.columns.values
dft = dft.reset_index()

dft_bh_1 = dft.loc[:,'iso_id_esi']
dft_bh_2 = dft.loc[:,'BH_sig']
dft_bh = pd.concat([dft_bh_1, dft_bh_2], axis=1)
dft_bh.columns = dft_bh.columns.str.replace('BH_sig','BH_sig_gestcatV2C')

dft_ns_id = dft_ns.iloc[:, 0:2]
dft_ns_id.columns = ('iso_id_esi','sig_gestcatV2C_neg')
dft_ns_id['sig_gestcatV2C_neg'] = dft_ns_id['sig_gestcatV2C_neg'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ns_id, on='iso_id_esi', how='left')

dft_ps_id = dft_ps.iloc[:, 0:2]
dft_ps_id.columns = ('iso_id_esi','sig_gestcatV2C_pos')
dft_ps_id['sig_gestcatV2C_pos'] = dft_ps_id['sig_gestcatV2C_pos'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ps_id, on='iso_id_esi', how='left')

df_s = pd.merge(df_s, dft_bh, on='iso_id_esi', how='left')
print(df_s)
print(df_s.columns.values)
print(df_s['BH_sig_gestcatV2C'].astype(float).sum())

# Finding chemicals that are differentially expressed with Gestational Age
# Gestational Age Numeric
# Maternal and Cord samples

df = dfMC
df = pd.merge(df, df_g, on='enrollment_id')

loc1 = df.loc[:,'logA']
loc2 = df.loc[:,'iso_id_esi']
loc3 = df.loc[:,'enrollment_id']
loc4 = df.loc[:,'ga_days_mr_imputed']
dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)

dft = pd.concat([loc1,loc2,loc3,loc4], axis=1)
dft['ga_days_mr_imputed'] = dft['ga_days_mr_imputed'].astype(str)
dft['enrollment_id'] = dft['enrollment_id'].astype(str)
dft['ga_days_id'] = dft['ga_days_mr_imputed'] + '_' + dft['enrollment_id']
dft = dft.drop(['ga_days_mr_imputed','enrollment_id'], axis=1)

dft = dft.pivot_table('logA','iso_id_esi','ga_days_id')

axisvalues = dft.columns.values
axisvalues_df = pd.DataFrame({'ga_days_id': axisvalues})
axisvalues_ = axisvalues_df.ga_days_id.str.split('_').apply(lambda x: x[0])
axisvalues_ = axisvalues_.astype(float)

dft = dft.astype(float)
print (dft.apply(calc_slope,axis=1))
dft = dft.join(dft.apply(calc_slope,axis=1))
dft.columns.values

#negative slope
dft['neg_slope'] = np.where(dft['slope'] < 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ns = dft.loc[(dft['neg_slope']=='1') & (dft['sig_pvalue']=='1')]

#positive slope
dft['pos_slope'] = np.where(dft['slope'] > 0, '1','0')
dft['sig_pvalue'] = np.where(dft['pvalue'] < 0.05, '1','0')
dft_ps = dft.loc[(dft['pos_slope']=='1') & (dft['sig_pvalue']=='1')]

dft_ns = dft_ns.reset_index(drop=False)
dft_ns = dft_ns.loc[:, 'iso_id_esi':'slope']
dft_ns = dft_ns.drop('slope', axis=1)
dft_ns = dft_ns.set_index('iso_id_esi')

dft_ps = dft_ps.reset_index(drop=False)
dft_ps = dft_ps.loc[:, 'iso_id_esi':'slope']
dft_ps = dft_ps.drop('slope', axis=1)
dft_ps = dft_ps.set_index('iso_id_esi')

htm = sns.heatmap(dft_ns, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

htm = sns.heatmap(dft_ps, cmap='coolwarm')
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

dft_ds = pd.concat([dft_ns, dft_ps], axis=0)
ctm = sns.clustermap(dft_ds, cmap='coolwarm', col_cluster = False)
sns.set(font_scale=0.6)
plt.yticks(rotation=0)
plt.show()

#Benjamini-Hochberg filtering p-values
dft = dft.sort_values(by='pvalue')
dft['rank'] = dft.reset_index().index + 1
dft['(I/m)Q'] = (dft['rank']/len(dft))*0.05
dft['(I/m)Q - p'] = dft['(I/m)Q'] - dft['pvalue']
dft['BH_sig'] = np.where(dft['(I/m)Q - p'] < 0, '0', '1')

dft_ns = dft_ns.reset_index()
dft_ps = dft_ps.reset_index()
dft_ns.columns.values
dft = dft.reset_index()

dft_bh_1 = dft.loc[:,'iso_id_esi']
dft_bh_2 = dft.loc[:,'BH_sig']
dft_bh = pd.concat([dft_bh_1, dft_bh_2], axis=1)
dft_bh.columns = dft_bh.columns.str.replace('BH_sig','BH_sig_gestMC')

dft_ns_id = dft_ns.iloc[:, 0:2]
dft_ns_id.columns = ('iso_id_esi','sig_gestMC_neg')
dft_ns_id['sig_gestMC_neg'] = dft_ns_id['sig_gestMC_neg'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ns_id, on='iso_id_esi', how='left')

dft_ps_id = dft_ns.iloc[:, 0:2]
dft_ps_id.columns = ('iso_id_esi','sig_gestMC_pos')
dft_ps_id['sig_gestMC_pos'] = dft_ps_id['sig_gestMC_pos'].notnull().astype(int)
df_s = pd.merge(df_s, dft_ps_id, on='iso_id_esi', how='left')

df_s = pd.merge(df_s, dft_bh, on='iso_id_esi', how='left')
print(df_s)
print(df_s.columns.values)
print(df_s['sig_gestcatV2M_neg'].astype(float).sum())
print(df_s['sig_gestcatV2M_pos'].astype(float).sum())
print(df_s['BH_sig_gestM'].astype(float).sum())

df_s = df_s.fillna(0)
df_s.to_csv('R01_posneg_gest_stats_fdr05.csv')
