# original name in my thesis project
# SCF1_compiling_percentiles.py
# Download files via SCF_preparation.md

import dfgui # dfgui.show(df)
import os
import numpy as np
from numpy.linalg import inv
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import pandas as pd
from pandas import Series, DataFrame
import scipy
import scipy.optimize
from scipy.stats.stats import pearsonr 
import itertools
import time

# Global parameters
start_time = time.time()
# Working Directory and global parameter
data_source = 'C:/Users/libin/Downloads' 
output_folder = 'C:/Users/libin/Downloads/CleanData'
plot_folder = 'C:/Users/libin/Downloads/Plot'
base_year = 1983
collist_mult = ['networth','nonhouses','homeeq','houses','vehic']
collist_stat = ['1st','2nd','3rd','avg']

# Equivalence adjustment
def family_size_divider(x):
    tmp = x['sizefam']
    #if tmp>=12.:
    #    return 2.1
    #dray = [i/10. for i in range(10,22,1)]
    dray = [1.,1.34,1.65,1.97,2.27,2.57,2.87,3.17,3.47,3.77,4.07,4.37,4.67,4.97,5.27,5.57,5.87,6.17,6.47,6.77] # the last one is for age 20 (included) and older. \CEX\final.sas
    if tmp>len(dray):
        tmp = len(dray) 
    return dray[int(tmp)-1]
# ============================================================================= 1. checked.
# CPI.-----------------------------CPI Inflation Multiplier Generation--------- I. Data cleaning.
    
os.chdir('E:/GoogleDrive/python_projects/research/PhDThesisPython/research/11-SCF') # the folder where I store the BLS CPI series.
df = pd.read_csv("CPI-U_1983-2016.csv")
# cleaning raw data
df['Month'] = df['Period'].str.extract('(\d+)',expand=False).astype(int) # extract only digits [1]
dg = df.loc[df['Year']<2017,['Year','Month','Value']]
# choose the average CPI over July to October as the representative CPI of the year
data_source = 'C:/Users/libin/Downloads'
os.chdir(data_source) 
def cpi(x):
    y = x['Value'][1:12].mean() # annual average
    return y
dc = dg.groupby('Year').apply(cpi)
dc = DataFrame(dc,columns=['Value']) # merge [2]
os.chdir(output_folder)
dc.to_csv("cpi.csv", header=True, index=True) # index=True is needd to show column Year.
cpi = pd.read_csv("cpi.csv")
cpi.columns = cpi.columns.str.lower()

def inflate2year2016(xxxx):
    v2016 = cpi.loc[cpi['year']==2016,'value'].values[0]
    vxxxx = cpi.loc[cpi['year']==xxxx,'value'].values[0]
    return (v2016/vxxxx)

# -------------------------SCF 1983-2013 Data Cleaning-------------------------
# 1983 ------------------------------------------------------------------------ II. survey 1983
# see codebook for definition. 
os.chdir(data_source) 
this_year = 1983
shift = this_year - base_year 

collist = ['b1','b4504','b3010','b3324','b3710','b3708','b3902','b3101','b3505','b3201'] # B3004-3005-3010
dg = pd.read_stata("scf83b.dta",columns=collist)
# netequity (in home) refers to the current value of home minus amount outstanding on first and second mortgage.
dg.rename(columns={'b4504':'age','b3010':'weight','b3324':'networth','b3710':'netequity','b3708':'houses','b3902':'vehic','b3101':'sizefam','b3505':'bstruct','b3201':'income'},inplace=True)
conditions = (dg['houses']>=0.) & (dg['weight']>0.) # Variant 1.
dh = dg.loc[conditions, dg.columns]
dh['weight'] = dh['weight']/5.
dh['nonhouses'] = dh['networth'] - dh['houses'] # nonhouses: financial assets, business, durable goods = vehic, financial or housing?
dh = DataFrame(np.where(dh<1.e-66, 0., dh), columns=dh.columns) 
dh.rename(columns={'netequity':'homeeq'}, inplace=True)
dh['year'] = 1983

mult = inflate2year2016(this_year)
for col in collist_mult:
    dh['d{}'.format(col)] = dh[col]*mult

# Age Group
dh = dh.loc[dh['age']>19+shift, dh.columns] # keep only 19 years old and older
bins = np.arange(20+shift, 90+1+shift, 5)
lbin = np.arange(22+shift, 87+1+shift, 5)
dh['agegroup'] = pd.cut(dh['age'], bins, right=False, labels=lbin) # crate age groups [3]
dh = dh.loc[dh['agegroup'].notnull(),dh.columns] # drop entries with HH older than 90
# equivalent-adjustment factor
dh['equfactor'] = dh.apply(family_size_divider,axis=1)

def B3505LegalStructure(x):
    tmp = x['bstruct']
    if (1<=tmp) & (tmp<=2) : # 1: proprietorship, 2: parternship
        return 1 # a business man
    else:
        return 0
    
dh['workcode'] = dh.apply(B3505LegalStructure, axis=1)
dh.drop('b1', axis=1, inplace=True)
os.chdir(output_folder)
dh.to_csv("1983cleaned.csv", header=True, index=False)
print(this_year)

# 1986 ------------------------------------------------------------------------ II. survey 1983
os.chdir(data_source) 
this_year = 1986
shift = this_year - base_year
#df = pd.read_sas("scf86bx.xport", format='xport')
collist = ['c1','c1113','c1013','c1457','c1515','c1512','c1421','c1101','c1810','c1301'] # 1013
dg = pd.read_stata("scf86b.dta",columns=collist)
# C1013: FRB 1986 WEIGHT #1, which is post-straighted to 1986 CPS control totals independently.
# the resulting weight is post-stratified to control totals derived from the Current Population Survey.
dg.rename(columns={'c1113':'age', 'c1013':'weight', 'c1457':'networth', 'c1515':'netequity', 'c1512':'houses', 'c1421':'vehic', 'c1101':'sizefam', 'c1810':'bstruct', 'c1301':'income'},inplace=True)
conditions = (dg['houses']>=0.) & (dg['weight']>0.)
dh = dg.loc[conditions, dg.columns]
dh['weight'] = dh['weight']/5.
dh['nonhouses'] = dh['networth'] - dh['houses'] # durable = vehic, financial or housing?
dh = DataFrame(np.where(dh<1.e-66, 0., dh),columns=dh.columns) 
dh.rename(columns={'netequity':'homeeq'},inplace=True)
dh['year'] = 1986

mult = inflate2year2016(this_year)
for col in collist_mult:
    dh['d{}'.format(col)] = dh[col]*mult

# Age Group
dh = dh.loc[dh['age']>19+shift, dh.columns] # keep only 19 years old and older
bins = np.arange(20+shift, 90+1+shift, 5)
lbin = np.arange(22+shift, 87+1+shift, 5)
dh['agegroup'] = pd.cut(dh['age'], bins, right=False, labels=lbin) # crate age groups
dh = dh.loc[dh['agegroup'].notnull(), dh.columns] # drop entries with HH older than 90
# Adjustment According to Family Size.
dh['equfactor'] = dh.apply(family_size_divider, axis=1)
dh['workcode']  = np.where(dh['bstruct']==6,1,0) # type of employer: 6, self-employed.
dh.drop('c1', axis=1, inplace=True)
os.chdir(output_folder)
dh.to_csv("1986cleaned.csv", header=True, index=False)
print(1986)

#1989 ------------------------------------------------------------------------- II. survey 1989
#df1 = pd.read_sas("scf89x.xport", format='xport') # too large to handle
os.chdir(data_source) 
this_year = 1989
shift = this_year - base_year
dg1 = pd.read_stata("p89i6.dta", columns=['X1','X101'])   
dg1.columns = dg1.columns.str.lower()

#df2 = pd.read_sas("scfp1989.sas7bdat", format='sas7bdat')
collist2 = ['X1','wgt','age','networth','houses','vehic','OCCAT1','homeeq','income']
dg2 = pd.read_stata("rscfp1989.dta", columns=collist2)
dg2.columns = dg2.columns.str.lower()

dg = pd.merge(dg1, dg2, on='x1')

dg.columns = dg.columns.str.lower()
dg.rename(columns={'x101':'sizefam','wgt':'weight'}, inplace=True)
conditions = (dg['houses']>=0.) & (dg['weight']>0.)
dh = dg.loc[conditions, dg.columns]
dh['weight'] = dh['weight']/5.
dh['nonhouses'] = dh['networth'] - dh['houses'] # durable = vehic, financial or housing?
dh = DataFrame(np.where(dh<1.e-66, 0., dh), columns=dh.columns) # []
dh['year'] = 1989

mult = inflate2year2016(this_year)
for col in collist_mult:
    dh['d{}'.format(col)] = dh[col]*mult

# Age Group
dh = dh.loc[dh['age']>19+shift, dh.columns] # keep only 19 years old and older
bins = np.arange(20+shift, 90+1+shift, 5)
lbin = np.arange(22+shift, 87+1+shift, 5)
dh['agegroup'] = pd.cut(dh['age'], bins, right=False, labels=lbin) # crate age groups
dh = dh.loc[dh['agegroup'].notnull(),dh.columns] # drop entries with HH older than 90
# equivalent-adjustment factor
dh['equfactor'] = dh.apply(family_size_divider, axis=1)
dh['workcode'] = np.where(dh['occat1']==2, 1, 0) # 2, self-employed/partnership []
dh.drop('x1', axis=1, inplace=True)
os.chdir(output_folder)
dh.to_csv("1989cleaned.csv", header=True, index=False)
print(1989)

# 1992-2013 ------------------------------------------------------------------- II. surveys 1992-2013
for i in range(1992,2014,3):
    this_year = i
    part_year = str(this_year)[2:] # []
    shift = this_year-base_year

    collist1 = ['Y1','X101'] # X101: number of persons in HHL
    os.chdir(data_source) 
    dg1 = pd.read_stata("p{}i6.dta".format(part_year),columns=collist1)
    dg1.columns = dg1.columns.str.lower()
    
    collist2 = ['Y1','wgt','age','networth','houses','vehic','OCCAT1','homeeq','income']
    dg2 = pd.read_stata("rscfp{}.dta".format(this_year),columns=collist2)
    dg2.columns = dg2.columns.str.lower()
    
    dg = pd.merge(dg1,dg2,on='y1')
    
    dg.rename(columns={'x101':'sizefam','wgt':'weight'},inplace=True)
    conditions = (dg['houses']>=0.) & (dg['weight']>0.)
    dh = dg.loc[conditions, dg.columns]
    dh['weight'] = dh['weight']/5.
    dh['nonhouses'] = dh['networth'] - dh['houses'] # durable = vehic, financial or housing?
    dh = DataFrame(np.where(dh<1.e-66, 0., dh), columns=dh.columns) 
    dh['year'] = this_year
    
    mult = inflate2year2016(this_year)
    for col in collist_mult:
        dh['d{}'.format(col)] = dh[col]*mult    
    
    # Age Group
    dh = dh.loc[dh['age']>19+shift, dh.columns] # keep only 19 years old and older
    bins = np.arange(20+shift, 90+1+shift, 5)
    lbin = np.arange(22+shift, 87+1+shift, 5)
    dh['agegroup'] = pd.cut(dh['age'], bins, right=False, labels=lbin) # crate age groups
    dh = dh.loc[dh['agegroup'].notnull(), dh.columns] # drop entries with HH older than 90
    
    # equivalent-adjustment factor 
    dh['equfactor'] = dh.apply(family_size_divider,axis=1)
    dh['workcode']  = np.where(dh['occat1']==2, 1, 0) # 2, self-employed/partnership
    dh.drop('y1', axis=1, inplace=True)
    os.chdir(output_folder)
    dh.to_csv("{}cleaned.csv".format(this_year), header=True, index=False)
    print(this_year)
#
delta_sec = int(time.time() - start_time)
print("run time: {}".format(delta_sec))

# Compile into an integrated dataset
os.chdir(output_folder) 
df = pd.read_csv("1983cleaned.csv")
for i in range(1986,2014,3):
    dg = pd.read_csv("{}cleaned.csv".format(i))
    df = pd.concat([df,dg])    
df.drop(['bstruct','occat1'], axis=1, inplace=True)
os.chdir(output_folder)
df.to_csv('19832013cleaned.csv',index=False)


# weight inspection. []
grouped = df.groupby('year')
grouped['weight'].sum()

#============================================================================== 2. checked.
#------------------- Summary Statistics Generation ---------------------------- III.1 Summary statistics.

startyear = 1983 # <----------------------------------------------------------- changeable
endyear = 2013 # <------------------------------------------------------------- changeable
yearvec = range(startyear,endyear+1,3)
lbin = np.arange(22, 87+1, 5) # just follows what is defined above.
assetvec = ['dhouses','dnonhouses'] # dnonhouses
assetstr = ['Home Equity','Financial Wealth']
quantvec = ['1qt','2qt','3qt']
quantstr = ['1st quartile','Median','3rd quartile']
quantNvec = [0.25, 0.5, 0.75]
workvec = ['wok','ent','mix']
workstr = ['Worker Households','Business Households','all households']
option_fabricate_values_for_missing_agegroup = False

# Quantile
def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile # shfit to make the first bin weighted zero.
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def weighted_mean(values, sample_weight):
    values = np.array(values)
    sample_weight = np.array(sample_weight)
    inner_product = np.inner(values,sample_weight)
    return inner_product/np.sum(sample_weight)

# Checked! 11-17-2017

## histogram 
#grouped = df.groupby('agegroup')
#size_grouped = grouped.size()
#size_grouped.plot.bar()

def houses_quantile(x,quantile=0.5):
    return weighted_quantile(x['dhouses'],quantile,x['weight'])
def nonhouses_quantile(x,quantile=0.5):
    return weighted_quantile(x['dnonhouses'],quantile,x['weight'])
def houses_mean(x):
    return weighted_mean(x['dhouses'],x['weight'])
def nonhouses_mean(x):
    return weighted_mean(x['dnonhouses'],x['weight'])

#------------------------------------------------------------------------------III.2
os.chdir(output_folder)
for l in workvec: # ['wok','ent','mix']
    for j in assetvec: # ['dhouses','dnonhouses']
        # quantiles block -----------------------------------------------------
        for k in quantNvec: # [0.25, 0.5, 0.75]
            agemat = DataFrame(np.nan, index=yearvec, columns=range(0,len(lbin),1))
            figmat = DataFrame(np.nan, index=yearvec, columns=range(0,len(lbin),1))
            for i in yearvec:   
                df = pd.read_csv("{}cleaned.csv".format(i))
                if workvec.index(l)==1: # entrepreneurs
                    df = df.loc[df['workcode']==1, df.columns]
                elif workvec.index(l)==0: # worker
                    df = df.loc[df['workcode']==0, df.columns]
                else: 
                    print(i)# do nothing.                
                # see final.sas in ..housing\owner\final.sas
                # quantiles are computd per combinations of agegroups and survey years.
                grouped = df[[j,'weight']].groupby(df['agegroup']) # see final.sas in ..housing\owner\final.sas
                num_bins = len(grouped.size().index) # number of valid age groups in a specific year.
                num_dummy = len(lbin) - num_bins
                if j=='dhouses':
                    quantV = grouped.apply(houses_quantile, quantile=k)                
                else:
                    quantV = grouped.apply(nonhouses_quantile, quantile=k)    
                if option_fabricate_values_for_missing_agegroup==False:
                    # don't create fake data point for missing agegroup.
                    fake_value = np.nan
                else:
                    fake_value = 0.                        
                if num_dummy>0:
                    temp = list(quantV.values)
                    temp.extend(list(np.repeat(fake_value,num_dummy))) 
                    figmat.loc[i,:] = temp
                    #                            
                    temp = list(quantV.index)
                    l_tmp = temp[-1]+1 # fill in redundant age values following the last valid age cell.
                    u_tmp = temp[-1]+num_dummy+1
                    temp.extend(list(range(l_tmp, u_tmp, 1)))
                    agemat.loc[i,:] = temp
                else:
                    figmat.loc[i,:] = quantV.values
                    agemat.loc[i,:] = quantV.index
            if quantNvec.index(k)==0:
                agemat = agemat.iloc[::-1] # stack from top to bottom starting with the latest survey 2013 for example.
                agemat.to_csv("{}_{}_age.csv".format(l,j), index=False)                
            figmat = figmat.iloc[::-1] # stack from top to bottom starting with the latest survey 2013 for example.
            figmat.to_csv("{}_{}_{}.csv".format(l, j, quantvec[quantNvec.index(k)]), index=False)
        # ------ average block, given loop index of l and j.------------------- 
        figmat = DataFrame(np.nan, index=yearvec, columns=range(0,len(lbin),1))        
        for i in yearvec:
            df = pd.read_csv("{}cleaned.csv".format(i))
            if workvec.index(l)==1:
                df = df.loc[df['workcode']==1,df.columns]
            elif workvec.index(l)==0:
                df = df.loc[df['workcode']==0,df.columns]
            else:
                print(i)
                #do nothing
            grouped = df[[j,'weight']].groupby(df['agegroup'])
            num_bins = len(grouped.size().index) # number of valid age groups in a specific year.
            num_dummy = len(lbin) - num_bins
            if j=='dhouses':
                quantV = grouped.apply(houses_mean)                
            else:
                quantV = grouped.apply(nonhouses_mean)            
            if num_dummy>0:
                temp = list(quantV.values)
                temp.extend(list(np.repeat(fake_value,num_dummy))) 
                figmat.loc[i,:] = temp                            
            else:
                figmat.loc[i,:] = quantV.values
        figmat = figmat.iloc[::-1] # stack from top to bottom starting with the youngest cohort 2013 for example.
        figmat.to_csv("{}_{}_avg.csv".format(l, j), index=False)
              
#============================================================================== 3. checked.       
# -------------------------Plot Generation------------------------------------- IV. Plot Single asset data vs kernel smoothing
cohort_cutoff = 2001 # in paper, the author drops people more than 90 year old in 2001.   
h = 5 # bandwidth of kernel smoothing

# y = m(z) : w, dependent variables; z, independent variables
# x, the new locations to be evaluated, on the same space of z.
# output: smat, smooth matrix. fray, values evaluated at x.
# Yang (2006), technical appendix, page 8.
def k_regression(x,z,h,w): # x, z, w have to be array.
    n = len(x)
    g = np.zeros([n,1])
    num, S = ukernel(x,z,h,w)
    denom, S = ukernel(x,z,h,np.ones([len(w)])) # has to be array for the 4th argument.
    I = denom==0.
    s = I.astype(int)
    g = num/(denom+s)
    g = (1.-s)*g
    return g, S
    
def ukernel(x,z,h,w): # x, z, w have to be array. x: to predict at, z: observed independent, w: observed dependent.
    n = len(x) # length to be estimated
    m = len(z) # should be the same as w.
    f = np.zeros([n,1])    
    S = np.zeros([m,m])
    for i in range(0,m,1):
        u  = (z[i]-z)/h
        kf = k_epan(u) # kf is array.
        S[i,:] = kf/sum(kf)
    for i in range(0,n,1):
        u    = (x[i]-z)/h
        kf   = k_epan(u)
        f[i] = np.inner(kf,w)/(m*h)
    return f, S # f; S: smoother matrix
    
# kernel function -- Epanechnikov kernel.        
def k_epan(v): # v has to be narray.
    I = (abs(v)<=1.).astype(int) # debug
    c = 3./4.
    return c*(1.-v**2.)*I        
        
# year matrix
yearmat = DataFrame(np.nan, index=yearvec, columns=range(0,len(lbin),1))
yearmat = yearmat.iloc[::-1] # reverse the index order so that we stack from top to bottom starting with the youngest cohort 2013 for example.       
for i in yearvec:
    yearmat.loc[i] = i
temp = np.matrix(yearmat)

year_column = DataFrame(temp.reshape(len(yearvec)*len(lbin),1,order='F'))
# flatten year, fig and age matrices into column vector in column major

os.chdir(output_folder) 
for k in ['wok','ent','mix']: #workvec:
    for i in ['dhouses','dnonhouses']: #assetvec:
        for j in ['2qt']: #quantvec:
            temp = pd.read_csv("{}_{}_{}.csv".format(k,i,j))
            temp = np.matrix(temp)
            data_column = DataFrame(temp.reshape(len(yearvec)*len(lbin),1,order='F'))
            temp = pd.read_csv("{}_{}_age.csv".format(k,i))
            temp = np.matrix(temp)
            age_column = DataFrame(temp.reshape(len(yearvec)*len(lbin),1,order='F'))
            combined = pd.concat([age_column, data_column, year_column], axis=1)
            combined.columns = ['agegroup','stat','year']
            combined = combined.loc[combined['stat'].notnull(),combined.columns] # remove rows with NaN in Column stat.
            ageprofile = combined.reset_index(drop=True)
            ageprofile['cohort'] = ageprofile['year'] - ageprofile['agegroup']
            #ageprofile.sort_values(by='cohort', inplace=True)
            condition = ageprofile['cohort']>=(cohort_cutoff-90) # only consider subjects born after year 1923 <-----------------
            profile1 = ageprofile.loc[condition,ageprofile.columns]
            profile1.sort_values(by='agegroup', inplace=True) # as in paper.

            # cohort dummy
            cohortdum = pd.get_dummies(profile1['cohort'])
            cohortdum = cohortdum[cohortdum.columns[::-1]]
            cohort_unique_year = cohortdum.columns
            cohort_unique_length = len(cohort_unique_year)
            cohort_unique_year = cohort_unique_year.astype(int)
            cohortdum.columns = [ 'c{}'.format(x) for x in cohort_unique_year]
            profile2  = pd.concat([profile1, cohortdum], axis=1)

            # year dummy
            yeardum = pd.get_dummies(profile1['year']) 
            year_unique_year = yeardum.columns
            year_unique_length = len(year_unique_year)
            year_unique_year = year_unique_year.astype(int)
            yeardum.columns  = [ 'y{}'.format(x) for x in year_unique_year]
            profile3 = pd.concat([profile2, yeardum], axis=1)
            profile4 = profile3.drop(['cohort'], axis=1)

            # prepare for eliminating two degree of freedom in year columns
            profile5 = profile4.loc[:,profile4.columns] # (agegroup, stat, year, cohort dummies, year dummies)
            temp = range(len(profile5.columns)-1, 3+cohort_unique_length-1, -1)
            for m in temp: # correct 11-16-2017
                profile5.iloc[:,m] = profile5.iloc[:,m] - (m-(3+cohort_unique_length))*profile5.iloc[:,3+cohort_unique_length+1] + (m-(3+cohort_unique_length+1))*profile5.iloc[:,3+cohort_unique_length]

            # drop one dimension in cohort dummy and another two in year dummy
            profile6 = profile5.loc[:,profile5.columns]
            str1 = "c{}".format(cohort_unique_year[0])
            str2 = "y{}".format(year_unique_year[0])
            str3 = "y{}".format(year_unique_year[1])
            profile6.drop(str1, axis=1, inplace=True)
            profile6.drop(str2, axis=1, inplace=True)
            profile6.drop(str3, axis=1, inplace=True)
            profile6.sort_values(by="agegroup", inplace=True) # important
            profile6 = profile6.loc[profile6['agegroup']<=85,profile6.columns]

            # deaton estimation

            # convert column into list and then numpy array.
            y = profile6['stat'].values # array
            z = profile6['agegroup'].values # array []
            _, S = ukernel([0.],z,h,np.ones([len(z)])) # x, z, w have to be array.

            #y = np.array(y)[:,None] # column vector []
            Y = np.array(y).reshape([len(y),1])
            Unit = np.identity(S.shape[0])
            Dmat = profile6.iloc[:,3:] # dummy matrix
            yres = np.dot((Unit-S),Y) # residual can be expalined by cohort and year effect rather than nonlinear kernel smoothing function of age.
            xres = np.dot((Unit-S),Dmat)
            xresT = np.transpose(xres)

            temp = np.dot(xresT,xres)
            temp = inv(temp)
            temp = np.dot(temp,xresT)
            beta = np.dot(temp,yres)

            maxage = 85
            x = np.array([n/10. for n in range(290,maxage*10+1,10) ])
            temp = np.dot(Dmat,beta)
            w = np.dot(S,Y-temp) 
            g, _ = k_regression(x,z,h,np.squeeze(w)) # trick to convert one dimention data frame into array []

            # plot zone
            fig, ax = plt.subplots() 
            # or
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)

            if np.squeeze(Y).max()/1000.<300.:
                scale_factor  = 1000. # in unit of thousands
                ylabel_option = 1
            else:
                scale_factor  = 1000.*1000. # in unit of millions
                ylabel_option = 2
            
            line1, = ax.plot(z, np.squeeze(Y)/scale_factor, label='data confouned by cohort and time effects')
            line2, = ax.plot(x, g/scale_factor, label='prediction without chort and time effects')
            #ax.set_xlim(25, 90)
            #ax.set_xticks(np.arange(25, 90, 10))
            if ylabel_option == 1:
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter('\$%2.fk')) # if use percent sign (%), the option is '%i%%'                                
                plt.figtext(0.015,0.7,'(thousands of 2013 US dollars)', fontsize=12, color=ax.xaxis.label.get_color(),rotation='vertical')                
            elif ylabel_option ==2:
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter('\$%2.1fM')) # if use percent sign (%), the option is '%i%%'                                                
                plt.figtext(0.015,0.7,'(millions of 2013 US dollars)', fontsize=12, color=ax.xaxis.label.get_color(),rotation='vertical')                
            if k == 'ent':
                ax.legend(loc='upper left', fancybox=True, shadow=True)
            elif k == 'wok' and i == 'dhouses':
                ax.legend(loc='lower right', fancybox=True, shadow=True)
            elif k == 'wok' and i == 'dnonhouses':
                ax.legend(loc='upper left', fancybox=True, shadow=True)
            ax.set_xlabel('age',fontsize=12)
            fig.suptitle('{} {}, {}'.format(workstr[workvec.index(k)].title(), assetstr[assetvec.index(i)], quantstr[quantvec.index(j)]), fontsize=12)
            ax.set_title('(Survey of Consumer Finances 1983-2013)',fontsize=10)
            # ax.set_title('(2013 US dollars)',fontsize=10)
            # Tweak spacing to prevent clipping of ylabel
            #plt.subplots_adjust(left=0.15)

            os.chdir(plot_folder) 

            major_ticks = np.arange(25, 86, 10)                                              
            minor_ticks = np.arange(25, 86, 5)             
            ax.set_xticks(major_ticks)                                                       
            ax.set_xticks(minor_ticks, minor=True)                                                      
            #ax.grid(which='both')                                                            
            # or if you want differnet settings for the grids:                               
            ax.grid(which='minor', alpha=0.2)                                                
            ax.grid(which='major', alpha=0.5)       
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = 'black'
            plt.rcParams['grid.alpha'] = 1
            plt.rcParams['grid.color'] = "#cccccc"            
            plt.grid(True)
            plt.savefig('{}_{}_{}.png'.format(i,j,k), dpi=150, transparent=True)
            plt.savefig('{}_{}_{}.svg'.format(i,j,k), format='svg')
            plt.show()
            
            #
            outputX = DataFrame(z)
            outputY = DataFrame(np.squeeze(Y))
            df = pd.concat([outputX, outputY], axis = 1) # z: observed x; y: corresponding statistics
            df.columns = ['z','y']
            os.chdir(output_folder)
            df.to_csv('original_{}_{}_{}.csv'.format(k,i,j), index=False) # from pseudo panel data.
            outputX = DataFrame(x)
            outputY = DataFrame(g)
            df = pd.concat([outputX, outputY], axis = 1) # x: x to be evaluated; g: corresponding prediction
            df.columns = ['x','g']
            df.to_csv('smooth_{}_{}_{}.csv'.format(k,i,j), index=False)
            
# ============================================================================= V. presence of entrepreneurs, concentration of wealth, income group's asset holdings
# ---- reproduce graphs used in oral prelim
# First, generate the summary statistics for graphs in next section V.            
# ---------------------------------------------------------------------- part 1            
trivial = 1.e-8
os.chdir(output_folder)
df = pd.read_csv('19832013cleaned.csv')
df.rename(columns={'weight':'den','dhomeeq':'hom','dnonhouses':'ast'},inplace=True)
df = df.loc[df.index,['den','hom','ast','year']] # partial frame obtained from a subset of columns.
df['wel'] = df['hom'] + df['ast']
yearvec = range(1983,2014,3)
quantv1 = [20, 80, 85, 90, 95, 99]
quantv2 = list(quantv1)
quantv2.insert(0,0)
quantv2.append(100)
stat_str_vec1 = ['B20%','B80%','T15%','T10%','T5%','T1%']
numvec1 = [x/100. for x in quantv1] # [0.2, 0.8, 0.95, 0.99]

ds = DataFrame(np.nan, index=yearvec, columns=stat_str_vec1) # answer frame

for m in range(0,len(yearvec)): #range(1983,2014,3):
    #m = 0
    dg = df.loc[df['year']==yearvec[m], df.columns]
    dg['wgtwel'] = dg['den']*dg['wel'] # the objects we like to sum over by groups.
    bins = [] # bins used for pd.cut
    for n in [x/100. for x in quantv1]:
        bins.append(weighted_quantile(dg['wel'], n, dg['den']))    
    bins.insert(0,dg['wel'].min()-trivial) # the minimum in bins
    bins.append(dg['wel'].max()+trivial) # the maximum in bins
    gid = pd.cut(dg['wel'], bins=bins, right=True) # classification based on bins
    dg['gid'] = gid.cat.codes # convert categorical data into integer codes.
    totwgtwel = dg['wgtwel'].sum()
    for n in range(0,2): # do cumsum over 'wgtwel' for the first two summary statistics, bottom 20% and bottom 80%.
        cond = (dg['gid']<=n)
        dh = dg.loc[cond, dg.columns]
        ds.loc[yearvec[m], stat_str_vec1[n]] = dh['wgtwel'].sum() # store the outcome.
    for n in range(3,len(stat_str_vec1)+1):
        cond = (dg['gid']>=n)
        dh = dg.loc[cond, dg.columns]
        ds.loc[yearvec[m], stat_str_vec1[n-1]] = dh['wgtwel'].sum()
    ds.loc[yearvec[m],ds.columns] = ds.loc[yearvec[m],ds.columns]/totwgtwel

ds.to_csv('share_net_worth_rich2poor.csv')
    
temp = [] # to store the average of each net worth groups (bottom 20%, bottom 80%, etc.)
for n in ds.columns:
    temp.append(ds[n].mean())
avg_sr = Series(temp,index=stat_str_vec1)
        
# ---------------------------------------------------------------------- part 2
trivial = 1.e-8
stat_num_vec1 = [1,20,40,60,80,99]
stat_str_vec1 = ['99%','80%','60%','40%','20%','1%']
year_num_vec1 = range(1983,2014,3)
os.chdir(output_folder)
df = pd.read_csv('19832013cleaned.csv')
df.rename(columns={'weight':'den','dhomeeq':'hom','dnonhouses':'ast','workcode':'boss'}, inplace=True)
df = df.loc[df.index,['den','ast','hom','boss','year']] # partial frame obtained from a subset of columns.
df['wel'] = df['ast'] + df['hom']
ds = DataFrame(np.nan, index=stat_str_vec1, columns=year_num_vec1)
for m in range(0,len(year_num_vec1)):
    dg = df.loc[df['year']==year_num_vec1[m], df.columns]
    tot_ent_the_year = dg[dg['boss']==1]['den'].sum()
    bins = []
    for n in [x/100. for x in stat_num_vec1]:
        bins.append(weighted_quantile(dg['wel'], n, dg['den']))
    bins.insert(0,dg['wel'].min()-trivial)
    bins.append(dg['wel'].max()+trivial)
    gid = pd.cut(dg['wel'], bins=bins, right=True)
    dg['gid'] = gid.cat.codes
    for n in range(0,len(stat_num_vec1)):
        cond1 = (dg['gid']>n)
        cond2 = (dg['gid']>n) & (dg['boss']==1)
        di = dg.loc[cond1, dg.columns]
        dj = dg.loc[cond2, dg.columns]
        ds.loc[stat_str_vec1[n], year_num_vec1[m]] = dj['den'].sum()/di['den'].sum()
ds.to_csv('ent_share_trending_upward_19832013.csv')        
# ---------------------------------------------------------------------- part 3        
trivial = 1.e-8
os.chdir(output_folder)
df = pd.read_csv('19832013cleaned.csv')    
df.rename(columns={'weight':'den', 'dhomeeq':'hom', 'dnonhouses':'ast', 'workcode':'boss', 'income':'inc'}, inplace=True)
df = df.loc[df['year']==2013, ['den', 'ast', 'hom', 'inc', 'boss', 'year']] # partial frame obtained from a subset of columns. 
statNumVec1 = range(20,81,20)
statStrVec1 = [str(i) for i in range(1,len(statNumVec1)+2)]
jobStrVec1 = ['w','e']
ds = DataFrame(np.nan, index = ['e_avg_ast','e_avg_hom','e_med_ast','e_med_hom','w_avg_ast','w_avg_hom','w_med_ast','w_med_hom','rat_ent'], columns=statStrVec1)
bins = []
for n in [x/100. for x in statNumVec1]:
    bins.append(weighted_quantile(df['inc'], n, df['den']))
bins.insert(0,df['inc'].min()-trivial)
bins.append(df['inc'].max()+trivial)
gid = pd.cut(df['inc'], bins=bins, right=True)
df['gid'] = gid.cat.codes # array[0,1,2,3,4]
for n in range(0, len(np.unique(df['gid'].values))):
    #n = 0
    cond1 = (df['gid']==n)
    dh = df.loc[cond1, df.columns] 
    totMass = dh['den'].sum()
    for w in [0,1]:
        cond2 = (df['gid']==n) & (df['boss']==w)
        dg = df.loc[cond2, df.columns] # If .loc is missing, pops up the message "TypeError: 'Series' objects are mutable, thus they cannot be hashed"
        ds.loc['{}_avg_ast'.format(jobStrVec1[w]), statStrVec1[n]] = weighted_mean(dg['ast'], dg['den'])
        ds.loc['{}_avg_hom'.format(jobStrVec1[w]), statStrVec1[n]] = weighted_mean(dg['hom'], dg['den'])
        ds.loc['{}_med_ast'.format(jobStrVec1[w]), statStrVec1[n]] = weighted_quantile(dg['ast'], 0.5, dg['den'])
        ds.loc['{}_med_hom'.format(jobStrVec1[w]), statStrVec1[n]] = weighted_quantile(dg['hom'], 0.5, dg['den'])   
        if w==1:
            ds.loc['rat_ent', statStrVec1[n]] = dg['den'].sum()/totMass
ds.to_csv('income_group_ent8wok.csv')            
# ============================================================================= V.
# Plot data in section IV
# ---------------------------------------------------------------------- part 1
linewidth = 1
scale = 100.
stat_str_vec1 = ['B20%','B80%','T15%','T10%','T5%','T1%']
os.chdir(output_folder)
df = pd.read_csv('share_net_worth_rich2poor.csv', index_col=0 )
fig, ax = plt.subplots()
xray = df.index
line_b20, = ax.plot(xray, df['B20%']*scale, '--', marker='o', linewidth=linewidth, label='bottom 20%') 
line_b80, = ax.plot(xray, df['B80%']*scale, '--', marker='o', linewidth=linewidth, label='bottom 80%') 
line_t15, = ax.plot(xray, df['T15%']*scale, '--', marker='o', linewidth=linewidth, label='top 15%') 
line_t05, = ax.plot(xray, df['T5%']*scale, '--', marker='o', linewidth=linewidth, label='top 5%') 
line_t01, = ax.plot(xray, df['T1%']*scale, '--', marker='o', linewidth=linewidth, label='top 1%') 
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%i%%'))
fig.suptitle('Trend of wealth concentration'.title())
ax.set_title('(Survey of consumer finances, 1983-2013)'.title(),fontsize=10)

ax.xaxis.set_ticks(range(1983,2014,6))
xticklabels = ax.get_xticks().tolist()
xticklabels = [x for x in range(1983,2014,6)]
ax.set_xticklabels(xticklabels)

ax.set_xlabel('Year')
ax.set_ylabel('Share of net assets')
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)

major_ticks = np.arange(0, 101, 10)
minor_ticks = np.arange(0, 101, 5)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"

"""
plt.text(1998,1.5,'bottom 20%')
plt.text(1998,20.5,'bottom 80%')
plt.text(1998,35,'top 1%')
plt.text(1998,59,'top 5%')
plt.text(1998,78,'top 15%')
"""
plt.grid(True)

temp = [] # to store the average of each net worth groups (bottom 20%, bottom 80%, etc.)
for n in df.columns:
    temp.append(df[n].mean())
avg_sr = Series(temp,index=stat_str_vec1)

stat_str_vec1 = ['B20%','B80%','T15%','T10%','T5%','T1%']
for m in [0,1,2,4,5]:
    ax.axhline(y=avg_sr[m]*scale,xmin=0,xmax=3,linestyle="--",c="black",linewidth=0.5,zorder=0)
    if m==1: # bottom 80%
        ax.annotate('{:4.1f}%'.format(avg_sr[m]*scale),(1982, avg_sr[m]*scale-4))
    elif m==0: # bottom 20%
        ax.annotate('{:4.1f}%'.format(avg_sr[m]*scale),(1982, avg_sr[m]*scale+1.5))
    else:
        ax.annotate('{:4.1f}%'.format(avg_sr[m]*scale),(1982, avg_sr[m]*scale+0.5))

ax.legend(loc='upper right',ncol=3, fancybox=True, shadow=True)

# save with transparent background and customary dpi
os.chdir(plot_folder)
plt.savefig('Trends_of_wealth_concentration.png', dpi=450, transparent=True)
plt.savefig("Trends_of_wealth_concentration.svg", format="svg")
plt.show()

# ---------------------------------------------------------------------- part 2
linewidth = 1
os.chdir(output_folder)
df = pd.read_csv('ent_share_trending_upward_19832013.csv', index_col=0 )
fig, ax = plt.subplots()
xray = df.index
nray = [1,20,40,60,80,99]
scale = 100
#line_1983, = ax.plot(nray, df['1983']*scale, '-', linewidth=linewidth, label='1983') 
#line_1986, = ax.plot(nray, df['1986']*scale, '-', linewidth=linewidth, label='1986') 
line_1989, = ax.plot(nray, df['1989']*scale, linestyle='--', marker='o', linewidth=linewidth, label='1989') 
#line_1992, = ax.plot(nray, df['1992']*scale, '-', linewidth=linewidth, label='1992') 
line_1995, = ax.plot(nray, df['1995']*scale, linestyle='--', marker='o', linewidth=linewidth, label='1995') 
#line_1998, = ax.plot(nray, df['1998']*scale, '-', linewidth=linewidth, label='1998') 
line_2001, = ax.plot(nray, df['2001']*scale, linestyle='--', marker='o', linewidth=linewidth, label='2001') 
#line_2004, = ax.plot(nray, df['2004']*scale, '-', linewidth=linewidth, label='2004') 
line_2007, = ax.plot(nray, df['2007']*scale, linestyle='--', marker='o', linewidth=linewidth, label='2007') 
#line_2010, = ax.plot(nray, df['2010']*scale, '-', linewidth=linewidth, label='2010') 
line_2013, = ax.plot(nray, df['2013']*scale, linestyle='--', marker='o', linewidth=linewidth, label='2013') 

ax.xaxis.set_ticks([1,20,40,60,80,99])
xticklabels = ax.get_xticks().tolist()
xticklabels = ['99%','80%','60%','40%','20%','1%']
ax.set_xticklabels(xticklabels)

ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%i%%'))
fig.suptitle('Presence of entrepreneurs in wealth groups'.title())
ax.set_title('(Survey of consumer finances, 1983-2013)'.title(),fontsize=10)
ax.legend(loc='upper left', title="survey year", ncol=3, fancybox=True, shadow=True)

ax.set_xlabel('Top wealth groups')
ax.set_ylabel('Share of entrepreneurs within group')
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)

major_ticks = np.arange(10, 61, 10)
minor_ticks = np.arange(10, 61, 5)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"

plt.grid(True)

# save with transparent background and customary dpi
os.chdir(plot_folder)
plt.savefig('Trends_of_share_of_entrepreneurs.png', dpi=450, transparent=True)
plt.savefig("Trends_of_share_of_entrepreneurs.svg", format="svg")
plt.show()

#----------------------------------------------------------------------- part 3
os.chdir(output_folder)
df = pd.read_csv('income_group_ent8wok.csv', index_col=0 )
# reference for sharing axis: https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.subplots.html
# reference for subplot grid: https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html#plt.subplot:-Simple-Grids-of-Subplots
fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col')
fig.suptitle('Avergae household asset holdings'.title(),fontsize=12)
fig.subplots_adjust(hspace=.12,bottom=0.15) # adjust the space between subplots

e_avg_ast = df.loc['e_avg_ast', df.columns]
e_avg_hom = df.loc['e_avg_hom', df.columns]
e_avg_wel = e_avg_ast + e_avg_hom
rat_ent = df.loc['rat_ent',df.columns]
pos = list(range(1,6))
scale1 = 1.e-6
#
# broken axis reference: https://stackoverflow.com/questions/13027147/histogram-with-breaking-axis-and-interlaced-colorbar
#
ax[0,0].bar(pos, e_avg_hom*scale1, color='#F78F1E', label='home equity')
ax[0,0].bar(pos, e_avg_ast*scale1, bottom=e_avg_hom*scale1, color='#FFC222', label='liquid assets')
ax[1,0].bar(pos, e_avg_hom*scale1, color='#F78F1E', label='home equity')
ax[1,0].bar(pos, e_avg_ast*scale1, bottom=e_avg_hom*scale1, color='#FFC222', label='liquid assets')  
ax[0,0].set_ylim(1.5,8)
ax[1,0].set_ylim(0,1)
#ax[0,0].legend(loc='upper left', fancybox=True, shadow=True)
ax[1,0].xaxis.set_ticks(pos)
ax[1,0].set_xlabel('income group')
ax[1,0].xaxis.label.set_size(12)
ax[0,0].set_title('Business Households',fontsize=10)
handles, labels = ax[0,0].get_legend_handles_labels()
ax[0,0].legend(handles[::-1], labels[::-1], loc='upper left', fancybox=True, shadow=True)
ax[1,0].legend().set_visible(False)
ax[0,0].spines['bottom'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)
ax[0,0].xaxis.tick_top()
ax[0,0].tick_params(top='off',labeltop='off') # disable tick and label along the top edge.
ax[1,0].xaxis.tick_bottom()
# add the share of entrepreneurs ---
rat_ent = rat_ent*100.
e_avg_wel = e_avg_wel*scale1
#
# add text on top of bar reference: https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
# 
for i, v in enumerate(rat_ent): # index, value
    ax[1,0].text(i-0.25,e_avg_wel[i-1]+0.05,'{:2.0f}%'.format(rat_ent[i-1]),color='black',fontweight='light') #,fontweight='bold')
    if i == 4:
        ax[0,0].text(i+1-0.25,e_avg_wel[i]+0.3,'{:2.0f}%'.format(rat_ent[i]),color='black',fontweight='light') #,fontweight='bold')
# add the broken axis mark on x axis ---
d = 0.025
kwargs = dict(transform=ax[0,0].transAxes, color='gray', clip_on=False)
ax[0,0].plot((-d,+d),(-d,+d),**kwargs)
ax[0,0].plot((1-d,1+d),(-d,+d), **kwargs)
kwargs.update(transform=ax[1,0].transAxes)
ax[1,0].plot((-d,+d),(1-d,1+d), **kwargs)
ax[1,0].plot((1-d,1+d),(1-d,1+d), **kwargs)

# text reference: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.figtext
ax[1,0].text(-1.1,1.6,'(millions of 2013 US dollars)',rotation=90, color=ax[1,0].xaxis.label.get_color(), fontsize=12)
plt.figtext(0.05,0.01,'Source: 2013 Survey of Consumer Finances', color=ax[1,0].xaxis.label.get_color())

# ---- non-business family
rat_wok = 100.-rat_ent
w_avg_ast = df.loc['w_avg_ast', df.columns]
w_avg_hom = df.loc['w_avg_hom', df.columns]
w_avg_wel = w_avg_ast + w_avg_hom

ax[1,1].bar(pos, w_avg_hom*scale1, color='#F78F1E', label='home equity')
ax[1,1].bar(pos, w_avg_ast*scale1, bottom=w_avg_hom*scale1, color='#FFC222', label='liquid assets')
ax[0,1].bar(pos, w_avg_hom*scale1, color='#F78F1E', label='home equity')
ax[0,1].bar(pos, w_avg_ast*scale1, bottom=w_avg_hom*scale1, color='#FFC222', label='liquid assets')  
ax[0,1].set_ylim(1.5,8)
ax[1,1].set_ylim(0,1)
ax[0,1].legend(loc='upper left')
ax[1,1].xaxis.set_ticks(pos)
ax[1,1].set_xlabel('income group')
# ax[1,0].set_ylabel('(Millions of 2013 US dollars)')
ax[1,1].xaxis.label.set_size(12)
ax[0,1].set_title('Worker Households',fontsize=10)
handles, labels = ax[0,0].get_legend_handles_labels()
ax[0,1].legend(handles[::-1], labels[::-1], loc='upper left')
ax[0,1].legend().set_visible(False)
ax[1,1].legend().set_visible(False)
ax[0,1].spines['bottom'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)
ax[0,1].xaxis.tick_top()
ax[0,1].tick_params(top='off',labeltop='off') # disable tick and label along the top edge.
ax[1,1].xaxis.tick_bottom()
# add the share of entrepreneurs ---
w_avg_wel = w_avg_wel*scale1
for i, v in enumerate(rat_wok): # index, value
    if i<=3:
        ax[1,1].text(i+1-0.25,w_avg_wel[i]+0.05,'{:2.0f}%'.format(rat_wok[i]),color='black',fontweight='light') 
    if i == 4:
        ax[0,1].text(i+1-0.25,w_avg_wel[i]+0.3,'{:2.0f}%'.format(rat_wok[i]),color='black',fontweight='light') 
# add the broken axis mark on x axis ---
d = 0.025
kwargs = dict(transform=ax[0,1].transAxes, color='gray', clip_on=False)
ax[0,1].plot((-d,+d),(-d,+d),**kwargs)
ax[0,1].plot((1-d,1+d),(-d,+d), **kwargs)
kwargs.update(transform=ax[1,1].transAxes)
ax[1,1].plot((-d,+d),(1-d,1+d), **kwargs)
ax[1,1].plot((1-d,1+d),(1-d,1+d), **kwargs)

os.chdir(plot_folder)
plt.savefig('income_group_ent8wok.png', dpi=450, transparent=True)
plt.savefig("income_group_ent8wok.svg", format="svg")
plt.show()