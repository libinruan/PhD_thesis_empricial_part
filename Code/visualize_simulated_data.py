# Original file name in my Phd project: s3c_matrix.py
# ----------------------------------------------------------------------------- # 1
# It takes roughly 3 minutes to complete the program. 11-9-2017

import os
import dfgui # dfgui.show(df) # python versions earlier than v.3x is required.
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
#import matplotlib.pyplot as plt
import itertools
import scipy
import scipy.optimize
from scipy.stats.stats import pearsonr 
from matplotlib.ticker import AutoMinorLocator # automatically locate minor ticks between current major ticks
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ----------------------------------------------------------------------------- # 1.5 I. compile model results into an integrated frame.
data_source   = 'C:/Users/libin/Downloads' # SCF survey data
output_folder = 'C:/Users/libin/Downloads/CleanData' # .csv output
plot_folder   = 'C:/Users/libin/Downloads/Plot' # .png and .svg output
model_folder  = 'E:/GoogleDrive/python_projects/research/PhDThesisPython/research/10-thesis-outcome/result_experiment' # archives for fils from MSI.

# Working directory
os.chdir(model_folder) 
df = pd.read_csv('s3c.txt', sep='\s+', header=None, names=['a','h','k','z','y','kp','yp','op','t','idx'])

# Merge the two data frames
mode6taskid = 0 
a1 = pd.read_csv('ast_00{}.txt'.format(mode6taskid),header=None) 
x1 = pd.read_csv('axw_00{}.txt'.format(mode6taskid),header=None) 
b1 = pd.read_csv('buz_00{}.txt'.format(mode6taskid),header=None) 
c1 = pd.read_csv('csp_00{}.txt'.format(mode6taskid),header=None) 
h1 = pd.read_csv('hom_00{}.txt'.format(mode6taskid),header=None) 
i1 = pd.read_csv('inc_00{}.txt'.format(mode6taskid),header=None)
s1 = pd.read_csv('sef_00{}.txt'.format(mode6taskid),header=None) 
n1 = pd.read_csv('non_00{}.txt'.format(mode6taskid),header=None) 
p1 = pd.read_csv('sax_00{}.txt'.format(mode6taskid),header=None) 
w1 = pd.read_csv('wtx_00{}.txt'.format(mode6taskid),header=None) 

df1 = pd.concat([a1,x1,b1,c1,h1,i1,s1,n1,p1,w1], axis=1)
df1.columns = ['ast','axw','buz','csp','hom','inc','den','non','sax','wtx']
dg1 = pd.concat([df,df1], axis=1)
dg1['ttx'] = dg1['non'] + dg1['sax'] + dg1['wtx']
os.chdir(output_folder)
dg1.to_csv('benchmark_df.csv', index=False)

mode6taskid = 1
os.chdir(model_folder) 
a2 = pd.read_csv('ast_00{}.txt'.format(mode6taskid),header=None) 
x2 = pd.read_csv('axw_00{}.txt'.format(mode6taskid),header=None) 
b2 = pd.read_csv('buz_00{}.txt'.format(mode6taskid),header=None) 
c2 = pd.read_csv('csp_00{}.txt'.format(mode6taskid),header=None) 
h2 = pd.read_csv('hom_00{}.txt'.format(mode6taskid),header=None) 
i2 = pd.read_csv('inc_00{}.txt'.format(mode6taskid),header=None)
s2 = pd.read_csv('sef_00{}.txt'.format(mode6taskid),header=None) 
n2 = pd.read_csv('non_00{}.txt'.format(mode6taskid),header=None) 
p2 = pd.read_csv('sax_00{}.txt'.format(mode6taskid),header=None) 
w2 = pd.read_csv('wtx_00{}.txt'.format(mode6taskid),header=None) 

df2 = pd.concat([a2,x2,b2,c2,h2,i2,s2,n2,p2,w2], axis=1)
df2.columns = ['ast','axw','buz','csp','hom','inc','den','non','sax','wtx']
dg2 = pd.concat([df,df2], axis=1)
dg2['ttx'] = dg2['non'] + dg2['sax'] + dg2['wtx']
os.chdir(output_folder)
dg2.to_csv('experiment_df.csv', index=False)

# ===================== Model Plot ============================================ II. find out the equivalent consumption changes.
# ----------------------------------------------------------------------------- # 2.0

theta = 0.09 
sigma = 1.5

def u(x,pcg): # x: dataframe, pcg: percentage change of consumption
    if x['csp']>0.: # not need to screen out the case where 'hom'==0.
        c = x['csp']*(1.+pcg*0.01) 
        h = x['hom']
        g = c**(1.-theta)*(h+1.e-7)**theta
        return (g**(1.-sigma)-1.)/(1.-sigma)
    else:
        return 0.

def objfunc(x):
    dg1['u'] = dg1.apply(u,pcg=0.,axis=1)
    dg2['u'] = dg2.apply(u,pcg=x,axis=1)
    return dg1['u'].subtract(dg2['u']).sum()

# "dg1" and "dg2" are hard-wired dataframe named for the operation of objfunc function:
print scipy.optimize.brentq(objfunc, -99.0, 99.0) # objective function has to be continuous. search in the interval of changes by the percentage from -99% to 99%.

# ----------------------------------------------------------------------------- # 2.1 Primary functions for genearting "weighted" summary statistics later.
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
        sorter = np.argsort(values) #[]
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

# ----------------------------------------------------------------------------- # 2.2 Not used in this file.
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

# ----------------------------------------------------------------------------- # 2.3
def income_mean(x):
    return weighted_mean(x['inc'], x['den'])
def income_median(x):
    return weighted_quantile(x['inc'], 0.5, sample_weight=x['den'])
def asset_quantile(x,quant=0.5):
    return weighted_quantile(x['ast'], quant, sample_weight=x['den'])
def aftertaxwealth_quantile(x,quant=0.5):
    return weighted_quantile(x['axw'], quant, sample_weight=x['den'])
def business_quantile(x,quant=0.5):
    return weighted_quantile(x['buz'], quant, sample_weight=x['den'])
def consumption_quantile(x,quant=0.5):
    return weighted_quantile(x['csp'], quant, sample_weight=x['den'])
def housing_quantile(x,quant=0.5):
    return weighted_quantile(x['hom'], quant, sample_weight=x['den'])
def taxpayments_quantile(x,quant=0.5):
    return weighted_quantile(x['ttx'], quant, sample_weight=x['den'])

# ----------------------------------------------------------------------------- # 2.4
# similiar to the outcome of Fortran # in reality, 2012 U.S. median household income was $51.915.
# similiar to the outcome of Fortran # in reality, 2013 U.S. median household income was $52,250.
# NPR news, 2017 U.S. median household income is $59,000.
# os.chdir('C:/Users/libin/Downloads/PlotII')
    
assetvec = ['dhouses','dnonhouses'] # dnonhouses
assetstr = ['housing assets','nonhousing assets']
quantvec = ['1qt','2qt','3qt']
quantstr = ['1st quartile','2nd quartile','3rd quartile']
quantNvec = [0.25, 0.5, 0.75]
workvec   = ['wok','ent','mix']
workstr   = ['Worker households','Business households','all households']
trialvec  = ['benchmark','experiment']
trialstr  = ['Baseline model with capital income taxes','Alternative model with wealth taxes']
serialvec = ['ast','axw','csp','hom','inc']
serialvec2= ['ast','axw','csp','hom','inc','ttx']
serialstr = ['financial wealth','after-tax wealth','consumption','home equity','before-tax income']
serialstr2= ['financial wealth','after-tax wealth','consumption','home equity','before-tax income','tax payments']

# start here=================================================================== # 2.5 III. Single asset plot with its kernel smoothed couterparts.
h = 5 # 3.5
x = np.array([i/10. for i in range(10,141,2)])
x_xrelabel = [i*5+15 for i in x]
USmedian_inc = 52000.
# -----------------------------------------------------------------------------
from matplotlib.ticker import AutoMinorLocator # automatically locate minor ticks between current major ticks
for trial in ['benchmark','experiment']:
    for target in serialvec2: # serialvec2= ['ast','axw','csp','hom','inc','ttx']
        for workstatus in [0,1]: # workvec  = ['wok','ent','mix']
            os.chdir(output_folder)
            df = pd.read_csv('{}_df.csv'.format(trial))
            
            # compute the scaling factor based on median annual household income.
            condition = (df['k']==0) & (df['t']<=9)
            dg = df.loc[condition, df.columns]
            average_inc = income_mean(dg)
            median_inc = income_median(dg)            
            scale0 = USmedian_inc/median_inc
            
            if workstatus==0: # working households
                condition = (df['k']==0)
                dg = df.loc[condition, df.columns]
            elif workstatus>0:
                condition = (df['k']>0)
                dg = df.loc[condition, df.columns]            
            grouped = dg.groupby('t')
            
            if target=='ast':
                data = DataFrame(list(grouped.apply(asset_quantile,quant=0.5)))
            elif target=='axw':
                data = DataFrame(list(grouped.apply(aftertaxwealth_quantile,quant=0.5)))
            elif target=='buz':
                data = DataFrame(list(grouped.apply(business_quantile,quant=0.5)))
            elif target=='csp':
                data = DataFrame(list(grouped.apply(consumption_quantile,quant=0.5)))
            elif target=='hom':
                data = DataFrame(list(grouped.apply(housing_quantile,quant=0.5)))
            elif target=='inc':
                data = DataFrame(list(grouped.apply(income_median)))
            elif target=='ttx':
                data = DataFrame(list(grouped.apply(taxpayments_quantile,quant=0.5)))                

            data.index = np.array(range(1,15,1)) # 14 age brackets.
            data_xrelabel = [i*5+15 for i in data.index] # relabel it to [20, 25, ..., 85]

            z = data.index # (original) data of x-axis for kernel smooting 
            w = data.values # data (summary statistics) of y-axis to be smoothed.
            g, _ = k_regression(x,z,h,np.squeeze(w)) # kernel smoothed. g is the array of prediction evaluated at array x: np.array([i/10. for i in range(10,141,2)])
            
            # plot
            fig, ax = plt.subplots()   
            if w.max()*scale0/1000.<500.:
                scale = scale0/1000.
                ylabel_option = 1
            else:
                scale = scale0/(1000.*1000.)
                ylabel_option = 2
            #
            line1, = ax.plot(data_xrelabel, data.values*scale, '-', linewidth=2, label='model solution')
            line2, = ax.plot(x_xrelabel, g*scale, '--', label='kernel-smoothed model solution')
            #
            if ylabel_option == 1:
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%2.0f')) # if use percent sign (%), the option is '%i%%' and '\$%2.fk'    
                ax.set_ylabel('(thousands of 2013 U.S. dollars)')            
            elif ylabel_option ==2:
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%2.1f')) # if use percent sign (%), the option is '%i%%' and '\$%2.1fM'   
                ax.set_ylabel('(millions of 2013 U.S. dollars)')            
            
            fig.suptitle('{} {}'.format('2nd', 'quartile {} {}'.format(workstr[workstatus], serialstr2[serialvec2.index(target)]).title()))

            if (target=='ast'):
                ax.legend(loc='upper left', fancybox=True, shadow=True)      
            elif (target=='hom'):
                ax.legend(loc='lower right', fancybox=True, shadow=True)
                
            major_ticks = np.arange(25, 86, 10)                                              
            minor_ticks = np.arange(25, 86, 5)             
            ax.set_xticks(major_ticks)                                                       
            ax.set_xticks(minor_ticks, minor=True)      
            
            # automatically set the minor ticks on yaxis based on the major ticks
            minor_locator = AutoMinorLocator(2)
            ax.yaxis.set_minor_locator(minor_locator)
            plt.grid(which='minor')
                                                     
            #ax.grid(which='both')                                                            
            # or if you want differnet settings for the grids:                               
            #plt.figtext(0.015,0.7,'(millions of 2013 US dollars)', fontsize=12, color=ax.xaxis.label.get_color(),rotation='vertical')
            ax.grid(which='minor', alpha=0.2)                                                
            ax.grid(which='major', alpha=0.5)       
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = 'black'
            plt.rcParams['grid.alpha'] = 1
            plt.rcParams['grid.color'] = "#cccccc"  
            ax.set_xlabel('age',fontsize=12)
            ax.set_title('({})'.format(trialstr[trialvec.index(trial)]), fontsize=11)
            
            plt.grid(True)
            os.chdir(plot_folder)
            plt.savefig('{}_{}_{}_{}.png'.format(trial, workvec[workstatus], target, 'Q2'), dpi=150, transparent=True)
            plt.savefig('{}_{}_{}_{}.svg'.format(trial, workvec[workstatus], target, 'Q2'), format="svg")
            # plt.show()
            
            # store the unsmoothed model summary statistics
            os.chdir(output_folder)
            data.loc[:,data.columns] *= scale0
            ds = pd.concat([data], axis=1)
            ds.columns = [target]
            ds.to_csv('{}_{}_{}_{}_unsmoothed.csv'.format(trial,workvec[workstatus],target,'Q2'))
            
# Summary: generate one red model age profile of asset holding and a kernel-smoothed dashed line in blue, for each target. 
            
# ----------------------------------------------------------------------------- # 2.5.1 IV. Tax system comparison.
# tax system comparison plots.
#
for workstatus in [0,1]: # workvec[workstatus]
    # generate two frames one for benchmark and one for experiment each of which hosts the complete set of stats.
    for trial in ['benchmark','experiment']:
        for target in serialvec2: #serialvec: # serialvec; serialvec = ['ast','axw','csp','hom','inc']
            
            if (trial=='benchmark') & (serialvec2.index(target)==0) :
                os.chdir(output_folder)
                dg1 = pd.read_csv('{}_{}_{}_{}_unsmoothed.csv'.format(trial,workvec[workstatus],target,'Q2'), index_col=0)           
            elif (trial=='experiment') & (serialvec2.index(target)==0):
                os.chdir(output_folder)
                dg2 = pd.read_csv('{}_{}_{}_{}_unsmoothed.csv'.format(trial,workvec[workstatus],target,'Q2'), index_col=0)           
            else:
                if trial=='benchmark':
                    os.chdir(output_folder)
                    dg1[target] = pd.read_csv('{}_{}_{}_{}_unsmoothed.csv'.format(trial,workvec[workstatus],target,'Q2'), index_col=0)               
                elif trial=='experiment':
                    os.chdir(output_folder)
                    dg2[target] = pd.read_csv('{}_{}_{}_{}_unsmoothed.csv'.format(trial,workvec[workstatus],target,'Q2'), index_col=0)           
                    
    for target in serialvec2:
        b_yray = dg1.loc[:,target]
        e_yray = dg2.loc[:,target]
        if e_yray.max()/1000.<500.:
            scale = 1./1000.
            ylabel_option = 1
        else:
            scale = 1/(1000.*1000.)
            ylabel_option = 2    
            
        fig, ax = plt.subplots()
        xrelabel = [i*5+15 for i in dg1.index] # xrelable = [20,25,...,85]
        line1, = ax.plot(xrelabel, b_yray*scale, '-', label='baseline tax system')
        line2, = ax.plot(xrelabel, e_yray*scale, '-', label='alternative system') 
        ax.fill_between(xrelabel, b_yray*scale, e_yray*scale, where=b_yray>=e_yray, facecolor='red', alpha=0.2, interpolate=True, label='baseline dominates')
        ax.fill_between(xrelabel, b_yray*scale, e_yray*scale, where=b_yray<e_yray, facecolor='blue', alpha=0.2, interpolate=True, label='alternative dominates')
        
        if ylabel_option == 1:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%2.fk')) # if use percent sign (%), the option is '%i%%'; use dollar sign '\$%2.1fk'                                
            plt.figtext(0.015,0.7,'(thousands of 2013 US dollars)', fontsize=12, color=ax.xaxis.label.get_color(),rotation='vertical')
        elif ylabel_option ==2:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%2.1fM')) # if use percent sign (%), the option is '%i%%'; use dollar sign '\$%2.1fM'   
            plt.figtext(0.015,0.7,'(millions of 2013 US dollars)', fontsize=12, color=ax.xaxis.label.get_color(),rotation='vertical')
        
        fig.suptitle('{} {}'.format('2nd', 'quartile {} {}'.format(workstr[workstatus],serialstr2[serialvec2.index(target)]).title()))
        if (target=='ast'):
            ax.legend(loc='upper left', fancybox=True, shadow=True)      
        elif (target=='hom'):
            ax.legend(loc='lower right', fancybox=True, shadow=True)
                
        major_ticks = np.arange(25, 86, 10)                                              
        minor_ticks = np.arange(25, 86, 5)             
        ax.set_xticks(major_ticks)                                                       
        ax.set_xticks(minor_ticks, minor=True)      
            
        # automatically set the minor ticks on yaxis based on the major ticks
        minor_locator = AutoMinorLocator(2)
        ax.yaxis.set_minor_locator(minor_locator)
        plt.grid(which='minor')
                                                     
        ax.grid(which='minor', alpha=0.2)                                                
        ax.grid(which='major', alpha=0.5)       
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['grid.alpha'] = 1
        plt.rcParams['grid.color'] = "#cccccc"  
        ax.set_xlabel('age',fontsize=12)
        #ax.set_title('({})'.format(trialstr[trialvec.index(trial)]),fontsize=11)
        plt.grid(True)        
        os.chdir(plot_folder)
        plt.savefig('{}_{}_{}_sys_compare.png'.format(workvec[workstatus],target,'Q2'), dpi=150, transparent=True)
        plt.savefig('{}_{}_{}_sys_compare.svg'.format(workvec[workstatus],target,'Q2'), format="svg")        
        plt.show()    
        
# ----------------------------------------------------------------------------- # 2.5.2 V. Compare baseline model with real data (with shaded area illustration)
# compare baseline model with the real data

for workstatus in [0,1]:    
    for target in serialvec2:
        #workstatus = 0
        #target = 'ast'
        os.chdir(output_folder) 
        df1 = pd.read_csv('{}_{}_{}_{}_unsmoothed.csv'.format('benchmark',workvec[workstatus],target,'Q2'),index_col=0)
        print('{}_{}_{}_{}_unsmoothed.csv'.format('benchmark',workvec[workstatus],target,'Q2'),workstatus)
        xrelabel = [i*5+15 for i in df1.index]
        df1.index = xrelabel
        if target=='ast':
            dg1 = pd.concat([df1], axis=1)  
        else:
            dg1 = pd.concat([dg1, df1], axis=1)
        
        os.chdir(output_folder)
        temp = np.where(target=='ast',assetvec[1],assetvec[0])
        df2 = pd.read_csv('smooth_{}_{}_{}.csv'.format(workvec[workstatus],temp,'2qt'), index_col=0)    
        df2.columns = [target]        
        print('smooth_{}_{}_{}.csv'.format(workvec[workstatus],temp,'2qt'), workstatus)
        if target=='ast':
            dg2 = pd.concat([df2], axis=1)
        else:
            dg2 = pd.concat([dg2, df2], axis=1)
            
    fig, ax = plt.subplots(1,2)
    plt.tight_layout()
    ylabel_option = np.where(dg2['ast'].max()/1000.<500., 1, 2)
    scale = np.where(ylabel_option==1, 1/1000.,1/1000000.)
    ax[0].plot(dg2.index, dg2['ast']*scale, '-', label='financial wealth')
    ax[0].plot(dg2.index, dg2['hom']*scale, '-', label='home equity')   
    
    if workstatus==0:
        fig.suptitle('Age profile of worker household asset holdings'.title())
    elif workstatus==1:
        fig.suptitle('Age profile of business household asset holdings'.title())
        
    if ylabel_option == 1:
        ax[0].yaxis.set_major_formatter(plt.FormatStrFormatter('\$%2.fk')) # if use percent sign (%), the option is '%i%%'                                
        #ax[0].set_title('(thousands of 2013 US dollars)', fontsize=12, color=ax[0].xaxis.label.get_color())
        #plt.figtext(0.015,0.7,'(thousands of 2013 US dollars)', fontsize=12, color=ax[0].xaxis.label.get_color(),rotation='vertical')
    elif ylabel_option ==2:
        ax[0].yaxis.set_major_formatter(plt.FormatStrFormatter('\$%2.1fM')) # if use percent sign (%), the option is '%i%%'  
        #ax[0].set_title('(millions of 2013 US dollars)', fontsize=12, color=ax[0].xaxis.label.get_color())
        #plt.figtext(0.015,0.7,'(millions of 2013 US dollars)', fontsize=12, color=ax[0].xaxis.label.get_color(),rotation='vertical')    
    ax[0].set_title('SCF 1983-2013', fontsize=12, color=ax[0].xaxis.label.get_color())

    ylabel_option = np.where(dg1['ast'].max()/1000.<500., 1, 2)
    scale = np.where(ylabel_option==1, 1/1000.,1/1000000.) 
    ax[1].plot(xrelabel, dg1['ast']*scale, '-', label='financial wealth')
    ax[1].plot(xrelabel, dg1['hom']*scale, '-', label='home equity')
    if ylabel_option == 1:
        ax[1].yaxis.set_major_formatter(plt.FormatStrFormatter('\$%2.fk')) # if use percent sign (%), the option is '%i%%'                                
        #plt.figtext(0.015,0.7,'(thousands of 2013 US dollars)', fontsize=12, color=ax[1].xaxis.label.get_color(),rotation='vertical')
    elif ylabel_option ==2:
        ax[1].yaxis.set_major_formatter(plt.FormatStrFormatter('\$%2.1fM')) # if use percent sign (%), the option is '%i%%'   
        #plt.figtext(0.015,0.7,'(millions of 2013 US dollars)', fontsize=12, color=ax[1].xaxis.label.get_color(),rotation='vertical')        
    ax[1].set_title('Baseline model', fontsize=12, color=ax[1].xaxis.label.get_color())
    
    for panel in [0,1]:
        if (workstatus==1) & (panel==0):
            ax[panel].set_ylim(-0.15,1.1) # ====================================
        if (workstatus==0) & (panel==0):
            ax[panel].set_ylim(-58, 160)
            majorLocator = MultipleLocator(50)
            ax[panel].xaxis.set_major_locator(majorLocator)
            
        major_ticks = np.arange(25, 86, 10)                                              
        minor_ticks = np.arange(25, 86, 5)             
        ax[panel].set_xticks(major_ticks)                                                       
        ax[panel].set_xticks(minor_ticks, minor=True)      
        ax[panel].set_xlabel('age',fontsize=12)
        
        # automatically set the minor ticks on yaxis based on the major ticks
        #if panel==0:
        #    majorLocator = MultipleLocator(6)
        #    ax[panel].yaxis.set_major_locator(majorLocator)
        minor_locator = AutoMinorLocator(2)
        ax[panel].yaxis.set_minor_locator(minor_locator)
        plt.grid(which='minor')
        
        ax[panel].grid(which='minor', alpha=0.2)                                                
        ax[panel].grid(which='major', alpha=0.5) 
        ax[panel].axhline(y=0.,xmin=0,xmax=1,linestyle='--',c='k',linewidth=1, alpha=0.5)
             
    
    if workstatus==1:
        ax[0].legend(loc='upper left', fancybox=True, shadow=True)
        ax[1].legend().set_visible(False)
    else:
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(handles[::-1], labels[::-1], loc='upper left', fancybox=True, shadow=True)        
        ax[0].legend(loc='lower right', fancybox=True, shadow=True)
        ax[1].legend().set_visible(False)
        
    plt.figtext(0.05,0.01,'*measured in 2013 US dollars', color=ax[0].xaxis.label.get_color())    
    fig.subplots_adjust(left=0.1, top=0.875, bottom=0.15, wspace=0.25)        
    os.chdir(plot_folder)
    plt.savefig('{}_{}_ModelvData.png'.format(workvec[workstatus],'Q2'), dpi=150, transparent=True)
    plt.savefig('{}_{}_ModelvData.svg'.format(workvec[workstatus],'Q2'), format="svg")      
    plt.show()    
            

# ============================================================================= # 2.6 VI income groups statstics in policy experiments
epsilon = 1.e-8

def weighted_wealth(x,quant=0.5):
    return weighted_quantile(x['wel'], quant, x['den'])

def weighted_income(x,quant=0.5):
    return weighted_quantile(x['inc'], quant, x['den'])

def weighted_home(x,quant=0.5):
    return weighted_quantile(x['hom'], quant, x['den'])

def weighted_asset(x,quant=0.5):
    return weighted_quantile(x['ast'], quant, x['den'])

def group_ttx_mean(x):
    return weighted_mean(x['ttx'],x['den'])

def group_wel_mean(x):
    return weighted_mean(x['wel'],x['den'])

def group_inc_mean(x):
    return weighted_mean(x['inc'],x['den'])

def group_ast_mean(x):
    return weighted_mean(x['ast'],x['den'])

def group_hom_mean(x):
    return weighted_mean(x['hom'],x['den'])

# -----------------------------------------------------------------------------

for s in ['benchmark','experiment']:
    #s = 'benchmark'   
    # ---------- net worth groups --------------------------------------------- I.

    numbvec = [5,10,25,50,75,90,95,100] # we want to look at bottom 5%, 10%, ..., top 10% and top 5% net worth groups.
    os.chdir(output_folder)
    df = pd.read_csv('{}_df.csv'.format(s)) 
    df['wel'] = df['hom']+df['ast'] # column of net worth. 
    
    ans = DataFrame(0., index=numbvec, columns=['lb','ub']) # a frame to store upper and lower boundaries of each net worth bin.
    for x in numbvec: # compute all the upper boundaries of the (bin) division.
        ans.loc[x,'ub'] = weighted_quantile(df['wel'], x/100., sample_weight=df['den']) 
    ans.lb = ans.ub.shift(1) # shift should occure before any element-wise update. Trick!! So it has to be ahead of the following lines.
    ans.loc[numbvec[0],'lb'] = df['wel'].min() - epsilon # to expand the minimum lower boundary downward a little bit to ensure all the data are involved.
    bins_bound = list(ans.lb) # generate list as label list for pd.cut later.
    bins_bound.append(df['wel'].max()+epsilon) # since the bins for pd.cut has to be one element larger than label list, we do this...
    gid = pd.cut(df['wel'], bins=bins_bound, labels=numbvec, right=True) # geneate 10 equal distributed net worth groups.
    gid = gid.cat.codes # convert the categorical output into codes of 8, each of which refers to the group index.
    df['gid'] = gid # group index
    
    # Some summary statistics of the whole population, to be used later.
    df['wgt_ttx'] = df['den'] *df['ttx'] 
    tot_ttx = df['wgt_ttx'].sum() # total tax in the economy
    #
    dh = df.loc[df['k']>=1, df.columns]
    tot_ent_den = dh['den'].sum()
    
    # percentage of tax payment of all types of family # (1)
    # Since the computation of the poor and the rich works in opposite directions, we break the operation into two blocks.
    # ---- poor 5%, 10%, 25% and 50% (indexed as group 0, 1, 2 and 3) - block 1
    dh = df.loc[df['gid']<=3.0, df.columns] # 0, 1, 2 and 3, i.e, only those groups in the bottom of net worth distribution.
    grouped = dh.groupby('gid')
    temp1 = grouped['wgt_ttx'].sum()
    temp1 = temp1.cumsum()
    temp1 = DataFrame(temp1)
    temp1.index = ['B5%','B10%','B25%','B50%']
    temp1.columns = ['pct_ttx']
    temp1 = temp1/tot_ttx
    
    # size of entrepreneur # (2)
    # share of entrepreneurs in the whole entrepreneurial population # (3)
    cond = (df['gid']<=3.0) & (df['k']>=1)
    dh = df.loc[cond, df.columns]
    grouped = dh.groupby('gid')
    temp11 = grouped['den'].sum()
    temp11 = temp11.cumsum()
    temp11 = DataFrame(temp11)
    temp11.index = temp1.index
    temp11.columns = ['entmass'] 
    temp11['e1entpop'] = temp11/tot_ent_den
     
    # percentage of tax payment of all types of family # (4)
    # ---- rich 50%, 25%, 10% and 5% (indexed as group 4, 5, 6 and 7) - block 2
    dh = df.loc[df['gid']>=4.0, df.columns]
    grouped = dh.groupby('gid')
    temp2 = grouped['wgt_ttx'].sum()
    temp2 = temp2[::-1]
    temp2 = temp2.cumsum()
    temp2 = temp2[::-1]
    temp2 = DataFrame(temp2)
    temp2.index = ['T50%','T25%','T10%','T5%']
    temp2.columns = ['pct_ttx']
    temp2 = temp2/tot_ttx
    
    # size of entrepreneur # (5)
    # share of entrepreneurs in the whole entrepreneurial population # (6)    
    cond = (df['gid']>=4.0) & (df['k']>=1)
    dh = df.loc[cond, df.columns]
    grouped = dh.groupby('gid')
    temp22 = grouped['den'].sum()
    temp22 = temp22[::-1]
    temp22 = temp22.cumsum()
    temp22 = temp22[::-1]
    temp22 = DataFrame(temp22)
    temp22.index = temp2.index
    temp22.columns = ['entmass']
    temp22['e1entpop'] = temp22/tot_ent_den
    
    dmerge1 = pd.concat([temp1, temp2]) # divide by the total to get the percentage for each groups (B5%, B10% and so on).
    dmerge2 = pd.concat([temp11, temp22])
    dmerge  = pd.concat([dmerge1,dmerge2],axis=1)
    
    # average net worth # (7-8)
    # ---- average net worth ---- the poor 
    dmerge['avg_wel'] = np.nan
    for m in range(4): # [0,1,2,3] refers to temp1.index
        dh = df.loc[df['gid']<=m, df.columns]
        dmerge.loc[temp1.index[m], 'avg_wel'] = weighted_mean(dh['wel'], dh['den'])
    # ---- average net worth ---- the rich
    for m in range(4): # m ranges from 0 to 1 to refer to [4,5,6,7], i.e., temp2.index    
        dh = df.loc[df['gid']>=(m+4), df.columns]
        dmerge.loc[temp2.index[m], 'avg_wel'] = weighted_mean(dh['wel'], dh['den'])

    # average net worth, liquid asset, illiquid asset entrepreneur # (9-10)
    for y in [0,1]:
        jobcode = ['w','e'][y]
        for x in ['wel','hom','ast']:
            # ---- average net worth ---- the poor
            newcol = '{}_avg_{}'.format(jobcode,x)
            newcol2 = '{}_med_{}'.format(jobcode,x)
            dmerge[newcol] = np.nan
            dmerge[newcol2] = np.nan
            for m in range(4): # [0,1,2,3] refers to temp1.index
                cond = (df['gid']<=m) & (df['k']==y)
                dh = df.loc[cond, df.columns]
                dmerge.loc[temp1.index[m], newcol] = weighted_mean(dh[x], dh['den'])
                dmerge.loc[temp1.index[m], newcol2] = weighted_quantile(dh[x], 0.5, dh['den'])
            # ---- average net worth ---- the rich
            for m in range(4): # m ranges from 0 to 1 to refer to [4,5,6,7], i.e., temp2.index    
                cond = (df['gid']>=(m+4)) & (df['k']==y)
                dh = df.loc[cond, df.columns]
                dmerge.loc[temp2.index[m], newcol] = weighted_mean(dh[x], dh['den'])    
                dmerge.loc[temp2.index[m], newcol2] = weighted_quantile(dh[x], 0.5, dh['den'])

    # End of summary statistics (1. total tax share; 2. group average net worth) of net worth group                
    
    dmerge.to_csv('{}_net_worth_rich8poor.csv'.format(s)) # <------------------
    
    # net worth groups with job types ----------------------------------------- II.
    #
    numbvec = [x for x in range(20,101,20)] 
    df = pd.read_csv('{}_df.csv'.format(s)) 
    df['wel'] = df['hom']+df['ast'] # column of net worth. 
    df['wgt_ttx'] = df['den']*df['ttx']
    ans = DataFrame(0., index=numbvec, columns=['lb','ub']) # a frame to store upper and lower boundaries of each net worth bin.
    for x in numbvec: # compute all the upper boundaries of the (bin) division.
        ans.loc[x,'ub'] = weighted_quantile(df['wel'], x/100., sample_weight=df['den']) 
    ans.lb = ans.ub.shift(1) # shift should occure before any element-wise update. Trick!! So it has to be ahead of the following lines.
    ans.loc[numbvec[0],'lb'] = df['wel'].min() - epsilon # to expand the minimum lower boundary downward a little bit to ensure all the data are involved.
    bins_bound = list(ans.lb) # generate list as label list for pd.cut later.
    bins_bound.append(df['wel'].max()+epsilon) # since the bins for pd.cut has to be one element larger than label list, we do this...
    
    gid = pd.cut(df['wel'], bins=bins_bound, labels=numbvec, right=True) # geneate 10 equal distributed net worth groups.
    gid = gid.cat.codes # convert the categorical output into codes of 8, each of which refers to the group index.
    df['gid'] = gid # group index
    df['boss'] = np.where(df['k']==0, 0, 1)
    tot_ttx = df['den']*df['ttx']
    tot_ttx = tot_ttx.sum()
    
    grouped = df.groupby('gid')
    dg = DataFrame(grouped['wgt_ttx'].sum()/tot_ttx)
    #dg.index = range(1,len(dg)+1)
    dg.columns = ['pct_ttx']
    dg['avg_wel'] = DataFrame(grouped.apply(group_wel_mean))
    dg['avg_ast'] = DataFrame(grouped.apply(group_ast_mean))
    dg['avg_inc'] = DataFrame(grouped.apply(group_inc_mean))
    dg['avg_hom'] = DataFrame(grouped.apply(group_hom_mean))
    dg['ppl'] = grouped['den'].sum()
    dh = df.loc[df['k']==1,df.columns]
    tot_ent = dh['den'].sum()
       
    for y in [0,1]:
        jobcode = ['w','e'][y]
        dh = df.loc[df['k']==y,df.columns]
        grouped = dh.groupby('gid')
        # average and medain of wealth, home and asset.
        for x in ['wel','hom','ast']:
            newcol1 = '{}_avg_{}'.format(jobcode,x)
            newcol2 = '{}_med_{}'.format(jobcode,x)
            if x=='wel':
                dg[newcol1] = grouped.apply(group_wel_mean)
                dg[newcol2] = grouped.apply(weighted_wealth)
            elif x=='hom':
                dg[newcol1] = grouped.apply(group_hom_mean)
                dg[newcol2] = grouped.apply(weighted_home)                
            elif x=='ast':
                dg[newcol1] = grouped.apply(group_ast_mean)
                dg[newcol2] = grouped.apply(weighted_asset)  
        # job statstics
        cond = (df['k']==y) 
        dh = df.loc[cond,df.columns]
        grouped = dh.groupby('gid')
        dg['{}_ppl'.format(jobcode)] = grouped['den'].sum()
        dg['{}_in_group'.format(jobcode)] = dg['{}_ppl'.format(jobcode)]/dg['ppl']
    dg['e_in_e'] = dg['e_ppl']/tot_ent
    dg.to_csv('{}_wealth_groups.csv'.format(s)) # <----------------------------
              
    
    # ---- income groups ------------------------------------------------------ III.
    numbvec = [x for x in range(20,101,20)]   
    df = pd.read_csv('{}_df.csv'.format(s)) 
    df['wel'] = df['hom']+df['ast'] # column of net worth. 
    df['wgt_ttx'] = df['den']*df['ttx']    
    ans = DataFrame(0., index=numbvec, columns=['lb','ub']) # bin boundary
    for x in numbvec:
        ans.loc[x,'ub'] = weighted_quantile(df['inc'], x/100., sample_weight=df['den']) 
    ans.lb = ans.ub.shift(1) # this line has to be ahead of the next one.
    ans.loc[numbvec[0],'lb'] = df['inc'].min()-epsilon 
    bins_bound = list(ans.lb)
    bins_bound.append(df['inc'].max() + epsilon)
    
    gid = pd.cut(df['inc'], bins=bins_bound, labels=numbvec, right=True) 
    gid = gid.cat.codes
    df['gid'] = gid
    
    df['boss'] = np.where(df['k']==0, 0, 1)
    df['wel'] = df['hom']+df['ast'] 
    df['wgt_ttx'] = df['den']*df['ttx']
    tot_ttx = df['wgt_ttx'].sum()    

    grouped = df.groupby('gid')    
    dg = DataFrame(grouped['wgt_ttx'].sum()/tot_ttx)
    #dg.index = range(1,len(dg)+1)
    dg.columns = ['pct_ttx']
    dg['avg_wel'] = DataFrame(grouped.apply(group_wel_mean))
    dg['avg_ast'] = DataFrame(grouped.apply(group_ast_mean))
    dg['avg_inc'] = DataFrame(grouped.apply(group_inc_mean))
    dg['avg_hom'] = DataFrame(grouped.apply(group_hom_mean))
    dg['ppl'] = grouped['den'].sum()
    dh = df.loc[df['k']==1,df.columns]
    tot_ent = dh['den'].sum()
       
    for y in [0,1]:
        jobcode = ['w','e'][y]
        dh = df.loc[df['k']==y,df.columns]
        grouped = dh.groupby('gid')
        # average and medain of wealth, home and asset.
        for x in ['wel','hom','ast']:
            newcol1 = '{}_avg_{}'.format(jobcode,x)
            newcol2 = '{}_med_{}'.format(jobcode,x)
            if x=='wel':
                dg[newcol1] = grouped.apply(group_wel_mean)
                dg[newcol2] = grouped.apply(weighted_wealth)
            elif x=='hom':
                dg[newcol1] = grouped.apply(group_hom_mean)
                dg[newcol2] = grouped.apply(weighted_home)                
            elif x=='ast':
                dg[newcol1] = grouped.apply(group_ast_mean)
                dg[newcol2] = grouped.apply(weighted_asset)  
        # job statstics
        cond = (df['k']==y) 
        dh = df.loc[cond,df.columns]
        grouped = dh.groupby('gid')
        dg['{}_ppl'.format(jobcode)] = grouped['den'].sum()
        dg['{}_in_group'.format(jobcode)] = dg['{}_ppl'.format(jobcode)]/dg['ppl']
    dg['e_in_e'] = dg['e_ppl']/tot_ent    

    dg.to_csv('{}_income_grups.csv'.format(s)) # <-----------------------------
    
# ============================================================================= # 3.0 Nothing developed here.
    
    
    
    



# =============================================================================
#dh1[']
#grouped = dh1.groupby('t').
# -----------------------------------------------------------------------------
'''''
# Test of optimization with DataFrame data
df1 = DataFrame(np.repeat([1,2,3],3).reshape((3,3)),columns=['a','b','c'])
df2 = DataFrame(np.repeat([10,20,30],3).reshape((3,3)),columns=['a','b','c'])
def mulfunc(x,factor):
    a = x['a']*factor
    b = x['b']*factor
    c = x['c']*factor
    return a+b+c
def objfunc(x):
    df1['d'] = df1.apply(mulfunc,factor=2.,axis=1)
    df2['d'] = df2.apply(mulfunc,factor=x,axis=1)
    return df1['d'].subtract(df2['d']).sum()

print scipy.optimize.brentq(objfunc, 0., 100.) # objective function has to be continuous.
'''''

'''''
# elementwise operations test
df1['d'] = df1.apply(mulfunc,factor=2.,axis=1)
df2['d'] = df2.apply(mulfunc,factor=4.,axis=1)
sr3 = df1['d'].subtract(df2['d'])**2.
sr4 = sr3.sum()
sc5 = sr3.sum()**0.5
'''''

'''''
# Template of optimization
def f(x, a, c):
    return a*x - c
print scipy.optimize.brentq(
    f, 0.0, 100.0, args=(2,3)) # find a root in the interval [0,100], parameters n=77/27, a=1, b=1, c=10
'''''