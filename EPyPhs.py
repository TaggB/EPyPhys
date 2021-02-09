#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:29:48 2020
@author: benjamintagg
Department of Neuroscience, Physiology, and Pharmacology, UCL, UK
------------------------------DESCRIPTION-----------------------
This software is designed for analysis of electrophysiology records from piezo
actuator-driven fast jump protocols; or other excised-patch electrophysiology.

Sections:
    - ABF file import
    - Data cleaning: baselining, wave inspection and removal.
    - Analysis
    -   Pairwise Non-Stationary fluctuation (noise) analysis (Sigg,1997;Heinemann and Conti,1997)
    -   NBQX unbinding (Rosenmund et al., 1998; Coombs et al., 2017)
    -   Deactivation and Desensitisation kinetics
    - Embedded functions
    
--------------------------------LICENSE-------------------------
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"),to deal in
the Software without restriction, including without limitation the rights to
use,copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject tothe following conditions:

The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT,TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-----------------------------------------------------------------
-------------------------------REQUIREMENTS----------------------
-----------------------------------------------------------------
The anaconda distirbution of python is recommended. Otherwise:
    
Numpy package for scientific computing:                 https://numpy.org/
Pandas package for DataFrames:                          https://pandas.pydata.org/pandas-docs/stable/index.html
Matplotlib 2D plotting library:                         https://matplotlib.org/index.html
Pythonabf:                                              https://pypi.org/project/pyabf/
Seaborn data visualisation module:                      https://seaborn.pydata.org/
Scipy for scientific computing (curve fitting)          https://www.scipy.org/
lmfit for parabolic curve fitting                       https://lmfit-py.readthedocs.io/en/0.9.12/installation.html

-------------------------------CREDIT----------------------------
pyabf from Swhardan, used to import .abf (axon binary file) format
https://github.com/swharden/pyABF/
"""
#-----------------------------------------------------------------------------#
#_______________________________Configuration_________________________________#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import pyabf
import seaborn as sns
import scipy as sp
import os
print("Please set Ipython display to automatic in preferences to use interactive graphs. This will require a kernel restart and has been tested using IPython 7.13 with the Spyder IDE (4.1). This is required for interactive plotting")
print("\n \n \n WELCOME TO JAM-PACKED 1.0. \n for Jazzy Analysis of Macroscopic responses")



#_____________________________________________________________________________#
#___________________Data import, screening, merging, and cleaning______________________#

plt.style.use("ggplot")
print("\n \n Currently using ggplot as default plotting environment")

def load_data(path,panelsizein = 20,leg = False):
    """load_data accepts a trace files of type ABF or ABF2 for inspection
    Trace files can either be specified as a path of type(path)=str or can be 
    dragged between the open brackets.Panelsizein gives width, height of graph panel 
    in inches and is 20 by deafult. 
    
    When leg = False, no legend is plotted
    
    Data is loaded into a Pandas dataframe, where the index gives the time point
    in seconds. Each column [0:end] is a successive sweep of the record.
    
    Data is NOT baselined during this process
    """
    if type(path) !=str:
        return("A path should be specified as a string")
    open_tip = pyabf.ABF(path)
    sampling_freq = open_tip.dataRate # sampling rate in Hz.
    for sweepnum in open_tip.sweepList:
        open_tip.setSweep(sweepnum)
        if sweepnum ==0: # initialising pandas DataFrame with indexes as times for sweep 0
            data = pd.DataFrame(data=open_tip.sweepY,index=open_tip.sweepX)
        else: # adding remaining sweeps to the DataFrame object as a new column
            data[sweepnum] = open_tip.sweepY
    # plotting figures individually to allow quality control of waves imported by inspection at sufficient resoltion ##:NB subplotting loses resolution for sweep number identification
    figure_1,axes_1 = plt.subplots(1,figsize = (panelsizein,panelsizein)) #creation of subplots
    #axes_1=figure_1.add_axes([1,1,1,1]) --------> deprecated
    title1 = 'Current distribution from responses of {}'.format(path.split('/')[-1])
    axes_1.set_title(title1)
    data.plot.box(legend=False,ax = axes_1) # boxplots of each wave, side by side
    fig2,axs2 = plt.subplots(figsize = [panelsizein,panelsizein])
    for sweepNumber, value in enumerate(open_tip.sweepList):
        open_tip.setSweep(sweepNumber)
        axs2.plot(open_tip.sweepX,open_tip.sweepY,label = "{}".format(sweepNumber))
    title2 = 'Overlaid Sweeps from the open-tip response of {}'.format(path.split('/')[-1])
    axs2.set_title(title2)
    figure_1.show()
    figure_1.tight_layout()
    if leg:
        fig2.legend()
    print("Sampling freqency was", sampling_freq/1000,"kHz")
    print("Data is now in DataFrame format. Remove waves with variable_name = variable_name.drop(columns = [comma-separated numbers])")
    print("Data should be baselined before use. Type help(baseline)")
    print("For merging data, see merge and merge_protocols")
    print("For alignment, see align_at_t, align_at_a, or normalise_currents")
    print("For subtracting waves, see subtract_avg")
    return(data)
    

def merge(records_to_merge_as_dataframes):
    """Enter records to merge as a tuple of dataframes,e.g. merge((rec1,rec2))
    and returns a single dataframe containing all data."""
    merged = records_to_merge_as_dataframes[0]
    for item in records_to_merge_as_dataframes[1:]:
        merged = np.append(merged,item.to_numpy(),axis=1)
    record = pd.DataFrame(merged, index = records_to_merge_as_dataframes[0].index)
    return(record)
    
        
def cleaning_data(data,panelsizein=20,grpsize = 20):
    """Accepts a pandas dataframe of type produced in load_data, and returns
    used sweeps and removed sweeps of that file"""
    ## Now to plot all included sweeps as average in one panel,
    # split waves into groups of 20 with offset,showing average of that group as an insert to that panel
    multiplot_clean_data(data,groupsize = grpsize,separate_avg = False,panelsizein = panelsizein)
    print('Please Inspect the Graphs produced during loading.\nIdentify any to remove. For a reminder of cleaning practice, enter how_to_clean()')
    do_data_cleaning = input("Do you wish to remove any sweeps? ('y'/''n'') >")
    if 'y' in do_data_cleaning or 'Y' in do_data_cleaning:
        while True:
            used_sweeps,removed_sweeps = clean_data(data,remove_sweeps = True)
            continue_data_cleaning = input("Do you wish to remove more sweeps? ('y'/'n'>")
            if 'n' in continue_data_cleaning or 'N' in continue_data_cleaning:
                break
                return(used_sweeps,removed_sweeps)
            else:
                used_sweeps,removed_sweeps = clean_data(data,remove_sweeps = True,last_removed =removed_sweeps )
                return(used_sweeps,removed_sweeps)
    else:
        used_sweeps = data
        removed_sweeps = []
        return(used_sweeps,removed_sweeps)
    
def baseline(response_dataframe, individual = True, end_baseline = 20):
    """Performs baseline subtraction of data in a pandas DataFrame
    
    When individual is True, each sweep is subtracted from its own baseline.
    When individual is false, each sweep is subtracted from the average
    baseline. Baseline is specified from t = 0 to t = end_baseline (in ms).
    By default, this is 20ms."""
    if individual:
        baseline_each_wave = response_dataframe[response_dataframe.index<(end_baseline/1000)].mean(axis=0) # mean baseline for each sweep
        baselined_waves = response_dataframe.copy(deep = True)#creating deep copy of response dataframe
        for item, value in enumerate(baselined_waves): # subtracting baseline, enumeration added to avoid conflict in zero base
            baselined_waves.iloc[:,item] = baselined_waves.iloc[:,item] - baseline_each_wave.iloc[item] # problematic as
        return(baselined_waves)
    elif not individual:
        avg_baseline = response_dataframe[response_dataframe.index<(end_baseline/1000)].mean().mean() # mean of all baselines
        baselined_waves = response_dataframe.copy(deep = True)
        for item in baselined_waves: # subtracting baseline
            baselined_waves.iloc[:,item] = baselined_waves.iloc[:,item] - avg_baseline
        return(baselined_waves)

def split_sweeps(response_dataframe,times_to_split):
    """Splits each sweep at times_to_split (ms) to return a list of size x of 
    ordered sweep fragements as dataframes. size x = size(times_to_split)+1
    
    baselinining is performed on the wave before splitting.
    times to split can either be a single value or a tuple of values,
    excluding t =0 and t = max(t)
    
    e.g.
    first_split,window,remainder = split_sweeps(data,times_to_split = (500,800))
    to split into three sets 0:500, 500:800, 800:end
    
    """
    fragments = []
    nfragments = np.size(times_to_split)+1 # number of fragments produced from sweep cleave at this time 
    if nfragments ==2:  #if only splitting at one time poin
        times_to_split =times_to_split/1000
        fragments.append(response_dataframe[response_dataframe.index < times_to_split]) #first fragment, until t = times_to_split
        fragments.append(response_dataframe[response_dataframe.index >= times_to_split]) # second fragment
    else:
        times_to_split = [i/1000 for i in times_to_split] # converting ms to s
        times = np.zeros(nfragments+1) # +1 to account for zero
        times[-1] = response_dataframe.index[-1] # setting final time
        times[1:-1] = times_to_split # setting split times
        for item in np.arange(np.size(times)-1): # for each fragment
            low_threshold = times[item]
            up_threshold = times[item+1]
            fragments.append(response_dataframe[(response_dataframe.index >= low_threshold) & (response_dataframe.index <=up_threshold)])
    return(fragments)
    
def subtract_avg(waves,waves_to_subtract):
    """subtracts the average of waves_to_subtract from each wave of waves"""
    waves_to_subtract = waves_to_subtract.mean(axis=1)
    subtracted_waves = waves.subtract(waves_to_subtract,axis=0)
    return(subtracted_waves)

def merge_protocols(tuple_of_dataframes):
    """Pools sweeps together from different protocols. For some recording software,
    increment-based step protocols (e.g. increase Vm by +10 each sweep) necessarily
    generate one record per protocol. This function merges sweeps based on order
    
    e.g. sweep one from record one will form a dataframe with sweep one of all other records
    
    Records must be entered as a tuple, e.g. merge_protocols((dataframe1,dataframe2))
    
    This is then returned as a single dataframe with each multindexed column representing
    a record (merged.columns.levels) level=0 and a sweep in level =1.
    
    
    This information can be accessed using pandas in-built .xs or .groupby methods
    e.g.1
    To access complete information:
        merged.xs(1,axis=1,level=0) will provide all sweeps from record 1
        merged.xs(3,axis=1,level=1) will provide sweep 3 from all records
    e.g.2 
    Mean current of sweeps across protocol iterations:
    Mean(Record1Sweep1,Record2Sweep1 etc) using
        merged.groupby('sweep',axis=1,level=1).mean()
        or to average by record, set level = 1.
                                                        
    """
    keys = {record:record for record in range(len(tuple_of_dataframes))}
    merged = pd.concat(tuple_of_dataframes,axis=1,keys=keys)
    merged.columns.names = 'record','sweep'
    return(merged)

def align_at_t(data,make_zero=True):
    """Aligns each wave at a time point selected for each wave (column)
    if make_zero is True (default), this time point is set to t= 0
    (and the waveform before this point is absent from the return)
    
    returns aligned waveform.
    
    if make_zero = false, the waves a shifted to align at the desired point.
    In this case, returns aligned waveform and newtimes.
    Data for each wave can be accessed,e.g. for plotting, using:
        aligned_waveform.loc[newtimes[:,wavenum],wavenum].plot()
    """
    print("For each sweep, click on the graph for a time point to align. Selections can be removed by right click. Press enter for final selection on each sweep")
    times = np.zeros(data.shape[1])
    # selecting t to align
    for item in np.arange(data.shape[1]):
        alfig,alaxs = plt.subplots()
        alaxs.plot(data.iloc[:,item])
        binding_id = plt.connect('motion_notify_event', on_move)
        plt.connect('button_press_event', on_click)
        alaxs.set_title("Left Click to select t. Right click to remove. Press enter when finished")
        points = plt.ginput(-1,0)
        point = points[0][0]
        plt.show()
        times[item] = point
        plt.close(alfig)
    ### making aligned 't' the average of all t, or zero if make_zero = True
    if make_zero:
        align = data.copy(deep= True)
        # beginning each wave at selected t
        for item in np.arange(data.shape[1]):
            align.iloc[:,item][align.index<times[item]] = np.nan
            # setting first point to t = 0
            arr = align.iloc[:,item][align.iloc[:,item].notna()].to_numpy()
            align.iloc[:np.size(arr),item] = arr
            align.iloc[np.size(arr):,item] = np.nan # removing residual values
        plt.plot(align)
        return(align)
    else: # if simply want to align entire wave about point, then offset each wave to this point
        shifttimes = np.mean(times) - times
        newtimes = np.zeros([data.index.size,data.shape[1]])
        # creating new index
        for item in np.arange(data.shape[1]):
            newtimes[:,item] = data.index
            newtimes[:,item] = newtimes[:,item]+shifttimes[item]
            newindex = np.sort(np.unique(newtimes.flatten()))
        aligned=pd.DataFrame([],index=newindex) # creating new df with the new index
        # aligning waves
        for item in np.arange(data.shape[1]):
            shifted_wave = pd.Series(data.iloc[:,item].to_numpy(),index=newtimes[:,item])
            aligned[item] = shifted_wave
            plt.plot(aligned.loc[newtimes[:,item],item])
        return(aligned,newtimes)

def align_at_a(data,make_zero = True):
    """Aligns each wave at an amplitude selected for each wave
    
    if make_zero is True (default), this amplitude point is set to 0
    if make_zero is True, the value for the amplitude point is made the average for all points
        
    """
    amplitudes = np.zeros(data.shape[1])
    # selecting t to align
    for item in np.arange(data.shape[1]):
        alfig,alaxs = plt.subplots()
        alaxs.plot(data.iloc[:,item])
        binding_id = plt.connect('motion_notify_event', on_move)
        plt.connect('button_press_event', on_click)
        alaxs.set_title("Left Click to select t. Right click to remove. Press enter when finished")
        points = plt.ginput(-1,0)
        point = points[0][1]
        plt.show()
        amplitudes[item] = point
        plt.close(alfig)
    if make_zero:
        for item in np.arange(data.shape[1]):
            data.iloc[:,item] = data.iloc[:,item] - amplitudes[item]
            return(data)
    else:
        differences = amplitudes.mean() - amplitudes
        for item in np.arange(data.shape[1]):
            data.iloc[:,item] = data.iloc[:,item] + differences[item]
        return(data)
    
def normalise_currents(data,make01=True,poscurrent=False):
    """normalises current amplitudes to the peak of a current defined within a time window
    returns data where the peak is normalised
    
    if make01 = False, peak current amplitudes are normalised on their current scale
    if make01 = True (Default), they are normalised inthe range 0,1
    
    poscurrent is used for make01 to decide whether to normalise to largest
    positive (poscurrent=True) or largest negative current (=False)
    """
    normfig,normaxs = plt.subplots()
    normaxs.plot(data)
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    normaxs.set_title("Left Click to select start t. Right click to remove. Press enter when finished")
    points = plt.ginput(-1,0)
    point = points[0][0]
    plt.show()
    begin = point
    plt.close()
    normfig,normaxs = plt.subplots()
    normaxs.plot(data)
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    normaxs.set_title("Left Click to select end t. Right click to remove. Press enter when finished")
    points = plt.ginput(-1,0)
    point = points[0][0]
    plt.show()
    end = point
    plt.close()
    described = data[(data.index>=begin) & (data.index<=end)].describe()
    amplits= np.zeros(data.shape[1])
    for item in np.arange(data.shape[1]):
        #find the amplitude of the peak
        amplits[item]=described[item]['max']-described[item]['min']
        # normalising all peaks to max amplitude peak
    normalised = data*(amplits/amplits.max())
        # reselecting window of interest
    window = normalised[(data.index>=begin) & (data.index<=end)]
        # making minimum current = 0
    window = window - window.iloc[0,:]
        # optional: converting from absolute to 0,1 scale
    if make01:
        if poscurrent:
            normalised = window/window.max().max()
        else:
            normalised = window/window.min().min()
    return(normalised)


    
        
#-----------------------------------------------------------------------------    
#######_________________________ANALYSIS METHODS_________________________#####

def NSFA_optimal(dataframe,binrange = [5,20],parabola = True,Vrev = 0,voltage = False,open_tip_response = False,start_time = False,end_time = False,background_start = False,background_end = False):
    """Detects the optimal number of bins over binrange based on minimisation
    of the product of 1SD errors for N and i using binned pairwise NSFA,
    and then peforms the pairwise NSFA using optimal number of bins as:
   
    
    Performs Sigg (1997) style pairwise non-stationary fluctuation analysis between
    two points of interest.
    Variance of the noise is calculated for N waves as:
        Variance = (2/N)* (yi-yu)^2
        Where yi = 0.5*(xi-xu): 
            The average difference of each pairwise noise trace for each isochrone
            minus mean of all pairwise average differences for each isochrone
        As such, the varaince measured is the variance of noise between successive
        wave pairs at each isochrone.
        
    N and i are then fit using:
        Varaince = i*I - (I^2/N) + Background_varaince
            Where I = mean current at each isochrone.
            Background-varaince is the pairwise variance of noise (as calculated
            above) when the periodic background signal is convolved through
            the interval of interest.
    
    Graphs are produced (see Returns:)

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Containing data for all sweeps
    num_bins : list of two values
        A range of possible number of bins. The number of bins used for the final
        analysis is based on the minimisation of 1SD erros for each fit in this range
        
        The Default is [5,20].
    Vrev : 
        Reversal potential to calculate single channel condutance. The default is 0.
    parabola : True/False
        When True, data is simulated from maximum current:2* maximum current
        to 'extend' the fit. The default is True.
    voltage :
        voltage used to calculate single channel conductance. The default is False.
        When False, single channel current, but not conductance, is returned.
    open_tip_response : Pandas DataFrame
        Responses of an open_tip_response. If False, the first point is selected
        from the response of the experimental trace in dataframe.
        The default is False.
    start_time (s) : 
        Start time for interval in which to perform pairwise analysis. When
        False, user is asked to select from graph. The default is False.
    end_time (s) : T
        End time for interval in which to perform pairwise analysis. When
        False, user is asked to select from graph. The default is False.
    background_start (ms) : TYPE, optional
        Time from which to take background interval. The default is False = 0.
    background_end (ms)
        Time until which to take background interval. The default is False = 20

    Returns
    -------
    Graphs:
        Graph 1: mean current for each isochrone in the interval vs variance of the CURRENT
            + 1SD errors for variance of the current.
        Graph 2: The binned (or not) fit with N and i; the background corrected
            mean current vs mean varaince of the noise
        Graph 3: The binned fit + 1SD errors of fit
        Graph 4: If Parabola = True, then the fit is extended beyond the dataset.
                See Parbola arg

    """
    ####### config
    if not start_time: # if no start time is provided, see if an open-tip-response dataframe was given
        if not np.any(open_tip_response): # if not, allow onset time selection from experimental record
            start_time = get_open_tip_onset(dataframe)
        else:
            start_time = get_open_tip_onset(open_tip_response)
    if not end_time: # if no peak time is specified
        end_time = get_peak(dataframe) # obtained from graph
    if not background_start:
        background_start = 0
    if not background_end:
        background_end = 20 # baseline end in ms
    binrange = np.array(binrange)
    #### perofmring binned noise analysis with nbins
    SDerrorfit = np.zeros([2,binrange[1]-binrange[0]])
    for item, value in enumerate(np.arange(binrange[0],binrange[1],1)):
        _,_,_,SD_errors = NSFA(dataframe,num_bins = value,parabola = False, Vrev=Vrev,voltage = voltage,start_time = start_time,end_time = end_time,background_start = background_start, background_end = background_end,suppress_g =True)
        SDerrorfit[:,item] = np.product(SD_errors),value
    print("\n \n \n \n Ignore warnings")
    num_bins = np.where(SDerrorfit[0,:]==np.nanmin(SDerrorfit[0,:]))[0]
    ### no, minimisation of covaraince not great idea, as producing silly fit
    print("optimal Nbins = ",num_bins[0])
    N,i,P_open,removed = NSFA(dataframe,num_bins = num_bins[0],parabola = True, Vrev=Vrev,voltage = voltage,start_time = start_time,end_time = end_time,background_start = background_start, background_end = background_end,suppress_g =False)
    return(N,i,P_open,removed)
    

#### Might also try a Gaussian regression noise analysis, which may be able to
     # separate background from current component and thus more accurately extract fit
     # based on predicted maxima

def NSFA(dataframe,num_bins = 10,Vrev = 0,parabola = True,voltage = False,open_tip_response = False,start_time = False,end_time = False,background_start = False,background_end = False,suppress_g = False):
    """
    Performs Sigg (1997) style pairwise non-stationary fluctuation analysis between
    two points of interest.
    
    A Heinemann and Conti, 1997 QC check is performed, and any waves with
    RMSE variance > 7 in either the background or period of interest are removed.
    
    Variance of the noise is calculated for N waves as:
        Variance = (2/N)* (yi-yu)^2
        Where yi = 0.5*(xi-xu): 
            The average difference of each pairwise noise trace for each isochrone
            minus mean of all pairwise average differences for each isochrone
        As such, the varaince measured is the variance of noise between successive
        wave pairs at each isochrone.
        
    N and i are then fit using:
        Varaince = i*I - (I^2/N) + Background_varaince
            Where I = mean current at each isochrone.
            Background-varaince is the pairwise variance of noise (as calculated
            above) when the periodic background signal is convolved through
            the interval of interest.
    
    Graphs are produced (see Returns:)

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Containing data for all sweeps
    num_bins : int
        The number of bins with which to perform pairwise NSFA. The default is 10.
        When num_bins = False, a fit is attempted unbinned. This may or may not
        be appropriate.
        Using NSFA_optimal() instead will select this number base on the minimisation
        of the fit error. 
    Vrev : 
        Reversal potential to calculate single channel condutance. The default is 0.
    parabola : True/False
        When True, data is simulated from maximum current:2* maximum current
        to 'extend' the fit. The default is True.
    voltage :
        voltage used to calculate single channel conductance. The default is False.
        When False, single channel current, but not conductance, is returned.
    open_tip_response : Pandas DataFrame
        Responses of an open_tip_response. If False, the first point is selected
        from the response of the experimental trace in dataframe.
        The default is False.
    start_time (s) : 
        Start time for interval in which to perform pairwise analysis. When
        False, user is asked to select from graph. The default is False.
    end_time (s) : T
        End time for interval in which to perform pairwise analysis. When
        False, user is asked to select from graph. The default is False.
    background_start (ms) : TYPE, optional
        Time from which to take background interval. The default is False = 0.
    background_end (ms)
        Time until which to take background interval. The default is False = 20
    suppress_g : True/False
        Suppress graphs. The default is False.

    Returns
    -------
    When suppress_g = False:
        N,i,P_open 
    When suppress_g = True:
        N_i_P_open, one standard deviation errors for N and i, removed waves
    Graphs:
        Graph 1: mean current for each isochrone in the interval vs variance of the CURRENT
            + 1SD errors for variance of the current.
        Graph 2: The binned (or not) fit with N and i; the background corrected
            mean current vs mean varaince of the noise
        Graph 3: The binned fit + 1SD errors of fit
        Graph 4: If Parabola = True, then the fit is extended beyond the dataset.
                See Parbola arg

    """
    ### config ###
    if not start_time: # if no start time is provided, see if an open-tip-response dataframe was given
        if not np.any(open_tip_response): # if not, allow onset time selection from experimental record
            start_time = get_open_tip_onset(dataframe)
        else:
            start_time = get_open_tip_onset(open_tip_response)
    if not end_time: # if no peak time is specified
        end_time = get_peak(dataframe) # obtained from graph
    if not background_start:
        background_start = 0
    if not background_end:
        background_end = 20 # baseline end in ms
    background_end = background_end/1000
    ### find all currents in window start_time:end_time
    peak_current = dataframe.loc[start_time:end_time,:] # taking peak current for all waves
    ### Repeating background period through peak current for background at isochrone
    base = dataframe.loc[background_start:background_end,:] # get background for each wave
    
    #Heinemann and Conti verification for background and periods of interest
    todiscardinterest = RMSDE_wave_check(peak_current)
    todiscardbackground = RMSDE_wave_check(base)
    to_remove = np.unique(np.append(todiscardinterest,todiscardbackground))
    base = base.drop(columns = list(to_remove))
    peak_current = peak_current.drop(columns = list(to_remove))
    #______ --- background variance for isochrones --_____ #
    ### number of repeats of periodic baseline during peak current
    r_baseline = (peak_current.index[-1]-peak_current.index[0])/(base.index[-1]-base.index[0])
    # fraction of period through baseline variance when peak current starts
    baseline_start = (peak_current.index[0])%(base.index[-1])/(base.index[-1])
    # calculating background noise for all isochronous time points in peak current
    # from point when background starts, calculate background varaince for all wave pairs
    background_period = base[base.index>=(baseline_start*base.index[-1])].to_numpy()
    background_period = np.vstack((background_period,base[base.index <(baseline_start*base.index[-1])].to_numpy()))
    background = background_period
    for item in np.arange(int(np.floor(r_baseline))-1):
        background = np.vstack((background,background_period))
    # from end of last whole period of background, repeat the remaining fraction of the period
    background = np.vstack((background,background_period[:peak_current.shape[0] - np.size(background,0),:]))

    # initialising plot objects
    if not suppress_g:
        plt.style.use('ggplot')    
        raw_fig,raw_axs = plt.subplots()
        raw_axs.set_xlabel("I(pA)")
        raw_axs.set_ylabel("$\sigma^2$ (pA$^2$)")
        raw_axs.set_title("95% CI of $\sigma^2$(I) in raw data (no background correction) ")
        noise_fig,noise_axs = plt.subplots()
        noise_axs.set_xlabel("I(pA)")
        noise_axs.set_ylabel("$\sigma^2$ (pA$^2$)")
        fit_fig,fit_axs =  plt.subplots()
        fit_axs.set_xlabel("I(pA)")
        fit_axs.set_ylabel("$\sigma^2$ (pA$^2$)")
    ###
    # convert to numpy: looping faster than dealing with pandas means and var.
    currents = peak_current.to_numpy()
    # perform Sigg (1994) type noise analysis for all time points.
    # mean for each isochrone, for pair diffs holding 'noise traces' of wave pairs
    # and then caluclating variance of the noise
    mean_isochrone_current = np.mean(currents,axis = 1)
    pair_diffs = np.zeros([np.size(currents,axis=0),(np.size(currents,axis=1)-1)])
    back_diffs = np.zeros([np.size(currents,axis=0),(np.size(currents,axis=1)-1)])
    isochrone_differences = np.zeros([np.size(currents,axis=0),(np.size(currents,axis=1)-1)])
    for item in np.arange(np.size(currents,axis=1)-1): # for each first wave of a pair
        # get noise trace of each wave pair
        pair_diffs[:,item] = 0.5*(currents[:,item] - currents[:,item+1])
        # get background noise trace for each wave pair
        back_diffs[:,item] = 0.5*(background[:,item]-background[:,item+1])
    # get isochrone noise differences ^2 with list comprehension
    isochrone_differences=np.vstack([(pair_diffs[:,item]-np.mean(pair_diffs[:,:],axis=1))**2 for item in np.arange(np.size(pair_diffs,1))]).transpose()  
    # for background
    background_differences = np.vstack([(back_diffs[:,item]-np.mean(back_diffs[:,:],axis=1))**2 for item in np.arange(np.size(back_diffs,1))]).transpose()  
    # variance for each isochrone = (2/N) * sum(yi-yu)^2
    #for each isochrone i, and mean of isochrone, u, N is the number of waves (points along isochrone)
    variance =  (2/np.size(currents,axis=1))*np.sum(isochrone_differences,axis=1)
    # background varaince
    background = (2/np.size(background_differences,axis=1))*np.sum(background_differences,axis=1)
    # 95% confidence interval of varaince for each isochrone:
    # removing current-variance pairs for wrong sign
    conf_int = sp.stats.norm.interval(0.95,loc=(np.var(currents,axis=1)),scale=np.std((np.var(currents,axis=1))))   # here, use this plot for mean current at time t against variance at t (i.e. for each isochrone current vs variance)
    if not suppress_g:
        noise_axs.scatter(np.abs(np.mean(currents,axis=1)),(variance-background),marker='.',color='grey',label = ' |Isochrone mean (I)| vs $\sigma^2$(noise) - $\sigma^2$B(noise)')
        raw_axs.errorbar(np.abs(np.mean(currents,axis=1)),(variance+background),yerr = conf_int,marker='.',color='black',barsabove=True,errorevery=10,label = 'Isochrone I vs $\sigma^2$(I) + $\sigma^2$B 95% CI')
    ### here: a background plot over time?
    if not num_bins: # if no bins, perform fit to whole current varaince relationship.
        x = np.zeros([3,np.size(variance)])
        x[0,:] = mean_isochrone_current # absolute mean current value
        x[1,:] = background
        x[2,:] = variance
        # inverting current so max negative amplitude on right
        x = np.flip(x,axis=1)
        # removing current of wrong sign (e.g. +ve current in outside-out or vice versa)
        popt,pcov = sp.optimize.curve_fit(noise_func, x[:2,:], x[2,:])
        popt = np.abs(popt) # for outside_out, making vals absolute
        # plotting fit against raw data
        if not suppress_g:
            noise_axs.scatter(-x[0,:],noise_func(x[:2,:],-popt[0],-popt[1]),linestyle="--",color='red',label='fit,N={},i={}'.format(np.round(popt[0],2),np.round(popt[1],2)))
        #plotting errors of raw data
        # plot fit with errorbars
        # 1SD of errors
        perr = np.sqrt(np.diag(pcov))
        # using to get error of the fit
        sdefit = noise_func(x[:2,:],perr[0],perr[1])
        if not suppress_g:
            fit_axs.errorbar(-x[0,:],noise_func(x[:2,:],-popt[0],-popt[1]),yerr = sdefit,barsabove=True,errorevery= np.floor(np.size(currents,axis=0)/10), fmt="o",capsize = 5)
            fit_axs.set_title("Unbinned Fitted I vs $\sigma^2$ ")
            noise_axs.set_title("Unbinned fit vs raw data: ")
            noise_axs.set_xlim(left=0)
            noise_axs.set_ylim(bottom=0)


    else: # if binning, bin by mean_current of isochrone
        # such that each bin ctributes to amplitude of current equally.
        x = (np.vstack((mean_isochrone_current,background,variance))).transpose()        
        x = pd.DataFrame(x)
        x.index = peak_current.index
        x[3] = pd.cut(x[0],num_bins)
        x = x.sort_values(3)
        x = x.groupby(x[3]).mean()
        x = (x.to_numpy()).transpose()
        x = np.flip(x,axis=1)
        popt,pcov = sp.optimize.curve_fit(noise_func, x[:2,:], x[2,:])
        popt = np.abs(popt) # for outside_out, making vals absolute
        # plotting the fit 
        perr = np.sqrt(np.diag(pcov))
        # inverting current to get region of interest
        if not suppress_g:
            noise_axs.scatter(x = -x[0,:],y = noise_func(x[:2,:],-popt[0],-popt[1]),linestyle="--",color='red',label='fit,N={},i={}'.format(np.round(popt[0],2),np.round(popt[1],2)))
            noise_axs.set_title("Binned fit (nbins={}) vs raw data".format(num_bins))
            noise_axs.set_xlim(left = 0)
            noise_axs.set_ylim(bottom = 0)

        sdefit = noise_func(x[:2,:],perr[0],perr[1])
        if not suppress_g:
            fit_axs.errorbar(np.abs(-x[0,:]),np.abs(noise_func(x[:2,:],-popt[0],-popt[1])),yerr = sdefit,barsabove=True, fmt="o",capsize = 5,color='black')
            fit_axs.set_title("Binned fit +- 1SD Errors of fit")
            fit_axs.set_xlim(left = 0)
            fit_axs.set_ylim(bottom = 0)


    if parabola:
        if not suppress_g:
            parabfig,parabaxs = plt.subplots()
            parabaxs.set_xlabel("I")
            parabaxs.set_ylabel("$\sigma^2$ (pA$^2$)")
            # simulating data: double current
            max_current = -(np.max(-x[0,:]))
            sim_current = np.linspace(max_current,2*max_current,np.size(x[0,:]))
            # need to simulate the variance for greater current values
            sim_curr_back = np.vstack((sim_current,x[1,:]))
            parabaxs.scatter(-x[0,:],noise_func(x[:2,:],-popt[0],-popt[1]),label = 'Parabolic fit to data',color='black')
            parabaxs.plot(-sim_current,noise_func(sim_curr_back,-popt[0],-popt[1]),linestyle="--",color='red',label='fit of noise parabola to simulated data,N={},i={}'.format(np.round(popt[0],2),np.round(popt[1],2)))
            parabaxs.legend()
            parabaxs.set_title("Fit of simulated and experimental data")
            parabaxs.set_ylim(bottom=0)
            parabaxs.set_xlim(left=0)
            parabfig.tight_layout()
            # simulating data for rest of curve
    SD_errors = np.sqrt(np.diag(pcov))
    N = popt[0]
    i = popt[1]
    P_open = np.max(-x[0,:])/(N*i)
    if not suppress_g:
        noise_axs.legend()
        raw_axs.legend()
        noise_fig.tight_layout()
        raw_fig.tight_layout()
        fit_fig.tight_layout()
        print("Waves {} were removed (background or ROI > 7RMSE)".format(to_remove))
        if not voltage:
            print('Fitted N = {} +-{},i = {} +-{},P_open = {}'.format(np.round(N,2),np.round(SD_errors[0],2),np.round(i,2),np.round(SD_errors[1],2),np.round(P_open,2)))
        else:
            g = i/(voltage-Vrev)
            print('Fitted N = {} +-{},i = {} +-{},P_open = {},gamma_mean = {}'.format(np.round(N,2),np.round(SD_errors[0],2),np.round(i,2),np.round(SD_errors[1],2),np.round(P_open,2),np.round(g,2)))
        if not parabola:
            print("If you wish to extend the parabola, call again with parabola = True. This can be useful to see whether a fit is adequate, and is particularly useful after binning")
        return(N,i,P_open,to_remove)
    else:
        return(N,i,P_open,SD_errors)


def NBQX_unbinding(dataframe,open_tip_response,give_onset = False,onset_lo_lim =False,onset_up_lim=False,give_peak = False):
    """
    Using Rosenmund et al., 1997 method of calculating subunit occupancy
    during unbinding of NBQX during saturating agonist concentrations.

    Fitted with a monoexponential Hodgkin-Huxley equation to calculate m,
    the exponent,corresponding the number of rate-limiting transitions (i.e.
    the number of agonist binding events required for channel opening. 
    
    Fitting is performed between onset and peak by user selection or
    input Args.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Should be baselined and cleaned. Containing current data, where
        each wave is a column of an indexed dataframe (produced by load_data).
        For removal of piezo artefacts, baseline until time of onset produced
        by an open_tip.
    open_tip_response : Pandas DataFrame
        The open_tip data. Should be specified. The default is False.
    give_onset(s) : 
        Time in seconds for the open-tip onset to use. When False, User
        selects from graph. The default is False.
    onset_lo_lim(s) : 
        If user selects from graph, LEFT x axis limit.
        Useful for expanding view to get open-tip onset
        The default is False.
    onset_up_lim(s) : TYPE, optional
        If user selects from graph, RIGHT x axis limit.
        Useful for expanding view to get open-tip onset
        The default is False.
    give_peak(s): specifies end point to fit until
    
    Returns
    -------
    max current(mean of the 100th bin), Tau_NBQX_unbinding, m, 10-90% rise time

    """
    
    if not give_onset:
        if not onset_lo_lim:
            onset = get_open_tip_onset(open_tip_response) # call embedded functions for user selected open_tiponset time and peak time
        else:
            onset = get_open_tip_onset(open_tip_response,lolimit = onset_lo_lim ,uplimit = onset_up_lim)
    else:
        onset = open_tip_response.index[open_tip_response.index>=give_onset][0]
    if not give_peak:
        peak = get_peak(dataframe)
    else:
        peak = give_peak
    peak_current = dataframe.loc[onset:peak,:] # taking peak current for all waves
    binned_peak=pd.DataFrame()
    
    ##### Performing binning into 100 equally-sized bins by amplitude
    binned_peak[0] = peak_current.mean(axis=1)
    binned_peak[1] = pd.cut(binned_peak[0],100)
    binned_peak.reset_index(inplace=True) #setting t as index
    binned_peak = binned_peak.sort_values(1)
    binned_peak = binned_peak.groupby(binned_peak[1]).mean()
    binned_peak = binned_peak.sort_values('index') # reordering so points closest to onset first
    ### making onset t = 0 for fitting
    binned_peak['index'] = binned_peak['index'] - onset
    # using this as the index
    binned_peak.index = binned_peak['index']
    binned_peak.drop('index',axis=1,inplace=True)
    dropped_cols = binned_peak.isna().sum().iloc[0]
    binned_peak.dropna(axis='rows',inplace=True)
    onset_current = binned_peak.iloc[0,0]
    binned_peak = binned_peak - onset_current # subtract onset current, such that begins at t=0,pA=0
    Tau_unbinding = (peak-onset)/(1/4+(1/3) +(1/2)+1)
    # vectorising for input into H-H:
            # A = most negative current: scaling HH eqn
            # t = norm t, to give bounds unscaled 0,1
    x = np.vstack((np.array(binned_peak.index/binned_peak.index[-1]),np.array([Tau_unbinding for item in binned_peak.index]),np.array([binned_peak.iloc[:,0].min() for item in binned_peak.index])))
    I =  np.array(binned_peak.iloc[:,0]) # currents to fit
    ## fitting unbiased: no initial estimates given for residuals.
    popt,_ = sp.optimize.curve_fit(H_H_monofit_NBQX,x,I)
    ##### preparing the plot objects: binned fit, binned fit vs binned data, binned fit vs unbinned data 
    # as 3 separate plots, with t (ms) rather than norm
    hhfitfig1,hhfitaxis1 = plt.subplots()
    hhfitaxis1.set_title("Binned fit")
    hhfitaxis1.set_ylabel("I(pA)")
    hhfitaxis1.set_xlabel("ms")
    hhfitfig2,hhfitaxis2 = plt.subplots()
    hhfitaxis2.set_title("Binned fit and Binned Data")
    hhfitaxis2.set_ylabel("I(pA)")
    hhfitaxis2.set_xlabel("ms")
    hhfitfig3,hhfitaxis3 = plt.subplots()
    hhfitaxis3.set_title("Binned fit and Data (inc offset)")
    hhfitaxis3.set_ylabel("I(pA)")
    hhfitaxis3.set_xlabel("ms")
    # plotting fit
    hhfitaxis1.plot(1000*(np.array(binned_peak.index)),H_H_monofit_NBQX(x,popt),linestyle = "--",color="red")
    # plotting binned fit against binned data
    hhfitaxis2.plot(1000*(np.array(binned_peak.index)),np.array(binned_peak.iloc[:,0]),linestyle = "-",color="grey",label = "Binned Data")
    hhfitaxis2.plot(1000*(np.array(binned_peak.index)),H_H_monofit_NBQX(x,popt),linestyle = "--",color="red",label = "fit:m={}".format(np.round(popt[0],2)))
    hhfitaxis2.legend()
    #plotting binned fit against current, with time and current offset at open-tip
    hhfitaxis3.plot(1000*(np.array(peak_current.index-onset)),np.array(peak_current.mean(axis=1)),color='grey',label = "Mean Current (unbinned)")
    hhfitaxis3.plot(1000*(np.array(binned_peak.index)),H_H_monofit_NBQX(x,popt)+onset_current,linestyle = "--",color='red',label = "fit:m={}".format(np.round(popt[0],2)))
    hhfitaxis3.legend()
    hhfitfig1.tight_layout()
    hhfitfig2.tight_layout()
    hhfitfig3.tight_layout()
    print(" \n \n \n \n  \n \n \n \n \n mean peak current =",binned_peak.min()[0],"(pA)",",T = ",Tau_unbinding,"(s)",",m = ",popt[0], ",10-90 = ",((peak-onset)*0.9) - ((peak-onset)*0.1),"(s)")
    print("{} empty bins were removed.".format(dropped_cols))
    return(binned_peak.min(),Tau_unbinding,popt,((peak-onset)*0.9) - ((peak-onset)*0.1)) # returning mean peak current, Tau_unbinding, m, and 10-90 rise time
    
def current_decay(dataframe,two_components=False):
    """
    Fits 95% peak to:
        A(t) = A*exp(-t/Taufast) + B*exp(-t/Tauslow) +Iss
    
    Parameters
    ----------
    dataframe : A pandas dataframe
        Should be baselined
    two_components : True/False
        When False, a single exponential component is fitted to the current
        decay (B is zero). When True, the sum of two exponential components
        is fitted.The default is False.

    Returns
    -------
    A Graph of the current decay with superimposed fit.
    Values for fast (and slow, if selected) time constants, or value for single
    time constant in mS

    """
    # will need to get peak - done
    # get steady state
    # need to get amplitude of the fast component (peak)
    # amplitude of the slower component
    # Currently going for an unbinned approach, but can always consider a binned
    peak = get_peak(dataframe,decay=True) # gives component A for both fittig routines
    peak_to_baseline = dataframe.loc[peak:,:].mean(axis=1)
    # using get_Iss() and get_component (both should return amplitude and time
    # Normalising times to time of peak current
    peak_to_baseline.index = peak_to_baseline.index-(peak_to_baseline.index[0])
    #### get 95% current to baseline
    current_at_t = peak_to_baseline[peak_to_baseline > (peak_to_baseline.iloc[0]*0.95)]
    # get times
    t = np.array(current_at_t.index) #####
    # reformat current to numpy.array
    current_at_t = np.array(current_at_t)
    # get Iss
    _,Iss = get_Iss(peak_to_baseline)
    # fast component,A, peak amplitude
    A = current_at_t[0] #####

    # preparing figure
    if two_components:
        xdata = np.zeros([np.size(t),4])
        xdata[:,0] = t
        xdata[:,1] = A
        xdata[:,2] = Iss
        _,B = get_component(peak_to_baseline,'slow') # amplitude of slow component
        plt.style.use('ggplot')
        decayfig,decayaxs = plt.subplots(1)
        decayaxs.set_xlabel("t(ms)")
        decayaxs.set_ylabel("I(pA)")
        xdata[:,3] = B
        xdata = xdata.transpose()
        times = t*10**3 # rescaling to mS
        popt,_ = sp.optimize.curve_fit(double_exp_fit,xdata,current_at_t) # popt = Tfast,Tslow
        decayaxs.plot(times,double_exp_fit(xdata,popt[0],popt[1]),linestyle="--",color= 'red',label = "fit")
        decayaxs.plot(times,current_at_t,color = 'black',label = "data")
        decayaxs.set_title("Decay from 95% Ipeak:baseline. Tauf = {}ms,Taus = {}ms".format((popt[0]*10**3),(popt[1]*10**3)))
        decayaxs.legend()
        decayfig.tight_layout()
        return(popt[0]*10**3,popt[1]*10**3)
    else:
        xdata = np.zeros([np.size(t),3])
        xdata[:,0] = t
        xdata[:,1] = A
        xdata[:,2] = Iss
        xdata = xdata.transpose()
        plt.style.use('ggplot')
        decayfig,decayaxs = plt.subplots(1)
        decayaxs.set_xlabel("t(ms)")
        decayaxs.set_ylabel("I(pA)")
        times = t*10**3 # rescaling to mS
        popt,_ = sp.optimize.curve_fit(exp_fit,xdata,current_at_t) #popt = Tau of single component
        decayaxs.plot(times,current_at_t,color = 'black',label = "data")
        decayaxs.plot(times,exp_fit(xdata,popt),linestyle="--",color= 'red',label = "fit")
        decayaxs.set_title("Decay from 95% Ipeak:baseline. Tau = {}ms".format((popt[0]*10**3)))
        decayaxs.legend()
        decayfig.tight_layout()
        return(popt[0]*10**3)
    

def exp_fitter(pandas_structure, double = False):
    """Fits an exponential or sum of expontials to pandas structure data (either dataframe or series).
    If DataFrame is used, the fit is performed to the average of all waves
    
    if double = True (Default = False), fits a double exponential
    
    if the beginning of the fit is more positive than the end, a negative exponential or sum of, is fitted"""
    if len(pandas_structure.shape) == 2: #if dataframe
        pandas_structure = pandas_structure.mean(axis=1)
    
    if not double:
        start_fit = get_component(pandas_structure,'start of the')
        # gives t= 0, and a
        end_fit = get_component(pandas_structure,'start of the')
        # gives y_max and t = 1 (when normalised)
        # isolate t and current to fit
        It= pandas_structure[(pandas_structure.index>=start_fit[0]) & (pandas_structure.index <=end_fit[0])]
        x = np.zeros([2,np.size(It.index)]) # normalise current
        x[0,:] = It.index-It.index[0] #t, starting at 0
        x[1,:] = It #current
        if end_fit[1] > start_fit[1]:# condition to fit positive exponential
            #p0 = (times[-1]/It.iloc[0])/(np.log(Imax)) # estimate Tau
            posmonoexp = lambda x,a,tau: a*(np.exp(x[0,:]/tau))
            #if np.isinf(p0) is True:
            popt,_ = sp.optimize.curve_fit(posmonoexp,x,x[1,:])
            mtype = "positive"
            plt.plot(x[0,:]+It.index[0],posmonoexp(x,popt[0],popt[1]),linestyle ='--',color='red',label = 'fit')
            plt.plot(It.index,It,color='black',label='data')
        else: # or else, fit negative exponential
            #p0 = -(It.index[-1]/It.iloc[-1])/(np.log(Imax)) #deprecated together with np.isinf codition. No initial guess now provided
            negmonoexp = lambda x,a,tau: a*(np.exp(-x[0,:]/tau))
            popt,_ = sp.optimize.curve_fit(negmonoexp,x,x[1,:])
            mtype = "negative"
            plt.plot(x[0,:]+It.index[0],negmonoexp(x,popt[0],popt[1]),linestyle ='--',color='red',label = 'fit')
            plt.plot(It.index,It,color='black',label= 'data')
        # get Tau and a real units:
        Tau = (It.index.max()-It.index.min())/popt[1]
        a = popt[0]*(It.index.min()/(It.index.max()-It.index.min()))
        print("A {} monoexponential function was fitted,a={},Tau = {}".format(mtype,a,Tau))
        plt.show()
        return(a,Tau)
    if double:
        start_fit = get_component(pandas_structure,'start of the fast')
        #start_slow = get_component(pandas_structure,'start of the slow')
        end_fit = get_component(pandas_structure,'end of the slow')
        # start fit also = amplitude of fast component
        # iss also = amplitude of the slow component, time at end of fast component
        # get current in window
        It= pandas_structure[(pandas_structure.index>=start_fit[0]) & (pandas_structure.index <=end_fit[0])]
        x = np.zeros([2,np.size(It.index)])
        # x[0,:] = It.iloc[0]# a # deprecated: fit better without entering a,b
        # x[1,:] = start_slow[1] #b
        x[0,:] = It.index - It.index.min() # t normalised such that t=0 at first time point
        x[1,:] = It # current
        if end_fit[1]>start_fit[1]: # condition fit positive exponential
             posdouble_exp = lambda x,a,b,taufast,tauslow: (a*(np.exp(x[0,:]/taufast)))+(b*(np.exp(x[0,:]/tauslow)))
             popt,_ = sp.optimize.curve_fit(posdouble_exp,x,x[1,:])
             mtype = "positive"
             plt.plot(x[0,:]+It.index.min(),posdouble_exp(x,popt[0],popt[1],popt[2],popt[3]),linestyle ="--",color='red',label = 'Fit')
             plt.plot(x[0,:]+It.index.min(),x[1,:],color='black',label = "Data")
             plt.legend()
        else:
             negdouble_exp = lambda x,a,b,taufast,tauslow: (a*(np.exp(-x[0,:]/taufast)))+(b*(np.exp(-x[0,:]/tauslow)))
             popt,_ = sp.optimize.curve_fit(negdouble_exp,x,x[1,:])
             mtype = "negative"
             plt.plot(x[0,:]+It.index.min(),negdouble_exp(x,popt[0],popt[1],popt[2],popt[3]),linestyle ="--",color='red',label = 'Fit')
             plt.plot(x[0,:]+It.index.min(),x[1,:],color='black',label = "Data")
             plt.legend()
        # correcting time constant and amplitude values for their real domain
        A = popt[0]*np.exp((It.index.min()/(It.index.max()-(It.index.min())))/popt[2])
        B = popt[1]*np.exp((It.index.min()/(It.index.max()-(It.index.min())))/popt[3])
        Taufast = (It.index.max()-It.index.min()) * popt[2]
        Tauslow = (It.index.max()-It.index.min()) * popt[3]
        weightedtau = (Taufast*(A/(A+B))) + (Tauslow*(B/(A+B)))
        print("A sum of {} exponentials was fitted. A = {}, B = {}, Taufast = {},Tauslow={}".format(mtype,A,B,Taufast,Tauslow))
        print("The weighted time constant is = {}".format(weightedtau))
        plt.show()
        return(popt[0],popt[1],popt[2],popt[3],weightedtau)


#____________________________________________________________________________#
##___________________________ Embedded functions____________________________##
def how_to_clean():
    print('When cleaning data, it is suggested to:\n -Examine detailed notes from the time of data collection\n -Examine the boxplots- do some waves have noticeably different quartiles, a distribution magnitude, or more outliers?\n -Examine the raw, overlaid traces - do some have an unexpected waveform or appear to contain interference? \n -Is the baseline consistent?\n -Try the multiplot and data cleaning functions to individually remove waves and examine the quality of data afterwards.')

def multiplot_clean_data(self,groupsize=20,separate_avg =False,panelsizein = 20,to_clean = False,removed_sweeps=False):
    """Shows Databefore & after cleaning.
    
    groupsize specifies the number of waves per group of qaulity control graphs
    ,e.g. if 10, the sweeps are split into groups of 10. By default, groupsize
    is 20. Groupsize <=10 assigns a unique color to each sweep, making 
    identification easier. If separate_avg is set True, will return a separate graph
    of the average of selected sweeps that may be used for figure creation etc.
    panelsizein is the width and height of returned fig (20 inches by default).
    """
    # axes[0: total number of sweeps/groupsize] = overlaid sweeps
    # axes[-1] = norm rundown curve
    # axes [-2] = avg non-selected waves
    # axes [-3] = avg selected waves
    figure_arrange = int(np.ceil(np.sqrt((np.ceil(self.count().count()/groupsize))+3))) # organising panel
    cleaning_figure,cleaning_figure_axes = plt.subplots(figure_arrange,figure_arrange,figsize = [panelsizein,panelsizein])
    #groupedsweeplist = sweeplist_splitter(self,groupsize) # generator containing sweeps numbers in groups
    cleaning_figure_axes = np.reshape(cleaning_figure_axes,-1)#reshaping axes object into iterable form
    grouped_sweeps = sweeplist_splitter(self,groupsize) # calling sweeplist_splitter to return sweeps in groups ahead of panel assignment
    rundown_by_grp = pd.DataFrame() # preallocation for grouped rundown distirbution plot
    for group in np.arange(len(grouped_sweeps)): # for each group 
        for sweep in grouped_sweeps[group]: # plot each sweep of that group
            cleaning_figure_axes[group].plot(self.iloc[:,sweep], label = "{}".format(sweep)) #plotting each sweep within a group, and assigning a handle
        cleaning_figure_axes[group].legend() # creating a key for each group
        ### group container filled for rundown distribution plot
        rundown_by_grp[group] = (self.abs().div(self.iloc[:,0].abs().max(),axis='rows')).iloc[:,grouped_sweeps[group]].mean(axis=1) # plotting responses of each group, normalised to max(absolute(first sweep))
    # rundown plot
    sns.violinplot(data = rundown_by_grp, ax = cleaning_figure_axes[-1]) # Plots violinplot of each group'sresponse normalised to the absolute maximum response of the first sweep
    sns.lineplot(data= self.mean(axis=1),color='black',ax = cleaning_figure_axes[-3],dashes= False, ci = None)
    if separate_avg:
        sep_avg_fig = plt.figure(figsize = [panelsizein,panelsizein])
        sep_avg_fig = sns.lineplot(data =self.mean(axis=1),color='black',dashes = False, ci = None)
    if to_clean:
        sns.lineplot(data = removed_sweeps.mean(axis=1),color = 'red',ax = cleaning_figure_axes[-2])
        cleaning_figure_axes[-2].set_title('Mean response of removed = {}'.format(list(removed_sweeps.columns)))
    else:
        cleaning_figure_axes[-2].set_title('No Sweeps Removed')
        
        
def clean_data(dataframe_to_clean,remove_sweeps = False,last_removed = False):
    """Repeatedly accesses multiplot_clean_data, allowing removal of and 
    visualisation of suspect sweeps.
    
    It returns the output graphs of multiplot_clean_data with an additional
    panel showing the average of all non-included waves"""
    if remove_sweeps:
        sweeps_to_remove = [int(i) for i in input("Please enter sweeps to remove as comma-separated values >").split(',')]
    if last_removed:
        sweeps_to_remove = sweeps_to_remove + list(last_removed.columns)
    sweeps_to_use = dataframe_to_clean.drop(sweeps_to_remove,axis=1) # removing sweeps in shallow copy
    sweeps_to_remove = dataframe_to_clean.iloc[:,sweeps_to_remove] # storing removed sweeps in shallow copy
    multiplot_clean_data(sweeps_to_use ,groupsize = 20,separate_avg= False,panelsizein=20,to_clean=True,removed_sweeps = sweeps_to_remove )
    return(sweeps_to_use,sweeps_to_remove)
## replaced self arguemnt with dataframe_to_clean
        
    ### should print hint to make sure that all panels of plot have similar x axis scale.
    # and for low Nchannel recording, rundown not nec correct, since noise is likely to be max, however after filtering...
    ## should print warning that rundown can be seen by groups <1, and >1is runup,either a specific effect, or perhaps indicating breakdown/interference
    
    
def sweeplist_splitter(sweeps_to_split,grouping):
    """Nests sweeps form a pandas DataFrame into groups of size grouping"""
    listofselfgroups = list(np.arange(0,sweeps_to_split.count().count()))
    grp_sweeplist = []
    for item in range(0,sweeps_to_split.count().count(),grouping):
        grp_sweeplist.append(listofselfgroups[item:item+grouping]) # return, rather than yield, as size never more than ~3Mb of RAM, and then easier to index from.
    return(grp_sweeplist)

def get_open_tip_onset(open_tip_dataframe,lolimit = False,uplimit=False):
    """Returns the onset time for a piezoelectric driven jump of an open-tip
    electrode across a fluid junction.
    
    Accepts an open-tip response in the form of a pandas Dataframe, and plots
    and interactive graph from which the onset time is retrieved. Data should
    be baselined beforehand using the baseline function. """
    figure_onset,axis_onset = plt.subplots()
    if lolimit:
        axis_onset.set_xlim(left = lolimit)
    if uplimit:
        axis_onset.set_xlim(right = uplimit)
    axis_onset.plot(open_tip_dataframe.mean(axis=1)) # plot the average of the entered response
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    print('Please view the graph and left click to select an ONSET. Points can be removed by right clicking anywhere on the graph. Press enter following final selection')
    data = plt.ginput(-1,0) #collecting data from graph for a single user click, no timeout
    plt.show()
    return(data[0][0]) # returning onset time

def get_peak(baselined_response_dataframe,decay=False):
    """Returns the 100% rise time and amplitude from a BASELINED pandas DataFrame
    , using the average response waveform for calculation of 10-90%
    
    Accepts a response across a solution exchange (open_tip or experimental)
    and opens an interactive graph that shoul be used to select the earliest
    point at which the current amplitude is maximal. Data should be cleaned 
    and baselined beforehand (baseline and clean_data functions)."""
    figure_rise_max,axis_rise_max= plt.subplots()
    axis_rise_max.plot(baselined_response_dataframe.mean(axis=1)) # plot the average of the entered response
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    if not decay:
        print('Please view graph and left click once at the earliest maximum amplitude current. The last point clicked can be removed by right clicking. After selection, press enter')
    else:
        print('Please view graph and left click once at the LATEST maximum amplitude current. The last point clicked can be removed by right clicking. After selection, press enter')
    data = plt.ginput(-1,0) # accepting a single input
    plt.show()
    #max_time = (np.array(data))[:,0] ____-------------------- deprecated
    return(data[0][0]) # returning peak time

def get_Iss(baselined_response_dataframe):
    """Accepts baselined dataframe and returns the ampltiude of the peak current"""
    figure_rise_max,axis_rise_max= plt.subplots()
    axis_rise_max.plot(baselined_response_dataframe) # plot the average of the entered response
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click) 
    print('Please view graph and left click once at the earliest maximum amplitude of the steady-state current. The last point clicked can be removed by right clicking. After selection, press enter')
    data = plt.ginput(-1,0) # accepting a single input
    plt.show()
    return(data[0]) # returning Iss time and amplitude

def get_component(baselined_response_dataframe,comp_type):
    """Accepts a baselined dataframe and returns the amplitude and time of a component"""
    figure_rise_max,axis_rise_max= plt.subplots()
    axis_rise_max.plot(baselined_response_dataframe) # plot the average of the entered response
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click) 
    print("Please view graph and left click once at the largest amplitude of the {} component. The last point clicked can be removed by right clicking. After selection, Press enter".format(comp_type))
    data = plt.ginput(-1,0) # accepting a single input
    plt.show()
    plt.close()
    return(data[0]) # returning Iss time and amplitude
    
def rise_time(open_tip_dataframe,experimental_current=False,show_rundown=False,outside_out=True):
    """Returns the 10-90% rise time of a fast jump for either an open-tip response
    or current from an excised patch. Data should be baselined.
    
    if experimental data is false, the 10-90% response time refers to the
    open_tip response (i.e. difference in time = 10%,90% of maximum
    fluid junction potential). If experimental data is set to a dataframe
    containing currents, the difference is calculated using the onset time
    from the open_tip response and the maxiumum from the experimental trace - 
    i.e. difference between drug solution onset and maximum response.
    If average_rise is True, the 10-90% rise will be calculated as a single
    value = time difference between 10% and mean 90%. If False, = time
    difference between 10% and 90% for each experimental sweep. By default
    average_rise = False. If show_rundown is True, a lineplot is returned
    showing the peak current over time"""
    open_tip_onset = get_open_tip_onset(open_tip_dataframe)
    if not np.any(experimental_current):
        experimental_current = open_tip_dataframe # if no experimental record, calcualte 10-90% for open-tip -response
    peak = get_peak(experimental_current)
    # getting time and amplitude of max current
    max_amplitude = experimental_current[experimental_current.index>=peak].iloc[0]
    max_time = experimental_current.index[experimental_current.index>peak][0]
    max_times = np.zeros(experimental_current.shape[1])
    max_times[max_times==0] = max_time
    # calcualting 10to 90 rise.
    response90 = max_amplitude.mean()*0.9 # 90% of max amplitude - either a single value, or series, depending on whether average_rise = True/False
    response10 = max_amplitude.mean()*0.1 # 10%of maxamplitude - as above
    if outside_out: # for outside-out
       timeresponse90 = experimental_current[(experimental_current<=response90) & (experimental_current.idxmin()>open_tip_onset)].idxmax() # finding time point of 90% response that occurs after onset, returns the index. See comment next line.
       timeresponse10 = experimental_current[(experimental_current<=response10) & (experimental_current.idxmin()>open_tip_onset)].idxmax() # finding time point of 10% response that occurs after onset: where 10% >=0.1(max response),max(10%) gives the least negative value for which this is true (i.e value approximately = 10% response). As such, idxmax, returns the index where this occurs, i.e. timepoint of 10%
    else: # for inside-out, as above, but accounting for positive current
        timeresponse90 = experimental_current[(experimental_current>=response90) & (experimental_current.idxmin()>open_tip_onset)].idxmax()
        timeresponse10 = experimental_current[(experimental_current>=response10) & (experimental_current.idxmin()>open_tip_onset)].idxmax()
    time10to90 = (timeresponse90 - timeresponse10).mean() # will return either mean(10-90% rise time for all sweeps) or 10-mean(90)rise time
    print("10-90% rise time = ",time10to90,"from {} to {} pA".format(response10,response90))
    if show_rundown:
        rundownfig,rundownax = plt.subplots(1,1)
        sns.lineplot(x=max_times,y=max_amplitude,ax=rundownax)
        rundownax.set_title("Peak Current Amplitude over Time")
    return(time10to90)

def RMSDE(dataframe_period):
    """"Calculates root mean squared deviation errors for a wave period provided in dataframe_period
    NB, RMSE calculated for time periods of waves - not the same as in RMSDE_wave_check"""
    # inverting current
    dataframe = -dataframe_period
    # for mean wave, get RMSE, which we will scale against our other average.
    meanwavs = dataframe.mean(axis=1)
    meanmeanwavs = meanwavs.mean()
    errs = meanwavs - meanmeanwavs
    sqerrs = errs**2
    sumerrs = sqerrs.sum()
    SDErrs = (1/dataframe.shape[0])* sumerrs
    RMSDErrs = SDErrs**0.5
    return(RMSDErrs)

def RMSDE_wave_check(dataframe_period):
    """checks waves for RMSDE(var)>7."""
# the mean variance of all 
    # inverting points, such that -ve values are positive (i.e. the real signal)
    dataframe = -dataframe_period
    meanwavs = dataframe.mean(axis=0)    
    errs = dataframe - meanwavs # subtracting expected value from actual for each wave
    sqerrs = errs**2
    sumerrs = sqerrs.sum(axis=0)
    SDErrs = (1/dataframe.shape[0])* sumerrs
    RMSDErrs = SDErrs**0.5
    return(RMSDErrs.index[RMSDErrs>7])
    
##### The below are routine functions from matplotlib library
def on_move(event):
    """Used by interactive functions to update axes with x and y coordinates"""
    # get the x and y data coordinates
    time_event, amplitude_event = event.x, event.y
    if event.inaxes:
        ax = event.inaxes  # axes instance
def on_click(event):
    """Used by interactive fucntions to return x and y coordinates"""
    if event.button is MouseButton.LEFT:
        time_event, amplitude_event = event.xdata, event.ydata
        print("event time =",time_event,", amplitude =", amplitude_event)


def noise_func(x,N,i): # where x is an independent variable containg data for I_mean and background_var
    """Embedded function used for curve fitting of non-stationary fluctuation
    analysis: ensemble variance = i*I_mean/(N+background variance)"""
    I_mean= x[0,:]   # unpack x in to I_mean and background_var
    background_var = x[1,:]
    return np.array(background_var + (i*I_mean) - ((I_mean*I_mean)/N))
        
def H_H_monofit_NBQX(x,m):
    """Embedded function to fit a monoexponential Hodgkin-Huxley
    type equation I_t = A*(1-e^(-t/Tau))^m, using Tau as an argument for sweeps in
    a dataframe.Here Tau is for NBQX unbinding. Takes tuple containing t and Tau"""
    t = x[0,:]
    Tau = x[1,:]
    A = x[2,:]
    return(np.array(A*(1-np.exp(-t/Tau))**m))


def exp_fit(x,Tau):
    # where x is an independent variable contianing data for A1, time, and Iss
    t = x[0,:]
    A = x[1,:]
    Iss = x[2,:]
    # by Banke,2000 priel 2005: fitting decay currents with a single exponential from 95% to baseline
    # A(t) = A*exp(-time/tau)+Iss
        # where A is peak 95% - i.e. the amplitude of the fast component.
        # Iss is steady state current
    return(np.array((A*np.exp(-t/Tau))+Iss))

def double_exp_fit(x,Taufast,Tauslow):
    # by banke,2000: fitting sum of two exponential components
    # A(t) = A*exp(-time/Taufast) + B*exp(-t/Tauslow)    +Iss
    # where A(t) is current amplitude, A is amplitude of fast component,B is amplitude of the slow component
    t = x[0,:]
    A = x[1,:]
    Iss = x[2,:]
    B = x[3,:]
    return(np.array((A*np.exp(-t/Taufast))+(B*np.exp(-t/Tauslow))+Iss))

def parabola(x,a,b,c):
    if not c:
        c = 0
    return((a*x**2)+(b*x)+c)