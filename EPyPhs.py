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

Scale bar / calibration bars use code from:
    https://gist.github.com/dmeliza/3251476
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
print("\n \n \n WELCOME TO EPyPhys 1.0.")

plt.style.use("ggplot")
import matplotlib.colors as mcolors
from cycler import cycler
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colorlist = ['black','dimgrey','teal','darkturquoise', 'midnightblue','lightskyblue','steelblue','royalblue','lightsteelblue','darkorange', 'orange','darkgoldenrod','goldenrod','gold','khaki','yellow']
mycycle = cycler(color=[colors[item] for item in colorlist])
#    plt.style.use("ggplot")
#    plt.rc('axes',prop_cycle=mycycle)
print("\n \n Currently using ggplot as default plotting environment")
print("Currently using EPyPhys Color Cycle: black-grey-blues-oranges-yellows")


#_____________________________________________________________________________#
#___________________Data import, screening, merging, and cleaning______________________#



def load_data(path,panelsizein = 20,leg = False,plot=True):
    """load_data accepts a trace files of type ABF or ABF2 for inspection
    Trace files can either be specified as a path of type(path)=str or can be 
    dragged between the open brackets.Panelsizein gives width, height of graph panel 
    in inches and is 20 by deafult. 
    
    When leg = False, no legend is plotted
    
    When plot = False, no graphs are returned
    
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
    if plot:
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
            title2 = 'Overlaid Sweeps from {}'.format(path.split('/')[-1])
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

def load_waves(path,panelsizein = 20,leg = False):
    """
    

    Parameters
    ----------
    Loads a wave in as well as the command wave form associated abf file. 
    If command is set manually through the amplifier, rather than driven by software command
    then there may be no associated command wave.
    
    
    path: Str
    Path to abf file
    panelsizein : TYPE, optional
        Panel size in inches. Larger panels use more memory.. The default is 20.
    leg : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    Data, command waves

    """
    if type(path) !=str:
        return("A path should be specified as a string")
    open_tip = pyabf.ABF(path)
    sampling_freq = open_tip.dataRate # sampling rate in Hz.
    for sweepnum in open_tip.sweepList:
        open_tip.setSweep(sweepnum)
        if sweepnum ==0: # initialising pandas DataFrame with indexes as times for sweep 0
            data = pd.DataFrame(data=open_tip.sweepY,index=open_tip.sweepX)
            com = pd.DataFrame(data=open_tip.sweepC,index=open_tip.sweepX)
        else: # adding remaining sweeps to the DataFrame object as a new column
            data[sweepnum] = open_tip.sweepY
            com[sweepnum] = open_tip.sweepC
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
    figure_3,axis_3 = plt.subplots()
    for sweepNumber, value in enumerate(open_tip.sweepList):
        open_tip.setSweep(sweepNumber)
        axs2.plot(open_tip.sweepX,open_tip.sweepC,label = "{}".format(sweepNumber))
    title3 = 'Command wave form'
    figure_3.tight_layout()
    return(data,com)
    
    print("Sampling freqency was", sampling_freq/1000,"kHz")
    print("Data is now in DataFrame format. Remove waves with variable_name = variable_name.drop(columns = [comma-separated numbers])")
    print("Data should be baselined before use. Type help(baseline)")
    print("For examining an open tip, enter NBQX_unbinding(open_tip_variable_name,open_tip_variable_name)")
    print("For merging data, see merge data")
    return(data)
    

def merge(records_to_merge_as_dataframes):
    """Enter records to merge as a tuple of dataframes,e.g. merge((rec1,rec2))
    and returns a single dataframe containing all data."""
    merged = records_to_merge_as_dataframes[0]
    for item in records_to_merge_as_dataframes[1:]:
        merged = np.append(merged,item.to_numpy(),axis=1)
    record = pd.DataFrame(merged, index = records_to_merge_as_dataframes[0].index)
    return(record)

def flatten_sweep(sweep):
    chained_sweeps = sweep.to_numpy().flatten('F')
    ind = [sweep.index +(sweep.index.max()* item) for item in np.arange(sweep.shape[1])]
    ind = np.array(ind).flatten('C').transpose()
    flat_sweep = pd.Series(chained_sweeps,index=ind)
    return(flat_sweep)

def append_to(record,record2):
    """Appends record2 to the end of record 1, such that the (time) index of record 2
    is a continuation of record 1"""
    r2s = record.index.max() # interval
    record2.index = record2.index+r2s
    appended = pd.concat((record,record2),axis=0)
    return(appended)
        
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

def split_sweeps(response_dataframe,times_to_split,t_zero=True):
    """Splits each sweep at times_to_split (ms) to return a list of size x of 
    ordered sweep fragements as dataframes. size x = size(times_to_split)+1
    
    baselining is performed on the wave before splitting.
    times to split can either be a single value or a tuple of values,
    excluding t =0 and t = max(t)
    
    If t_zero = True (Default, the index of each fragment is set to start at 0).
    Else, the index retains its original value from respinse_dataframe
    
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
    if t_zero:
        for item in np.arange(len(fragments)):
            fragments[item].index = fragments[item].index - fragments[item].index[0]
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

def align_within(record,num_events,plot=False):
    """Aligns events within an averaged record, such that they occur at t=0.
    
    e.g. for alignment of multiple stimuli within an averaged sweep
    
    - First, sweeps are split between selected events (for each event in num events)
    The first split occurs from t=0 to first selection.
    The last split occurs from the last selection to t = max
    - Then events are aligned
    
    - When plot= True (Default = False), plots are returned of the aligned sweeps
    
    nb, This differs from align_at_t: align at t is used for aligning each sweep to t=0, whereas align_within 
    aligns multiple events within an e.g. averaged record"""
    
    splittimes = []
    #splittimes.append(record.index.min()) # deprecated
    for item in np.arange(num_events):
        alfig,alaxs = plt.subplots()
        alaxs.plot(record.mean(axis=1),color='black')
        binding_id = plt.connect('motion_notify_event', on_move)
        plt.connect('button_press_event', on_click)
        alaxs.set_title("Left Click to select start of event {}/num_events. \nRight click to remove selection.\n Press enter when point selected".format(item))
        alfig.tight_layout()
        redpoints = plt.ginput(-1,0)
        plt.close(fig=alfig)
        points = redpoints[0][0]
        splittimes.append(points*1000) # correcting for use in pslit_sweeps(needs ms)
    #splittimes.append(record.index.max()) # deprecated
    split_records = split_sweeps(record.mean(axis=1),times_to_split=splittimes,t_zero = False)
    for item in np.arange(np.size(split_records)):
        split_records[item].index = split_records[item].index-(split_records[item].index.min())
    split_records.pop(0)
    for item in split_records:
        item.plot(color='black')
    return(split_records)
    ### need to decide on plot
    # and need to remove first fragment
        

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


from scipy.signal import butter, lfilter # freqz

def butter_lowpass(cutoff, fs, order=5):
    """
    Built-in used by butter_lowpass_filter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, fs=fs)
    return b, a

def butter_lowpass_filter(x, cutoff, fs, order=5):
    """
    Parameters
    ----------
    x : data
    cutoff : cutoff frequency in Hz
    fs : sampling frequency of the data
    order : Filter order. The default is 5.

    Returns
    -------
    Filtered data
    
    Can be used as .apply_along_axis method

    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, x)
    
#-----------------------------------------------------------------------------    
#######_________________________ANALYSIS METHODS_________________________#####

    # plot cov [0,:] which is the autocovariance centred at the peak vs the ensemble current
        # because i've aligned the current, such that row 0 is the aligned point 
    # which shows the relationship of currents fluctuations over time
    
    # also return corr, which gives strength of the relationship - might be useful for insight

    # what would the average of all points show?
        # WOULD IT NEED TO BE THE AVERAGE OF triangular (lower of symmetrical) elements?
        # For each ti, the extent to which ti is related to all other tjs
        
    # could clean up returns so that a single item is returned
        # e.g. a dict

def NSFA(dataframe,pairwise = False,voltage=-60,num_bins = 10,Vrev = 0,parabola = True,start_time = False,end_time = False,background_start = False,background_end = False,suppress_g = False,wave_check=False,alignment = True,return_var = False,cov_corr=False):
    """    Performs Sigg (1997) style pairwise (or unpaired) non-stationary fluctuation analysis between
    two points of interest with the options of alignmentand binning.
    
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
    voltage (mV): Default = -60. Used to calculated weighted mean single channel conductance
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
        This should include the current rise time if alignment is True
    end_time (s) : T
        End time for interval in which to perform pairwise analysis. When
        False, user is asked to select from graph. The default is False.
    background_start (ms) : TYPE, optional
        Time from which to take background interval. The default is False = 0.
    background_end (ms)
        Time until which to take background interval. The default is False = 20
    suppress_g : True/False: Suppress graphs when True. The default is False.


    ------ Important optional arguments -------
    Alignment:True/False: When True, alignment performed on 90% current rise. Default = True.
        When True, the ROI should include the current rise. When False, should not.
    
    pairwise:True/False: When True, performs Sigg, 1997 noise abalysis where the variance is calculated
        between wave pairs. A parabolic function is then fitted to discretised isochronal paired mean and variance values.
        Otherwise (False), the parabola is fitted using discretised values of current and varaince across the isochrone for all waves
            -i.e. False = fitted to the ensemble variance, mean, and background.
        
    if wave_check = True, waves >7 RMSE are discarded
    
    cov_corr: True/False: When True (Default False) returns the symmetric, 2D covariance and correlation matrices
    centred about the peak of the aligned traces, as well as the mean ensemble response for the same timepoints.
    When False, two empty arrays are returned instead for covariance and correlation
        These can be plotted as:
            plt.plot(aligned_macroscopic_response_dataframe.index,covaraince_matrix[0,:])
                # which shows the covariance of the first time point (the 90% alignment) with all other t
                    # where 0 denoees the first time point. For centering about other t, adjust
            - The same notation can be used to plot correlation
            - The enzemble average can be plotted simply with: aligned_macroscopic_response.mean(axis=1).plot()

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
    if not start_time: # if no start time is provided, see if an open-tip-response dataframe was given
            start_time = get_open_tip_onset(dataframe)
    if not end_time: # if no peak time is specified
        end_time = get_peak(dataframe) # obtained from graph
    if not background_start:
        background_start = 0
    if not background_end:
        background_end = 20 # baseline end in ms
    background_end = background_end/1000
    background_start = background_start/1000
    # fix background > start time
    if (background_end>start_time) & (background_start<start_time):
        background_end = start_time - start_time/1000
    
    ### find all currents in window start_time:end_time
    peak_current = dataframe.loc[start_time:end_time,:] # taking peak current for all waves
    # peak current is now aligned to first time point = or > 90% rise
    
    # need to account for that in baseline.
    ### Repeating background period through peak current for background at isochrone
    base = dataframe.loc[background_start:background_end,:] # get background for each wave
    
    # drop base < 90% rise time
    if alignment:
        aligned_peaks = peak_current.copy(deep=True)
        # first time point current at >=90% peak for each sweep - NB, might have wanted some rejection criterion for this
        t_90s =  aligned_peaks[aligned_peaks.abs()>=(aligned_peaks.abs().max()*0.9)].idxmin()
        for item in np.arange(aligned_peaks.shape[1]):
                aligned_peaks.iloc[:,item][aligned_peaks.index<t_90s[item]] = np.nan
                # setting first point to t = 0
                arr = aligned_peaks.iloc[:,item][aligned_peaks.iloc[:,item].notna()].to_numpy()
                aligned_peaks.iloc[:np.size(arr),item] = arr
                aligned_peaks.iloc[np.size(arr):,item] = np.nan # removing residual values
        # truncate background
        #inds = np.where(peak_current.index>=t_90s.min())[0]
    else:
        aligned_peaks = peak_current
    # background is not long enough
        
    #______ --- background variance for isochrones --_____ #
    ### number of repeats of periodic baseline during aligned peak current
    r_baseline = (aligned_peaks.index[-1]-aligned_peaks.index[0])/(base.index[-1]-base.index[0])
    # fraction of period through baseline variance when peak current starts
    if alignment:
        baseline_start = (peak_current.index[-1]-t_90s.index.min())%(base.index[-1])/(base.index[-1])
    else:
            baseline_start = (peak_current.index[0])%(base.index[-1])/(base.index[-1])

    # calculating background noise for all isochronous time points in peak current
    # from point when background starts, calculate background variance for all wave pairs
    background_period = base[base.index>(baseline_start*base.index[-1])].to_numpy()
    background_period = np.vstack((background_period,base[base.index <=(baseline_start*base.index[-1])].to_numpy()))
    background = background_period
    for item in np.arange(int(np.floor(r_baseline))-1):
        background = np.vstack((background,background_period))
    # from end of last whole period of background, repeat the remaining fraction of the period
    background = np.vstack((background,background_period[:aligned_peaks.shape[0] - np.size(background,0),:]))

    #Heinemann and Conti verification for background and periods of interest
    # performed on aligned ewaves
    if wave_check:
        todiscardinterest = RMSDE_wave_check(aligned_peaks)
        todiscardbackground = RMSDE_wave_check(base)
        to_remove = np.unique(np.append(todiscardinterest,todiscardbackground))
        background = np.delete(background,list(to_remove),axis=0)
        aligned_peaks = aligned_peaks.drop(columns = list(to_remove))
        
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
        
    ### getting covariance, corr etc; plot cov
    cov = np.array([])
    corr = np.array([])
    if cov_corr:
        cov = np.cov(aligned_peaks.to_nunmpy())
        corr = np.corrcoeff(aligned_peaks.to_numpy())
        covfig,covaxs = plt.subplots()
        covaxs.set_xlabel("t (s)")
        covaxs.set_ylabel("$\sigma^2$ (pA$^2$)")
        covaxs.plot(aligned_peaks.index,np.diag(cov),color='black',label = 'ensemble variance')
        covaxs.plot(aligned_peaks.index,cov[0,:],color='midnightblue',label = 'Cov(t$_{peak}$,t')
        covfig.tight_layout()
        
    ###
    # convert to numpy: looping faster than dealing with pandas means and var.
    currents = aligned_peaks.to_numpy()
    
    # perform Sigg (1994) type noise analysis for all time points.
    # mean for each isochrone, for pair diffs holding 'noise traces' of wave pairs
    # and then calculating variance of the noise
    mean_isochrone_current = aligned_peaks.mean(axis=1)
    conf_int = sp.stats.norm.interval(0.95,loc=(np.var(currents,axis=1)),scale=np.std((np.var(currents,axis=1))))   # here, use this plot for mean current at time t against variance at t (i.e. for each isochrone current vs variance)

    if pairwise:
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
            var_isochrone_current =  (2/np.size(currents,axis=1))*np.sum(isochrone_differences,axis=1)
            # background varaince
            background = (2/np.size(background_differences,axis=1))*np.sum(background_differences,axis=1)
            # 95% confidence interval of varaince for each isochrone:
            # removing current-variance pairs for wrong sign
            if not suppress_g:
                noise_axs.scatter(np.abs(np.mean(currents,axis=1)),(var_isochrone_current-background),marker='.',color='grey',label = ' |Isochrone mean (I)| vs $\sigma^2$(noise) - $\sigma^2$B(noise)')
                raw_axs.errorbar(np.abs(np.mean(currents,axis=1)),(var_isochrone_current+background),yerr = conf_int,marker='.',color='black',barsabove=True,errorevery=10,label = 'Isochrone I vs $\sigma^2$(I) + $\sigma^2$B 95% CI')

    else:
        var_isochrone_current = aligned_peaks.var(axis=1)
        background = background.var(axis=1)
        if not suppress_g:
                noise_axs.scatter(np.abs(np.mean(currents,axis=1)),(var_isochrone_current-background),marker='.',color='grey',label = ' |Isochrone mean (I)| vs $\sigma^2$(noise) - $\sigma^2$B(noise)')
                raw_axs.errorbar(np.abs(np.mean(currents,axis=1)),(var_isochrone_current+background),yerr = conf_int,marker='.',color='black',barsabove=True,errorevery=10,label = 'Isochrone I vs $\sigma^2$(I) + $\sigma^2$B 95% CI')
    if num_bins == 0: # if no bins, use each isochrone separately
        num_bins = int(mean_isochrone_current.size)
    x = (np.vstack((mean_isochrone_current,background,var_isochrone_current))).transpose()  
    if np.any(np.isnan(background))==True:
        ### catch cases where background not there (previous solution above forced all background to 0)
        x[x[1,:]==np.nan]=0
    x = pd.DataFrame(x)
    x.index = peak_current.index
    x[3] = pd.cut(x[0],num_bins)
    x = x.sort_values(3)
    x = x.groupby(x[3]).mean()
    x = (x.to_numpy()).transpose()
    x = np.flip(x,axis=1)
    if voltage <0:
        if np.abs(aligned_peaks.min().min())>aligned_peaks.max().max(): # if negative current at negative potentials
            x[0,:] = -x[0,:] # invert current so that current of interest on positive axis
    # remove non-finite elements
    x = np.delete(x,np.where(~np.isfinite(x)),axis=1)
    popt,pcov = sp.optimize.curve_fit(noise_func, x[:2,:], x[2,:])

    # plotting the fit 
    perr = np.sqrt(np.diag(pcov))
    if not suppress_g:
        noise_axs.scatter(x = x[0,:],y = noise_func(x[:2,:],popt[0],popt[1]),linestyle="--",color='red',label='fit,N={},i={}'.format(np.round(popt[0],2),np.round(popt[1],2)))
        noise_axs.set_title("Binned fit (nbins={}) vs raw data".format(num_bins))
        noise_axs.set_xlim(left = 0)
        noise_axs.set_ylim(bottom = 0)

    sdefit = noise_func(x[:2,:],perr[0],perr[1])
    if not suppress_g:
        fit_axs.errorbar(np.abs(x[0,:]),np.abs(noise_func(x[:2,:],popt[0],popt[1])),yerr = sdefit,barsabove=True, fmt="o",capsize = 5,color='black')
        fit_axs.set_title("Binned fit +- 1SD Errors of fit")
        fit_axs.set_xlim(left = 0)
        fit_axs.set_ylim(bottom = 0)
    if parabola:
        if not suppress_g:
            parabfig,parabaxs = plt.subplots()
            parabaxs.set_xlabel("I")
            parabaxs.set_ylabel("$\sigma^2$ (pA$^2$)")
            # simulating data: double current
            max_current = (np.max(x[0,:]))
            sim_current = np.linspace(max_current,2*max_current,np.size(x[0,:]))
            # need to simulate the variance for greater current values
            sim_curr_back = np.vstack((sim_current,x[1,:]))
            parabaxs.scatter(x[0,:],noise_func(x[:2,:],popt[0],popt[1]),label = 'Parabolic fit to data',color='black')
            parabaxs.plot(sim_current,noise_func(sim_curr_back,popt[0],popt[1]),linestyle="--",color='red',label='fit of noise parabola to simulated data,N={},i={}'.format(np.round(popt[0],2),np.round(popt[1],2)))
            parabaxs.legend()
            parabaxs.set_title("Fit of simulated and experimental data")
            parabaxs.set_ylim(bottom=0)
            parabaxs.set_xlim(left=0)
            parabfig.tight_layout()
            # simulating data for rest of curve
    SD_errors = np.sqrt(np.diag(pcov))
    N = np.abs(popt[0])
    i = np.abs(popt[1])
    # Nb, for outside out (-ve current)
    # get P_open as maximum of current/N*i
    P_operrs = (np.array([SD_errors[0]/i,SD_errors[1]/N])).prod() # % errors in i and N
    
    P_open = (np.nanmax(np.abs(x[0,:])))/(N*i)
    P_operrs = P_operrs*P_open # convert % error into P_open units
    y_mean = (np.max(np.abs(x[0,:]))/((voltage*10**-3)-Vrev)/P_open)/N
    pubfig,pubaxs = plt.subplots()
    pubaxs.plot(np.abs(-x[0,:]),np.abs(noise_func(x[:2,:],popt[0],popt[1]))-np.abs(x[1,:]),color='black',Label = 'Fit')
    pubaxs.scatter(np.abs(-x[0,:]),x[2,:]-x[1,:],color ='midnightblue',marker='.',label="Binned Data")
    pubaxs.grid(False)
    pubaxs.set_xlabel("I")
    pubaxs.set_ylabel("$\sigma^2$ (pA$^2$)")
    #pubaxs.set_facecolor("white")
    pubaxs.annotate("{}pS".format(np.abs(np.round(y_mean,2))),xycoords = 'figure fraction',xy=[0.5,0.9])
    pubfig.legend(loc=[0.75,0.85])
    pubfig.tight_layout()
    if not suppress_g:
        noise_axs.legend()
        raw_axs.legend()
        noise_fig.tight_layout()
        raw_fig.tight_layout()
        fit_fig.tight_layout()
        if wave_check:
            print("Waves {} were removed (background or ROI > 7RMSE)".format(len(to_remove)))
        if not voltage:
            print('Fitted N = {} +-{},i = {} +-{},P_open = {}+-{}'.format(np.round(N,2),np.round(SD_errors[1],2),np.round(i,2),np.round(SD_errors[0],2),np.round(P_open,2),np.round(P_operrs,3)))
        else:
            print('Fitted N = {} +-{},i = {} +-{},P_open = {}+-{},gamma_mean = {}'.format(np.round(N,2),np.round(SD_errors[1],2),np.round(i,2),np.round(SD_errors[0],2),np.round(P_open,2),np.round(P_operrs,3),np.round(y_mean,2)))
        if not parabola:
            print("If you wish to extend the parabola, call again with parabola = True. This can be useful to see whether a fit is adequate, and is particularly useful after binning")
        if return_var:
            return(N,i,P_open,y_mean,x,cov,corr)
        else:
            return(N,i,P_open,y_mean,cov,corr)
    else:
        if return_var:
            return(N,i,P_open,y_mean,x,cov,corr)
        else:
            return(N,i,P_open,y_mean,np.append(SD_errors,P_operrs),cov,corr)

def shared_N_NSFA(dataframe,voltages=[-60,-40,-20,0,20,40,60],num_bins = 20,Vrev = 0,open_tip_response = False,start_time = False,end_time = False,background_start = False,background_end = False):
    """
    Performs Sigg (1997) style pairwise non-stationary fluctuation analysis between
    two points of interest at different voltages with shared parameter N
    
    No RMSD check is performed
    
    Variance of the noise is calculated for N waves as:
        Variance = (2/N)* (yi-yu)^2
        Where yi = 0.5*(xi-xu): 
            The average difference of each pairwise noise trace for each isochrone
            minus mean of all pairwise average differences for each isochrone
        As such, the varaince measured is the variance of noise between successive
        wave pairs at each isochrone.
        
    N and i are then fit using equation at each voltage, with i as the free paramete:
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
        This is 20. If changed, must change num_bins in the two shared_N_NSFA functions
        
        
        Using NSFA_optimal() instead will select this number base on the minimisation
        of the fit error. 
    voltages (List of mV): Default = -60. Used to calculated weighted mean single channel conductance
    Vrev : 
        Reversal potential to calculate single channel condutance. The default is 0.
            # in this instance, colud be calculated from lm
    voltages :
        voltage used to calculate single channel conductance. Should be list
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
    Returns
    -------
    N,i,P_opens,y_means

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
    background_start = background_start/1000
    ### find all currents in window start_time:end_time
    
    
    # need separate 'peaks' for each voltage
    peaks = []
    base = []
    for item in np.arange(len(voltages)):
        peaks.append(dataframe.xs(item,1,1).iloc[np.where(dataframe.xs(item,1,1).index>=start_time)[0].min():np.where(dataframe.xs(item,1,1).index<=end_time)[0].max(),:])
        base.append(dataframe.xs(item,1,1).iloc[np.where(dataframe.xs(item,1,1).index>=background_start)[0].min():np.where(dataframe.xs(item,1,1).index<=background_end)[0].max(),:])
        
    # need to do this for each background
    
    backgrounds = []
    for item in np.arange(len(voltages)):
        #______ --- background variance for isochrones --_____ #
        ### number of repeats of periodic baseline during peak current
        r_baseline = (peaks[item].index[-1]-peaks[item].index[0])/(base[item].index[-1]-base[item].index[0])
        # fraction of period through baseline variance when peak current starts
        baseline_start = (peaks[item].index[0])%(base[item].index[-1])/(base[item].index[-1])
        # calculating background noise for all isochronous time points in peak current
        # from point when background starts, calculate background varaince for all wave pairs
        background_period = base[item][base[item].index>=(baseline_start*base[item].index[-1])].to_numpy()
        background_period = np.vstack((background_period,base[item][base[item].index <(baseline_start*base[item].index[-1])].to_numpy()))
        background = background_period
        for background_repeat in np.arange(int(np.floor(r_baseline))-1):
            background = np.vstack((background,background_period))
        # from end of last whole period of background, repeat the remaining fraction of the period
        background = np.vstack((background,background_period[:peaks[item].shape[0] - np.size(background,0),:]))
        backgrounds.append(background)

    # # initialising plot objects
    # if not suppress_g:
    #     plt.style.use('ggplot')    
    #     raw_fig,raw_axs = plt.subplots()
    #     raw_axs.set_xlabel("I(pA)")
    #     raw_axs.set_ylabel("$\sigma^2$ (pA$^2$)")
    #     raw_axs.set_title("95% CI of $\sigma^2$(I) in raw data (no background correction) ")
    #     noise_fig,noise_axs = plt.subplots()
    #     noise_axs.set_xlabel("I(pA)")
    #     noise_axs.set_ylabel("$\sigma^2$ (pA$^2$)")
    #     fit_fig,fit_axs =  plt.subplots()
    #     fit_axs.set_xlabel("I(pA)")
    #     fit_axs.set_ylabel("$\sigma^2$ (pA$^2$)")
    
    # converting to numpy for fast looping
    # at moment, only storing mean ucrrent, background variance, and variance in x
    x = np.zeros([3,num_bins*len(peaks)])
    for item,value in enumerate(peaks):
        value = value.to_numpy()
        # currents = np.concatenate((currents,value),axis=2)
        # mean_isochrone_currents = np.concatenate((mean_isochrone_currents,np.mean(value,axis=1)),axis=1)
        mean_isochrone_current = np.mean(value,axis=1)
        pair_diffs = np.zeros([np.size(value,axis=0),(np.size(value,axis=1)-1)])
        back_diffs = np.zeros([np.size(value,axis=0),(np.size(value,axis=1)-1)])
        for first_wave in np.arange(np.size(value,axis=1)-1): # for first wave of each pair
            # get noise trace
            pair_diffs[:,first_wave] = 0.5*(value[:,first_wave] - value[:,first_wave+1])
            # get background noise trace for each wave pair
            back_diffs[:,first_wave] = 0.5*(backgrounds[item][:,first_wave]-backgrounds[item][:,first_wave+1])
        # get isochrone noise differences ^2 with list comprehension
        isochrone_differences=np.vstack([(pair_diffs[:,adiff]-np.mean(pair_diffs[:,:],axis=1))**2 for adiff in np.arange(np.size(pair_diffs,1))]).transpose()  
        # get background differences
        background_differences = np.vstack([(back_diffs[:,adiff]-np.mean(back_diffs[:,:],axis=1))**2 for adiff in np.arange(np.size(back_diffs,1))]).transpose()  
        # get variance
        variance =  (2/np.size(value,axis=1))*np.sum(isochrone_differences,axis=1)
        # background varaince
        background = (2/np.size(background_differences,axis=1))*np.sum(background_differences,axis=1)
        
        #will have split each into bins and then change the fitted function
        
        unbinned_mean_iso_I = mean_isochrone_current
        unbinned_background = background
        unbinned_var = variance
        unbinned_all = np.vstack((unbinned_mean_iso_I,unbinned_background,unbinned_var))
        unbinned_all = pd.DataFrame(unbinned_all)
        unbinned_all = unbinned_all.transpose()
        #unbinned_all.index = peaks[item].index
        unbinned_all[3] = pd.cut(unbinned_all[0],num_bins)
        unbinned_all= unbinned_all.sort_values(3)
        unbinned_all = unbinned_all.groupby(3).mean()
        unbinned_all = (unbinned_all.to_numpy()).transpose()
        unbinned_all = np.flip(unbinned_all,axis=1)
        
        # collect the binned values into x
        x[:,item*num_bins:(item+1)*num_bins] = unbinned_all
        
    # might need to flip before fitting
    x = np.flip(x,axis=1)
    
    #Here:
        # so looking at x here:
            # have current of different sign
            # and not sure that zeo volts would work ever
        # would be clever to add extra 0 to each to force fit through zero
        # when fit individually, it's easy to see which value not great for N
        # here it's minus 20 that is the problem
        
        #but cna make all same sign would help
    
    # if using different number of voltages, this needs changing
        # it doesn't matter what the voltages are here, only that there are 6 for 
        # to +40 in mine, and 7 when full I-V
    # returns nan if error
    if 60 in voltages:
        try:
            popt,pcov = sp.optimize.curve_fit(shared_N_noise_fun_to_60,x[:2,:],x[2,:])
            perr = np.sqrt(np.diag(pcov))
        except Exception:
            popt = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan # this and repeat below are problematic
            perr = np.nan

    else:
        try:
            popt,pcov = sp.optimize.curve_fit(shared_N_noise_func,x[:2,:],x[2,:])
            perr = np.sqrt(np.diag(pcov))
        except Exception:
            popt = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
            perr = np.nan

    N = np.abs(popt[0]) # N is single value
    i = np.abs(popt[1:])
    P_opens = []
    y_means = []
    # get weighted mean conductance and Peak open prob for each voltage
    for item, voltage in enumerate(voltages):
        if voltage<0:
            P_open = np.max(-x[0,item*num_bins:(item+1)*num_bins])/(N*i[item])
            y_mean = (np.max(-x[0,item*num_bins:(item+1)*num_bins])/((voltage*10**-3)-Vrev)/P_open)/N
        else:
            P_open = np.max(x[0,item*num_bins:(item+1)*num_bins])/(N*i[item])
            y_mean = (np.max(x[0,item*num_bins:(item+1)*num_bins])/((voltage*10**-3)-Vrev)/P_open)/N
        P_opens.append(P_open)
        y_means.append(y_mean)
        
    ##--Plotting--##
        # haven't done yet.
        # was performed post-hoc
    return(N,i,P_opens,y_means,x)

def recovery_from(record,num_events=8):
    """Accepts an averaged, baselined record and calculates the time constant for the recovery
    from desensitisation. A number of events are accepted, as well as a record max
    
    Alignment is performed during execution.
    
    num_events = number of paired pulse repeats. Default = 8
    
    returns A,Tau,fractional peak recovery
    
    where fractional peak recovery is a pandas series containing the (negative)
    amplitudes of the second pulse normalised to the first pulse.
    """
    print("final event is record max")
    aligned_record = align_within(record=record,num_events=num_events)
    aligned_record.pop(-1)
    recfig,recaxs = plt.subplots()
    for item in aligned_record:
        recaxs.plot(item,color='black')
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    recaxs.set_title("Left click to select a time point \n just before the second pulse of the first pair (baseline)")
    recfig.tight_layout()
    points = plt.ginput(-1,0)
    plt.close(fig=recfig)
    baseline_t = points[0][0]
    baseline_scale = points[0][1]
    recfig,recaxs = plt.subplots()
    for item in aligned_record:
        recaxs.plot(item,color='black')
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    recaxs.set_title("Left click to select a time point \n just after the final paired pulse")
    recfig.tight_layout()
    points = plt.ginput(-1,0)
    plt.close(fig=recfig)
    end_t = points[0][0]
    norm_pp = []
    second_pulse_peak_amplitude = []
    second_pulse_peak_t = []
    ### normalise each second pulse of pair to first
    # and get the second pulse amplitude
    for item in aligned_record:
        norm_pp.append(-(item/item.min()))
        second_pulse_peak_amplitude.append((-item/item.min()).loc[(item/item.min()).index[(-(item/item.min())).index >= baseline_t].min():].min())
        second_pulse_peak_t.append((-item/item.min()).loc[(item/item.min()).index[(-(item/item.min())).index >= baseline_t].min():].idxmin())
    recovery_frame = pd.DataFrame(second_pulse_peak_amplitude,index=second_pulse_peak_t)
    print("select a time point before and then a second after the data points")
    recfig,recaxs = plt.subplots()
    for item in norm_pp:
        recaxs.plot(item,color='black')
    A,Tau = exp_fitter(recovery_frame,double=False,fitflag=False)
        
    recaxs.set_title("Normalised recovery, Tau = {}".format(Tau))
    
    ### cna then scale data by peak current of first pulse = aligned_record.min()
    ### which doesn't matter, as when add scalebar, lose axis, so can just multiply by 
    # and then add scale bar
    
    ## then scaling the plot back into pA 
    rec2fig,rec2axs = plt.subplots()
    for item in np.arange(np.size(recaxs.get_lines())):
        xcoords = recaxs.get_lines()[item].get_xdata()
        ycoords = recaxs.get_lines()[item].get_ydata()
        if item == np.arange(np.size(recaxs.get_lines()))[-1]:
            rec2axs.plot(xcoords,-(ycoords*aligned_record[0].min()),color='red',linestyle="--")
        else:
            rec2axs.plot(xcoords,-(ycoords*aligned_record[0].min()),color='black')
    rec2axs.grid(False)
    rec2axs.set_facecolor("white")
    add_scalebar(rec2axs,loc="lower right")
    rec2fig.tight_layout()
    print("Current of amplitude {} recovered from desensitisation with Tau ={}".format(aligned_record[0].min(),Tau))
    return(aligned_record[0].min(),Tau,recovery_frame)

def NBQX_unbinding(dataframe,open_tip_response,n_bins=100,give_onset = False,onset_lo_lim =False,onset_up_lim=False,give_peak = False):
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
    n_bins: number of bins to cut by. Default =1 00
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
    binned_peak[1] = pd.cut(binned_peak[0],n_bins)
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
    
    hhfitfig4,hhfitaxis4 = plt.subplots()
    hhfitaxis4.plot(1000*(np.array(peak_current.index-onset)),np.array(peak_current.mean(axis=1)),color='black',label = "Data")
    hhfitaxis4.plot(1000*(np.array(binned_peak.index)),H_H_monofit_NBQX(x,popt)+onset_current,linestyle = "--",color='red',label = "fit")
    hhfitaxis4.grid(False)
    add_scalebar(hhfitaxis4,loc='center right')
    hhfitfig4.tight_layout()

    
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
    

def exp_fitter(pandas_structure, double = False,fitflag=True,B_seed_val=False,show_components=False):
    """Fits an exponential or sum of expontials to pandas structure data (either dataframe or series).
    If DataFrame is used, the fit is performed to the average of all waves
    
    if double = True (Default = False), fits a double exponential
    
    fitflag = True, the fit is plotted against the data
    
    if B_seed value is set to a value, this will be used to seed the double exponential
        and the vlaue for the peak current will be used to seed A (detected)
    
    if show_component = True, the components of the double exponential will
    also be plotted
    
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
        x[1,:] = It - It.iloc[-1] #current
        if end_fit[1] > start_fit[1]:# condition to fit positive exponential
            #p0 = (times[-1]/It.iloc[0])/(np.log(Imax)) # estimate Tau
            posmonoexp = lambda x,a,tau: a*(np.exp(x[0,:]/tau))
            #if np.isinf(p0) is True:
            popt,pcov = sp.optimize.curve_fit(posmonoexp,x,x[1,:])
            mtype = "positive"
            if fitflag:
                plt.plot(x[0,:]+It.index[0],posmonoexp(x,popt[0],popt[1])+It.iloc[-1],linestyle ='--',color='red',label = 'fit')
            plt.plot(It.index,It,color='black',label='data')
            
        else: # or else, fit negative exponential
            #p0 = -(It.index[-1]/It.iloc[-1])/(np.log(Imax)) #deprecated together with np.isinf codition. No initial guess now provided
            negmonoexp = lambda x,a,tau: a*(np.exp(-x[0,:]/tau))
            popt,pcov = sp.optimize.curve_fit(negmonoexp,x,x[1,:])
            mtype = "negative"
            if fitflag:
                 plt.plot(x[0,:]+It.index[0],negmonoexp(x,popt[0],popt[1])+It.iloc[-1],linestyle ='--',color='red',label = 'fit')
            plt.plot(It.index,It,color='black',label= 'data')
           
            
        Tau = np.abs(popt[1])
        print("A {} monoexponential function was fitted,a={} from fit end,Tau = {}".format(mtype,popt[0],Tau))
        plt.show()
        print("1SD errors = {}".format(np.sqrt(np.diag(pcov))))
        return(popt[0],Tau)
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
        x[1,:] = It - It.iloc[-1] # current
        if end_fit[1]>start_fit[1]: # condition fit positive exponential
             posdouble_exp = lambda x,a,b,taufast,tauslow: (a*(np.exp(x[0,:]/taufast)))+(b*(np.exp(x[0,:]/tauslow)))
             if np.any(B_seed_val):
                 popt,pcov = sp.optimize.curve_fit(posdouble_exp,x,x[1,:],p0=[x[1,0],B_seed_val,1,1])
             else:
                 popt,pcov = sp.optimize.curve_fit(posdouble_exp,x,x[1,:])
             mtype = "positive"
             plt.plot(x[0,:]+It.index.min(),x[1,:],color='black',label = "Data")
             if fitflag:
                 plt.plot(x[0,:]+It.index.min(),posdouble_exp(x,popt[0],popt[1],popt[2],popt[3]+It.iloc[-1]),linestyle ="--",color='red',label = 'Fit')
             if show_components:
                 posmonoexp = lambda x,a,tau: a*(np.exp(x[0,:]/tau))
                 plt.plot(x[0,:]+It.index.min(),posmonoexp(x, popt[0], popt[2]),label='Fast component',color='lightskyblue',linestyle="--")
                 plt.plot(x[0,:]+It.index.min(),posmonoexp(x, popt[1], popt[3]),label='Slow component',color='royalblue',linestyle="--")
             plt.legend()
        else:
             negdouble_exp = lambda x,a,b,taufast,tauslow: (a*(np.exp(-x[0,:]/taufast)))+(b*(np.exp(-x[0,:]/tauslow)))
             if np.any(B_seed_val):
                 popt,pcov = sp.optimize.curve_fit(negdouble_exp,x,x[1,:],p0=[x[1,0],B_seed_val,1,1])
             else:
                 popt,pcov = sp.optimize.curve_fit(negdouble_exp,x,x[1,:])
             mtype = "negative"
             plt.plot(x[0,:]+It.index.min(),x[1,:]+It.min(),color='black',label = "Data")
             if fitflag:
                 plt.plot(x[0,:]+It.index.min(),negdouble_exp(x,popt[0],popt[1],popt[2],popt[3])+It.iloc[-1],linestyle ="--",color='red',label = 'Fit')
             if show_components:
                 negmonoexp = lambda x,a,tau: a*(np.exp(-x[0,:]/tau))
                 plt.plot(x[0,:]+It.index.min(),negmonoexp(x, popt[0], popt[2]),label='Fast component',color='lightskyblue',linestyle="--")
                 plt.plot(x[0,:]+It.index.min(),negmonoexp(x, popt[1], popt[3]),label='Slow component',color='royalblue',linestyle="--")
             plt.legend()
        # correcting time constant and amplitude values for their real domain
# =============================================================================
######===========================================######
# reinstated partially: 160321:
    #returning tau as absolute value
    # returning absolute value of A, rather than relative to B
        A = popt[1] + popt[0]
        B = np.abs(popt[1])
        Taufast = np.abs(popt[2])
        Tauslow =  np.abs(popt[3])
        weightedtau = (Taufast*(A/(A+B))) + (Tauslow*(B/(A+B)))
# =============================================================================
        #weightedtau = (popt[2]*(popt[0]/(popt[0]+popt[1]))) + (popt[3]*(popt[1]/(popt[0]+popt[1])))
        print("The weighted time constant is = {}".format(weightedtau))
        print("A sum of {} exponentials was fitted. A = {} from fit end, B = {}, Taufast = {},Tauslow={}".format(mtype,A,B,Taufast,Tauslow))
        print("1SD errors = {}".format(np.sqrt(np.diag(pcov))))
        plt.show()
        return(A,B,Taufast,Tauslow,weightedtau)
#____________________________________________________________________________#
#----------------------------- Data Saving ----------------------------------#
rootpath = os.path.dirname(os.path.realpath(__file__)) # getting module root
##### setting up file storage and analysis method logging
print("\n \n \n Outputs will be saved to: {}".format(rootpath))
for root,dirs,files in os.walk(rootpath):
    if "JAMPACK_Log.csv" in files:
        savepath = rootpath + '/JAMPACK_log.csv'
        print('JAMPACK log exists and will be updated on instances of save_file()')
    else:
        with open("JAMPACK_log.csv","w") as savepath:
            savepath = rootpath +'/savepath.name'
            print('JAMPACK log has been created. This message will not be repeated.')
            pass

def setup_saving():
    setup_saving.on = True

setup_saving() # turning on ability to save when script run

def save_file(kwargs): # giving option to save global environment from an analyses
    """Stores the output of a function called in kwargs to the JAMPACK_Log.csv
    
    usage example:
        save_file(NBQX_unbinding(dataframe,open_tip response))"""
    if setup_saving.on:
        save_file.apply_saving = True
        save_file.experiment_id = input("Input the experiment ID >")
    
# for each function, write a save file portion using flags from save_file that is used when save_file.apply_saving  = True:
    # experiment_id [done, in save_file]
    # File_id [from import], ability to add multiple/ overwrite
    # condtion, needs to be entered... this one tricky
    # patch_id [from history], just numbering 1-n
    # protocol_id [from anaysis method selected]
    # stats [from protocol] used, key should say what the stats are:
    # Info: if baselining, if cleaned etc.
    # graphs path: for each graph,what is the path to it inside a created graphs folder

# save all as dictionary:
    # can use dir(save_file) to get list of methods associated with save_file object
        # my contain apply, so can save those methods that contain apply
    
# may be easier to have sep function for open_tip to give the flag

# at moment, for each root/dirs, JAMPACK created when its not in that root.


# Then need open_records, which should:
    # import dictionary
    # return list of keys for each 'column' identifier
# Then user can filter data
# and have function that will take this dictionary and return it as pandas dataframe.



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

def shared_N_noise_func(x,N,i_m60,i_m40,i_m20,i_0,i_20,i_40):
    """
    x: current decays concatenated along axis 1, where rows are:
        x[0,:] = I_mean at each sample
        x[1,:] = background variance at each sample
    
    Will work as long as number of samples of each wave are same such that
    x[0,0:n_samples] is I_mean for one voltage, and x[0,y:2n_samples] is next for all yn_samples with y voltages
    
    make +60 values optional arg
    
    Noise functions with shared N are returned

    """
    num_bins=20
    # n_samples = x.shape[1]/6
    # return(np.concatenate((noise_func(x[:,:n_samples],N,i_m60),noise_func(x[:,n_samples:2*n_samples],N,i_m40),noise_func(x[:,2*n_samples:3*n_samples],N,i_m20),noise_func(x[:,3*n_samples:4*n_samples],N,i_0),noise_func(x[:,4*n_samples:5*n_samples],N,i_20),noise_func(x[:,5*n_samples:6*n_samples],N,i_40))))
    #return(np.concatenate((noise_func(x,N,i_m60),noise_func(x,N,i_m40),noise_func(x,N,i_m20),noise_func(x,N,i_0),noise_func(x,N,i_20),noise_func(x,N,i_40))))
    return(np.concatenate((noise_func(x[:,:num_bins],N,i_m60),noise_func(x[:,num_bins:2*num_bins],N,i_m40),noise_func(x[:,2*num_bins:3*num_bins],N,i_m20),noise_func(x[:,3*num_bins:4*num_bins],N,i_0),noise_func(x[:,4*num_bins:5*num_bins],N,i_20),noise_func(x[:,5*num_bins:6*num_bins],N,i_40))))

def shared_N_noise_fun_to_60(x,N,i_m60,i_m40,i_m20,i_0,i_20,i_40,i_60):
    """
    x: current decays concatenated along axis 1, where rows are:
        x[0,:] = I_mean at each sample
        x[1,:] = background variance at each sample
    
    Will work as long as number of samples of each wave are same such that
    x[0,0:n_samples] is I_mean for one voltage, and x[0,y:2n_samples] is next for all yn_samples with y voltages
    
    make +60 values optional arg
    
    Noise functions with shared N are returned

"""
    num_bins=20
    #return(np.concatenate((noise_func(x,N,i_m60),noise_func(x,N,i_m40),noise_func(x,N,i_m20),noise_func(x,N,i_0),noise_func(x,N,i_20),noise_func(x,N,i_40),noise_func(x,N,i_60))))
    return(np.concatenate((noise_func(x[:,:num_bins],N,i_m60),noise_func(x[:,num_bins:2*num_bins],N,i_m40),noise_func(x[:,2*num_bins:3*num_bins],N,i_m20),noise_func(x[:,3*num_bins:4*num_bins],N,i_0),noise_func(x[:,4*num_bins:5*num_bins],N,i_20),noise_func(x[:,5*num_bins:6*num_bins],N,i_40),noise_func(x[:,6*num_bins:7*num_bins],N,i_60))))

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

from matplotlib.offsetbox import AnchoredOffsetbox
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, barcolor="black", barwidth=None, 
                 **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, ec=barcolor, lw=barwidth, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, ec=barcolor, lw=barwidth, fc="none"))

        if sizex and labelx:
            self.xlabel = TextArea(labelx, minimumdescent=False)
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely)
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)

def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)
    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)
    if hidex and hidey: ax.set_frame_on(False)
    return sb
