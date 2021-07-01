# EPyPhys
##Analysis Routines for Patch Clamp Electrophysiology files
-----
Full Illustrative page coming soon
-----
EPyPhys was borne of necessity. Existing softwares for analysing excised patch data are built largely on those designed to handle whole-cell data, such as EPSCs or minis. This often feels as though the software uses the signal UI as a barrier between an experimenter and their data. In-built routines are often hidden, creating black box scenarios during processing. Creation of custom routines often require learning instance-specific (useless) macro language. Finally, data cleaning and merging routines in UI type softwares leave a lot to be desired. </br>  
EPyPhys:
* is accessible : It is Python based. Python is the consistently fastest growing programming language. It is easy to learn. It is mostly based in pandas and numpy. All custom wrangling can be performed by using these (frequently used) packages. They are beautifully written, fast, and easy to learn. EpyPhys uses a functional programming philosophy, rather than the more standard object-orientated practice. This allows users to build on, and integrate existing functions without learning about unique class structures (which would defeat the point of them not learning a macro language in other software) and also allows use of help()
* Focuses on the cleaning of data and quality control to minimise loss: Processing out electrical interference from a phone signal that appeared in a few time-separated waves is easy. From import to figure should not take long.
* Performs standard routines: e.g. loading data, baselining, merging different sweeps together, merging different iterations of the same protocol, splitting sweeps by timepoint, aligning data, normalising data
* Through scipy extension, allows any custom function to be written in using pythonic syntax. No more dragging, no more terrible macro language.
* Through matplotlib and seaborn extensions (with access to other object-orientated packages and styles) allows custom publication-quality figure creation within the same software.
* Incorporates smart means for common analysis methods for ligand-gated channels: Exponential fitting, noise analysis etc.
</br>
**EPyPhys is written using principles of functional programming. To understand how to use a function, type help(function_name) to return its documentation.** 
</br> 
To get started, we want to convert your data into a pandas format. Pandas is a table format that has many existing methods. It uses the default DataFrame class, which will appear familiar to excel users, since it follows an index row and column convention. The existing routine for this is used fairly under the terms its existing MIT license to import ABF type files. Other excellent projects address how to import files of other types, and the user can implement those file structures themselves.
</br>
To load in some data, use the function 'load_data()'. A file can be dragged into the brackets.</br>

'imported data = load_data('filename')'
</br>
Your data will be stored in a DataFrame where each row corresponds to a sample of the sweep in a column. The rows are indexed so we cna see the time point.
</br>
I said that quality control was the focus, and so as well as being presented with the graph of all imported sweeps, you will also see a boxplot contianing a separate box for each sweep. This cna be useful for identifying rogue sweeps that you may want to remove, or current rundown.
</br>  
You may want to perform an intial inspection of the data - such as examining the mean. This can be done using pandas methods, which make quick plotting super easy.
'data.mean(axis=1).plot()'
</br>  
Now lets say that you have five records you want to merge together, e.g. for a single patch. Or you want to analyse all the data from multiple records together. This can be accomplished with merge 
</br>
'merged_data = merge((data1,data2,data3))' using double bracket notation as a tuple must be passed.
</br>  
In freely available acquisition softwares, incremented style protocols (e.g. where each record is one instance of the protocol, but at each sweep, the voltage increases by 10mV) cna be a pain to merge. Use merge protocols to perform this
'merged_protocols = merge_protocols((data1,data2,data3))' agains, using double bracket notation.
</br>  
Baseline data by:
'baselined_data = baseline(data,time_to_baseline_until)
</br>  
IF your sweep contains multiple protocols, e.g. fast jumps of different lengths, such as in a recovery from desensitisation protocol, you may want to isolate each component. Easy
'Intervalone,intervaltwo = split_sweeps(data,times_to _split)' time_to_split can be a tuple of split times, e.g. 'split_sweeps(data,times_to_split = (0.004343,332525)'
</br>  

And what if you want to perform operations on records, e.g. subtract a record containing waves of an artefact?
use 'subtracted = subtract_avg(data,wave_to_subtract)
</br>  
or normalise some waves:
'normalised_data = normalise_currents(data)'
</br>

Waves can also be aligned either at a givne time point or amplitude
'time_aligned = align_at_t(data)'
'amplitude_aligned = align_at_a(data)'
</br>
What if you want to remove some sweeps?
This can be performed iteratively using 'cleaning_data'
'sweeps_to_use,sweeps_to_remove' = cleaning_data(data)
 You will be presented with the open to remove some sweeps, shown a plot of the sweeps grouped, each group average, the average of all used waves, and the average of all rmeoved waves. This cna be useful for isolating that one pesky sweep that contains interference.
 </br>  
 Current analysis methods that are implemented are popular for ligand-gated channels, and a further example of a more-niche type is included. The user should feel free to write their own. All currently use interactive plotting methods and guide the user in their implementation.
 </br>
 'NSFA_optimal()' uses a machine learning approach, together with the Heinneman and Conti 7RMSE criterion to perform non-statioanry fluctuation analysis of the Sigg style.
 'NSFA()' is an instance referred to by the above that cna be used for more general noise analysis.
 </br>  
 'exp_fitter()' allows fitting fo mono or sum of exponential components to some data. It will detect whether these should be positiv eor negative exponentials.
 </br>  
 'NBQX_unbinding()' is a special form of poisson jump analysis used for the glutamate channels (Rosenmund (1998)). Together with the above, it should guide the user on implementing custom function fits to various types of data.
 
/<br>
And that's all for now. Current WIP include data logging and other items raised as issues.
