# import statements
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import copy

#################################### Functions for loading in and plotting the CHORD data ##########################

# parse the files

def parsetsv(filename):
    # opens file, pulls contents into list of dicts per line
    data = []
    with open(filename, 'r') as file:
        content = file.readlines()
    
    # cut lines that are commented out with # in the file
    startindx = 0
    while True:
        if content[startindx][0][0] == '#':
            startindx += 1
            
        else:
            break
            
    # pull labels. strip gets rid of extraneous fields
    labels = content[startindx].strip().split('\t')
    # remove labels from content (for convenience)
    content = content[startindx+1:]
    for ln in content:
        splitline = ln.strip('\n').split('\t') # have to use [:-1] here so we don't eliminate unfilled trailing fields
        data.append({labels[kk]:splitline[kk] for kk in range(len(labels))})
    
    return data

# define the plotting function

def plotfull(patient,test,ptdetails,xlim=None,minorgridlines=False,ylim=None):
    # plots the full panel of information for the given patient and test, using information from ptdetails that we pulled from the files
    
    # extract the patient we are interested in
    workingpt = ptdetails[patient]
    
    # get test and treatment data (abstract this out so we can use it when doing fits)
    (sampletimes,samplevalues,units,treatments,startdates,stopdates) = get_test_treatment(workingpt,test)
    
    # extract surgery
    surgerydates = workingpt['SURGERY_DATES']
    
    # extract RT
    RTdates = workingpt['RT_DATES']
    
    # extract progression
    progressiondates = workingpt['PROGRESSION_DATES']
    progressionvalues = workingpt['PROGRESSION_VALUES']
    
    #extract imaging
    imagingdates = workingpt['IMAGING_DATES']
    hascancer = workingpt['HAS_CANCER']
    
    return plotdata(patient, test, sampletimes, samplevalues, units, treatments, startdates, stopdates, surgerydates=surgerydates, RTdates=RTdates, progressiondates=progressiondates, progressionvalues=progressionvalues, imagingdates=imagingdates, hascancer=hascancer, logflag=True, xlim=xlim, minorgridlines=minorgridlines, ylim=ylim)

def plotdata(patient, test, sampletimes, samplevalues, units, treatments, startdates, stopdates, surgerydates=[], RTdates=[], progressiondates=[], progressionvalues=[], imagingdates=[], hascancer=[], logflag=False, xlim=None, minorgridlines=False, ylim=None):
    # returns the fig that plots the extracted data series from the files
    # test series is plotted on main plot, drugs and imaging in separate small plot.
    
    # set up the figure
    if len(set(treatments)) < 15:
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1]}, figsize=(10, 8), sharex=True)
    else:
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,1]}, figsize=(10, 8), sharex=True)
    
    # plot the test series
    plt.sca(axs[0])
    axs[0].plot(sampletimes,samplevalues,'k.-',linewidth=3,markersize=12,label=test)
    axs[0].tick_params(axis='x', labelbottom=True)
    axs[1].set_xlabel('time [d]',fontsize=12)
    axs[0].set_ylabel(test + ' [' + units + ']',fontsize=12)
    axs[0].set_title(patient,fontsize=14)
    
    # plot the treatments
    plt.sca(axs[1])
    
    # first find the unique ones and set up the labels/colormap
    uniquetreatments = []
    txlabels = []
    for tt in treatments:
        # if treatment is new, add it with a label
        if tt not in uniquetreatments:
            # not there, assign label
            uniquetreatments.append(tt)
            txlabels.append(tt)
        else:
            txlabels.append(None)
        
    # make color map
    # Choose a colormap
    cmap_name = 'viridis'
    cmap = colormaps[cmap_name]

    # Number of colors to extract
    num_colors = len(uniquetreatments)

    # Generate evenly spaced values between 0 and 1
    color_indices = np.linspace(0, 1, num_colors)

    # Extract colors from the colormap
    colors = [cmap(x) for x in color_indices]
    colordict = {tx:cc for tx,cc in zip(uniquetreatments,colors)}
    
    # now plot them all
    ypositiondict = {uniquetreatments[kk]:kk for kk in range(len(uniquetreatments))}
    startlabel = 'Treatment Start'
    for kk in range(len(treatments)):
        # plot it
        #axs[1].axvspan(startdates[kk],stopdates[kk],color=colordict[treatments[kk]],label=txlabels[kk])
        rect = patches.Rectangle((startdates[kk], ypositiondict[treatments[kk]] + 1/2), stopdates[kk] - startdates[kk], 1, linewidth=2, edgecolor='k', facecolor=colordict[treatments[kk]],label=txlabels[kk])
        axs[1].add_patch(rect)
        
        # plot the startdates on the main plot, as well as the durations
        axs[0].axvline(x=startdates[kk],color='tab:gray',linestyle='--',linewidth=2,label=startlabel,alpha=0.5)
        startlabel = None
        axs[1].axvline(x=startdates[kk],color='tab:gray',linestyle='--',linewidth=2,alpha=0.5)
        
    # set yticks in second plot to be the rectangle endpoints
    axs[1].set_yticks([kk+1/2 for kk in range(len(uniquetreatments)+1)])
    
    # label the rectangles
    # Set minor ticks halfway between major ticks
    minor_ticks = [kk+1 for kk in range(len(uniquetreatments))]
    axs[1].set_yticks(minor_ticks, minor=True)

    # Set labels for minor ticks
    minor_tick_labels = [ut[:min(5,len(ut))] for ut in uniquetreatments]
    axs[1].set_yticklabels(minor_tick_labels, minor=True)

    # Hide major tick labels and minor ticks
    axs[1].tick_params(axis='y', which='major', labelleft=False)
    axs[1].tick_params(axis='y', which='minor', length=0, labelright=True)
    
    # plot when on treatment
    otlabel = "On Treatment"
    regions = [[startd,stopd] for startd,stopd in zip(startdates,stopdates)]
    
    # sort them by startdate
    regions = sorted(regions)
    toplotregions = [regions[0]]
    for kk in range(1,len(regions)):
        if regions[kk][0] >= toplotregions[-1][1]:
            # next start date is after last end date that we have parsed so far
            # therefore, it is a new, later region
            toplotregions.append(regions[kk])
            
        if regions[kk][0] < toplotregions[-1][1]:
            # next start date is before last end date
            if regions[kk][1] >= toplotregions[-1][1]:
                # and next end date is after current end date
                # update current end date
                toplotregions[-1][1] = regions[kk][1]
                
            # else do nothing, because this treatment is entirely taken up by the previous one
            
    # now plot them
    for tpr in toplotregions:
        axs[0].axvspan(tpr[0],tpr[1],color='b',alpha=0.1,label=otlabel)
        otlabel = None
        axs[1].axvspan(tpr[0],tpr[1],color='b',alpha=0.1)
    
    if len(surgerydates) > 0:
        # extract the surgery dates
        slabel = 'Surgery Procedure'
        for sd in surgerydates:
            axs[1].axvline(x=sd, color='tab:orange', linestyle='-', linewidth=2, label=slabel)
            slabel = None
            axs[0].axvline(x=sd, color='tab:orange', linestyle='-', linewidth=2)
                
    # now RT
    if len(RTdates) > 0:
        # extract the surgery dates
        RTlabel = 'Radiotherapy'
        for rd in RTdates:
            axs[1].axvline(x=rd, color='tab:green', linestyle='-', linewidth=2, label=RTlabel)
            RTlabel = None
            axs[0].axvline(x=rd, color='tab:green', linestyle='-', linewidth=2)
    
    # Progression data
    if len(progressiondates) > 1:
        plabel = 'Progression'
        for pd,pv in zip(progressiondates,progressionvalues):
            if pv == 'Y':
                axs[1].plot([pd],[0],'x',color='tab:brown',label=plabel)
                plabel = None
    
    # imaging data
    if len(imagingdates) > 0:
        dlabel = 'Imaging'
        hcylabel = 'Has cancer? Yes'
        hcnlabel = 'Has cancer? No'
        for dd,hc in zip(imagingdates,hascancer):
            axs[1].plot([dd],[-1],'c.',label=dlabel)
            dlabel = None
                
            if hc == 'Y':
                axs[1].plot([dd],[-1/2],'r.',label=hcylabel)
                hcylabel = None
            
            if hc == 'N':
                axs[1].plot([dd],[-1/2],'g.',label=hcnlabel)
                hcnlabel = None

    # put the axis in a nice place    
    lines_labels = [ax.get_legend_handles_labels() for ax in axs]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(1, 0.9))
    axs[0].grid(True,axis='y')
    axs[1].grid(True,axis='y')
    if minorgridlines:
        axs[0].grid(True,axis='y',which='minor', linestyle=':', linewidth='0.5', color='black')
        
    if logflag:
        axs[0].set_yscale('log')

    if xlim:
        axs[0].set_xlim(xlim)

    if ylim:
        axs[0].set_ylim(ylim)
        
    return fig

# Also define the function to parse out and return the test and treatment params

def get_test_treatment(workingpt,test):
    # returns the test and treatment values for the given patient
    
    # get the data series that we are interested in 
    if test in ['PSA','CEA','CA15-3','CA19-9']:
        sampletimes = workingpt[test + '_DATES']
        samplevalues = workingpt[test + '_VALUES']
        units = workingpt[test + '_UNITS']
    else:
        raise Exception('Test not recognized. Please use a valid test!')
        
    if len(samplevalues) == 0:
        raise Exception('Patient ID has no lab test entries. Please enter a valid patient ID!')
        
    # extract the treatments
    treatments = workingpt['TREATMENTS']
    startdates = workingpt['START_DATES']
    stopdates = workingpt['STOP_DATES']
    
    return (sampletimes,samplevalues,units,treatments,startdates,stopdates)

def printpatientsummary(patient,ptdt):
    # given the patient's information, print the details.
    print("Summary for patient " + patient + ', ' + ptdt['OS_STATUS'][2:] + ' (day {}).'.format(int(ptdt['OS_MONTHS'] * 30.42)))
    print('Diagnosed at {} with '.format(ptdt['PRIMARY_DIAGNOSIS']['DATE']) + ptdt['CANCER_TYPE'] + '. Specifically, ' + ptdt['CANCER_TYPE_DETAILED'] + '.')
    if ptdt['HR'] == 'Yes':
        print('HR+')
        
    if ptdt['HR'] == 'No':
        print('HR-')
        
    if ptdt['HER2'] == 'Yes':
        print('HER2+')
        
    if ptdt['HER2'] == 'No':
        print('HER2-')
        
    print('Histology description: ' + ptdt['ICD_O_HISTOLOGY_DESCRIPTION'] + '.')
    print('Diagnosis description: ' + ptdt['PRIMARY_DIAGNOSIS']['DX_DESCRIPTION'] + '. ' + ptdt['PRIMARY_DIAGNOSIS']['STAGE'] + '. ' + ptdt['PRIMARY_DIAGNOSIS']['SUMMARY'] + '.')
    print('Other staging information: AJCC = ' + ptdt['PRIMARY_DIAGNOSIS']['AJCC'] + '. Clinical group = ' + ptdt['PRIMARY_DIAGNOSIS']['CLINICAL_GROUP'] + '. Path group = ' + ptdt['PRIMARY_DIAGNOSIS']['PATH_GROUP'] + '.\n')
    return

# function for finding active treatments on a given day

def activetreatments(day,treatments,startdates,stopdates):
    return {tm for tm,start,stop in zip(treatments,startdates,stopdates) if day >= start and day <= stop}

## load in the data
def load_CHORD_data(filedir):
    # load in all files and return the patient details dictionary

    # files to be loaded in
    clinicalpatientfile = filedir + 'data_clinical_patient.txt'
    clinicalsamplefile = filedir + 'data_clinical_sample.txt'
    diagnosisfile = filedir + 'data_timeline_diagnosis.txt'
    ca153file = filedir + 'data_timeline_ca_15-3_labs.txt'
    ca199file = filedir + 'data_timeline_ca_19-9_labs.txt'
    ceafile = filedir + 'data_timeline_cea_labs.txt'
    priormedsfile = filedir + 'data_timeline_prior_meds.txt'
    progressionfile = filedir + 'data_timeline_progression.txt'
    psafile = filedir + 'data_timeline_psa_labs.txt'
    RTfile = filedir + 'data_timeline_radiation.txt'
    surgeryfile = filedir + 'data_timeline_surgery.txt'
    treatmentfile = filedir + 'data_timeline_treatment.txt'
    hascancerfile = filedir + 'data_timeline_cancer_presence.txt'
    tumorsitesfile = filedir + 'data_timeline_tumor_sites.txt'

    # load and parse
    clinicalpatientdata = parsetsv(clinicalpatientfile)
    clinicalsampledata = parsetsv(clinicalsamplefile)
    diagnosisdata = parsetsv(diagnosisfile)
    ca153data = parsetsv(ca153file)
    ca199data = parsetsv(ca199file)
    ceadata = parsetsv(ceafile)
    priormedsdata = parsetsv(priormedsfile)
    progressiondata = parsetsv(progressionfile)
    psadata = parsetsv(psafile)
    RTdata = parsetsv(RTfile)
    surgerydata = parsetsv(surgeryfile)
    treatmentdata = parsetsv(treatmentfile)
    hascancerdata = parsetsv(hascancerfile)
    tumorsitesdata = parsetsv(tumorsitesfile)

    # create a dictionary of patient summary statitstics and attributes for easy filtering
    attributedefaults = {'CANCER_TYPE':'', 'CANCER_TYPE_DETAILED':'', 'ICD_O_HISTOLOGY_DESCRIPTION':'', 'NUM_PSA':0, 'NUM_CEA':0, 'NUM_CA15-3':0, 'NUM_CA19-9':0, 'TREATMENTS':[], 'START_DATES':[], 'STOP_DATES':[], 'PRIOR_MED_TO_MSK':'', 'OS_STATUS':'', 'HR':'', 'HER2':'', 'NUM_ICDO_DX':0, 'PSA_VALUES':[], 'PSA_DATES':[], 'PSA_UNITS':'', 'CEA_VALUES':[], 'CEA_DATES':[], 'CEA_UNITS':'', 'CA15-3_VALUES':[], 'CA15-3_DATES':[], 'CA15-3_UNITS':'', 'CA19-9_VALUES':[], 'CA19-9_DATES':[], 'CA19-9_UNITS':'', 'TREATMENT_COMBOS':[], 'RT_DATES':[], 'NUM_RT':0, 'SURGERY_DATES':[], 'NUM_SURGERY':0, 'COMBO_START_DATES':[], 'COMBO_STOP_DATES':[], 'HAS_CANCER':[], 'PROGRESSION_DATES':[], 'PROGRESSION_VALUES':[], 'IMAGING_DATES':[], 'PRIMARY_DIAGNOSIS':dict(), 'TUMOR_SITES_DATES':[], 'TUMOR_SITES_VALUES':[], 'IMAGING_MODALITY':[], 'TREATMENT_TYPES':[], 'OS_MONTHS':-1, 'TUMOR_SITES_SUMMARY_DATES':[], 'TUMOR_SITES_SUMMARY':[]}

    patients = [cpd['PATIENT_ID'] for cpd in clinicalpatientdata]
    ptdetails = {pt:copy.deepcopy(attributedefaults) for pt in patients} # each patient has a dictionary with the useful information

    # set up a dictionary to tell what each type of drug is
    drugtypes = dict()

    # take data from clinicalpatientdata
    for entry in clinicalpatientdata:
        ptdetails[entry['PATIENT_ID']]['NUM_ICDO_DX'] = float(entry['NUM_ICDO_DX'])
        ptdetails[entry['PATIENT_ID']]['PRIOR_MED_TO_MSK'] = entry['PRIOR_MED_TO_MSK']
        ptdetails[entry['PATIENT_ID']]['OS_STATUS'] = entry['OS_STATUS']
        ptdetails[entry['PATIENT_ID']]['OS_MONTHS'] = float(entry['OS_MONTHS'])
        ptdetails[entry['PATIENT_ID']]['HR'] = entry['HR']
        ptdetails[entry['PATIENT_ID']]['HER2'] = entry['HER2']
    
    # take data from clinicalsampledata
    for entry in clinicalsampledata:
        ptdetails[entry['PATIENT_ID']]['CANCER_TYPE'] = entry['CANCER_TYPE']
        ptdetails[entry['PATIENT_ID']]['CANCER_TYPE_DETAILED'] = entry['CANCER_TYPE_DETAILED']
        ptdetails[entry['PATIENT_ID']]['ICD_O_HISTOLOGY_DESCRIPTION'] = entry['ICD_O_HISTOLOGY_DESCRIPTION']
    
    # take data from PSA data
    for entry in psadata:
        ptdetails[entry['PATIENT_ID']]['NUM_PSA'] += 1
        ptdetails[entry['PATIENT_ID']]['PSA_VALUES'].append(float(entry['RESULT']))
        ptdetails[entry['PATIENT_ID']]['PSA_DATES'].append(float(entry['START_DATE']))
        if ptdetails[entry['PATIENT_ID']]['PSA_UNITS'] == '':
            ptdetails[entry['PATIENT_ID']]['PSA_UNITS'] = entry['LR_UNIT_MEASURE'].strip()
    
    # take data from CEA data
    for entry in ceadata:
        ptdetails[entry['PATIENT_ID']]['NUM_CEA'] += 1
        ptdetails[entry['PATIENT_ID']]['CEA_VALUES'].append(float(entry['RESULT']))
        ptdetails[entry['PATIENT_ID']]['CEA_DATES'].append(float(entry['START_DATE']))
        if ptdetails[entry['PATIENT_ID']]['CEA_UNITS'] == '':
            ptdetails[entry['PATIENT_ID']]['CEA_UNITS'] = entry['LR_UNIT_MEASURE'].strip()
    
    # take data from CA15-3 data
    for entry in ca153data:
        ptdetails[entry['PATIENT_ID']]['NUM_CA15-3'] += 1
        ptdetails[entry['PATIENT_ID']]['CA15-3_VALUES'].append(float(entry['RESULT']))
        ptdetails[entry['PATIENT_ID']]['CA15-3_DATES'].append(float(entry['START_DATE']))
        if ptdetails[entry['PATIENT_ID']]['CA15-3_UNITS'] == '':
            ptdetails[entry['PATIENT_ID']]['CA15-3_UNITS'] = entry['LR_UNIT_MEASURE'].strip()
    
    # take data from CA19-9 data
    for entry in ca199data:
        ptdetails[entry['PATIENT_ID']]['NUM_CA19-9'] += 1
        ptdetails[entry['PATIENT_ID']]['CA19-9_VALUES'].append(float(entry['RESULT']))
        ptdetails[entry['PATIENT_ID']]['CA19-9_DATES'].append(float(entry['START_DATE']))
        if ptdetails[entry['PATIENT_ID']]['CA19-9_UNITS'] == '':
            ptdetails[entry['PATIENT_ID']]['CA19-9_UNITS'] = entry['LR_UNIT_MEASURE'].strip()
        
    # take data from RTdata
    for entry in RTdata:
        ptdetails[entry['PATIENT_ID']]['NUM_RT'] += 1
        ptdetails[entry['PATIENT_ID']]['RT_DATES'].append(float(entry['START_DATE']))
    
    # take data from surgerydata
    for entry in surgerydata:
        if entry['SUBTYPE'] == 'PROCEDURE':
            ptdetails[entry['PATIENT_ID']]['NUM_SURGERY'] += 1
            ptdetails[entry['PATIENT_ID']]['SURGERY_DATES'].append(float(entry['START_DATE']))
        
    # take data from progressiondata
    for entry in progressiondata:
        ptdetails[entry['PATIENT_ID']]['PROGRESSION_DATES'].append(float(entry['START_DATE']))
        ptdetails[entry['PATIENT_ID']]['PROGRESSION_VALUES'].append(entry['PROGRESSION'])
    
    # take data from hascancerdata
    for entry in hascancerdata:
        ptdetails[entry['PATIENT_ID']]['IMAGING_DATES'].append(float(entry['START_DATE']))
        ptdetails[entry['PATIENT_ID']]['HAS_CANCER'].append(entry['HAS_CANCER'])
        ptdetails[entry['PATIENT_ID']]['IMAGING_MODALITY'].append(entry['PROCEDURE_TYPE'])
        
    # take data from tumorsitesdata
    for entry in tumorsitesdata:
        ptdetails[entry['PATIENT_ID']]['TUMOR_SITES_DATES'].append(float(entry['START_DATE']))
        ptdetails[entry['PATIENT_ID']]['TUMOR_SITES_VALUES'].append({'TUMOR_SITE':entry['TUMOR_SITE'], 'CHEST':entry['CHEST'], 'ABDOMEN':entry['ABDOMEN'], 'PELVIS':entry['PELVIS'], 'HEAD':entry['HEAD'], 'OTHER':entry['OTHER'], 'MODALITY':entry['SOURCE_SPECIFIC']})

    # re-order the tumorsites data so that it goes in chronological order
    # also make a list of sets of where the tumor was detected. This may be the primary data we can get out of tumorsites
    for pt in list(ptdetails.keys()):
        ptdt = ptdetails[pt]
        sortedlist = sorted([[dt,val] for dt,val in zip(ptdt['TUMOR_SITES_DATES'],ptdt['TUMOR_SITES_VALUES'])],key=lambda x: (x[0], x[1]['MODALITY']))
        ptdt['TUMOR_SITES_DATES'] = [doublet[0] for doublet in sortedlist]
        ptdt['TUMOR_SITES_VALUES'] = [doublet[1] for doublet in sortedlist]
        # now make a summary statistic for each
        detected_locations = [vv['TUMOR_SITE'] for vv in ptdt['TUMOR_SITES_VALUES']]
        for dt,loc in zip(ptdt['TUMOR_SITES_DATES'],detected_locations):
            if dt not in ptdt['TUMOR_SITES_SUMMARY_DATES']:
                # add it
                ptdt['TUMOR_SITES_SUMMARY_DATES'].append(dt)
                ptdt['TUMOR_SITES_SUMMARY'].append({loc})
            else:
                # it is there, just append to currently working set
                ptdt['TUMOR_SITES_SUMMARY'][-1].add(loc)
        
    # take data from diagnosisdata
    for entry in diagnosisdata:
        if entry['EVENT_TYPE'] == 'Diagnosis' and entry['SUBTYPE'] == 'Primary':
            ptdetails[entry['PATIENT_ID']]['PRIMARY_DIAGNOSIS']['DATE'] = float(entry['START_DATE'])
            ptdetails[entry['PATIENT_ID']]['PRIMARY_DIAGNOSIS']['DX_DESCRIPTION'] = entry['DX_DESCRIPTION']
            ptdetails[entry['PATIENT_ID']]['PRIMARY_DIAGNOSIS']['STAGE'] = entry['STAGE_CDM_DERIVED']
            ptdetails[entry['PATIENT_ID']]['PRIMARY_DIAGNOSIS']['AJCC'] = entry['AJCC']
            ptdetails[entry['PATIENT_ID']]['PRIMARY_DIAGNOSIS']['CLINICAL_GROUP'] = entry['CLINICAL_GROUP']
            ptdetails[entry['PATIENT_ID']]['PRIMARY_DIAGNOSIS']['PATH_GROUP'] = entry['PATH_GROUP']
            ptdetails[entry['PATIENT_ID']]['PRIMARY_DIAGNOSIS']['SUMMARY'] = entry['SUMMARY'].strip()
    
    # get treatment data
    for entry in treatmentdata:
        ptdetails[entry['PATIENT_ID']]['START_DATES'].append(float(entry['START_DATE']))
        ptdetails[entry['PATIENT_ID']]['STOP_DATES'].append(float(entry['STOP_DATE']))
        ptdetails[entry['PATIENT_ID']]['TREATMENTS'].append(entry['AGENT'])
        ptdetails[entry['PATIENT_ID']]['TREATMENT_TYPES'].append(entry['SUBTYPE'])
        drugtypes[entry['AGENT']] = entry['SUBTYPE']
    
    # sort PSA, CEA, CA15-3, and CA19-9 data by start date, just in case they are not already
    for pt in list(ptdetails.keys()):
        ptdt = ptdetails[pt]
        if ptdt['NUM_PSA'] > 0:
            # sort it
            sortedlist = sorted([(dt,val) for dt,val in zip(ptdt['PSA_DATES'],ptdt['PSA_VALUES'])])
            ptdt['PSA_DATES'] = [doublet[0] for doublet in sortedlist]
            ptdt['PSA_VALUES'] = [doublet[1] for doublet in sortedlist]
    
        if ptdt['NUM_CEA'] > 0:
            # sort it
            sortedlist = sorted([(dt,val) for dt,val in zip(ptdt['CEA_DATES'],ptdt['CEA_VALUES'])])
            ptdt['CEA_DATES'] = [doublet[0] for doublet in sortedlist]
            ptdt['CEA_VALUES'] = [doublet[1] for doublet in sortedlist]
        
        if ptdt['NUM_CA15-3'] > 0:
            # sort it
            sortedlist = sorted([(dt,val) for dt,val in zip(ptdt['CA15-3_DATES'],ptdt['CA15-3_VALUES'])])
            ptdt['CA15-3_DATES'] = [doublet[0] for doublet in sortedlist]
            ptdt['CA15-3_VALUES'] = [doublet[1] for doublet in sortedlist]
    
        if ptdt['NUM_CA19-9'] > 0:
            # sort it
            sortedlist = sorted([(dt,val) for dt,val in zip(ptdt['CA19-9_DATES'],ptdt['CA19-9_VALUES'])])
            ptdt['CA19-9_DATES'] = [doublet[0] for doublet in sortedlist]
            ptdt['CA19-9_VALUES'] = [doublet[1] for doublet in sortedlist]
        
    # sort treatments by start date, just in case they are not already
    for pt in list(ptdetails.keys()):
        ptdt = ptdetails[pt]
        sortedlist = sorted([(sdt,edt,trt) for sdt,edt,trt in zip(ptdt['START_DATES'],ptdt['STOP_DATES'],ptdt['TREATMENTS'])])
        ptdt['START_DATES'] = [trio[0] for trio in sortedlist]
        ptdt['STOP_DATES'] = [trio[1] for trio in sortedlist]
        ptdt['TREATMENTS'] = [trio[2] for trio in sortedlist]

    # put treatments into proper form, where TREATMENT_COMBOS gives a list of sets. Each set is the active treatments between COMBO_START_DATES and COMBO_STOP_DATES
    # to do so, let's find each active treatment for each day. then, let's combine everything to end up with the final list
    for pt in list(ptdetails.keys()):
        ptdt = ptdetails[pt]
        if len(ptdt['TREATMENTS']) > 0:
            fulltreatdates = list(range(int(min(ptdt['START_DATES'])),int(max(ptdt['STOP_DATES']))+1))
            fullactivetreatments = [activetreatments(day,ptdt['TREATMENTS'],ptdt['START_DATES'],ptdt['STOP_DATES']) for day in fulltreatdates]
            switchindices = [kk for kk in range(len(fullactivetreatments)-1) if fullactivetreatments[kk] != fullactivetreatments[kk+1]]
            combostartdates = [fulltreatdates[kk+1] for kk in switchindices]
            combostartdates.insert(0,fulltreatdates[0])
            combostopdates = [fulltreatdates[kk] for kk in switchindices]
            combostopdates.append(fulltreatdates[-1])
            combotreats = [fullactivetreatments[kk] for kk in switchindices]
            combotreats.append(fullactivetreatments[-1])
            # now remove the empty sets
            triples = [(treat,start,stop) for treat,start,stop in zip(combotreats,combostartdates,combostopdates) if treat != set()]
            # now do the assignment
            ptdt['TREATMENT_COMBOS'] = [triple[0] for triple in triples]
            ptdt['COMBO_START_DATES'] = [triple[1] for triple in triples]
            ptdt['COMBO_STOP_DATES'] = [triple[2] for triple in triples]

    return ptdetails

########################## Next part is for fitting the data #######################################

# we need the functional form to fit. Use the known solution
def knownsolution(t,T0,gamma,r,delta):
    volume = [T0 * np.exp(gamma * tt + delta * (np.exp(-r * tt) - 1) / r) for tt in t]
    return volume

# auxiliary function for plotting
def plotfit(tdata,data,tdatadense,densesolution,test,units,fignum=1,plottitle=None,reindexdates=0):
    # plots the fit, returns the fig
    if reindexdates:
        # then we want to undo reindexing to zero
        tdatadense = [tt + reindexdates for tt in tdatadense]
        tdata = [tt + reindexdates for tt in tdata]
    fig = plt.figure(fignum,figsize=(8,6))
    plt.plot(tdatadense,densesolution,'k-',linewidth=7,label='Fit')
    plt.plot(tdata,data,'r*',markersize=12,label='Actual')
    plt.ylabel(test + ' [' + units + ']',fontsize=20)
    plt.xlabel('Time [days]',fontsize=20)
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.yscale('log')
    if plottitle:
        plt.title(plottitle,fontsize=20)
        
    return fig

# auxiliary function for the log fit
def knownsolutionlog(t,T0,gamma,r,delta):
    # first argument of curve_fit for the log fitting function
    knownsol = knownsolution(t,T0,gamma,r,delta)
    return [np.log(ks) for ks in knownsol]

# new fits using the method of doing the log first then unlogging for plotting
def dofitslog(sampletimes,samplevalues,units,newtreats,newstartdates,newstopdates,gamma,test,plotflag=1,extended=0):
    # given gamma, fit r and delta for each simplified treatment with start and stop dates as specified here
    # save the params to arrays
    T0s = []
    rs = []
    deltas = []
    
    # let's fit the U's
    for indx in range(len(newstartdates)):
        tdata = []
        data = []
        for tt,sv in zip(sampletimes,samplevalues):
            if (tt >= newstartdates[indx] and tt <= newstopdates[indx]):
                tdata.append(tt - newstartdates[indx])
                data.append(sv)

        paramsguess = [data[0], 0.02,0.13] # T0, r, delta
        if indx == 1:
            paramsguess[1] = 0.004

        params,cov = curve_fit(lambda t,T0,r,delta: knownsolutionlog(t,T0,gamma,r,delta),tdata,[np.log(dd) for dd in data],p0=paramsguess,bounds=(0,np.inf))

        T0s.append(params[0])
        rs.append(params[1])
        deltas.append(params[2])

        # plot
        if plotflag:
            endt = tdata[-1]
            if extended:
                endt = newstopdates[indx] - newstartdates[indx]
                
            tdatadense = [tdata[0] + (endt - tdata[0]) * kk / 100 for kk in range(101)]
            densesolution = knownsolution(tdatadense,params[0],gamma,params[1],params[2])
            fitfig = plotfit(tdata,data,tdatadense,densesolution,test,units,fignum=1000+indx,plottitle=newtreats[indx],reindexdates=newstartdates[indx])
            plt.show(fitfig)
        
    return (T0s,rs,deltas)