# import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.optimize import root
from itertools import product
from random import randint
from random import seed # the random package is only used to draw random colors for plotting many different drugs. So, setting its seed to get a nice reproducible color palette is ok
from math import factorial
from itertools import combinations_with_replacement
import matplotlib.patches as patches
import pickle
import time
from scipy.optimize import fsolve
import copy
from sklearn.utils import Bunch

def GDRSode(t, y, treatmentschedule, timepercycle, gamma, delta, r, s):
    # treatmentschedule is a list of sets. The sets contain indices of the drugs being used
    # timepercycle is how long a set is applied for
    # gamma is the growth rate (number)
    # Ninf is the max tumor size (number)
    # delta is the drug kill parameters (list)
    # r is the resistance parameters (list)
    # s is the re-sensitization parameters (list)
    
    # unpack params
    N = y[0]
    E = y[1:]

    drug = [0 for kk in range(len(E))]
    cyclenumber = int(t // timepercycle) # index of treatmentschedule to use
    
    for indx in treatmentschedule[cyclenumber]:
        drug[indx] = 1
    
    # find derivatives
    dNdt = (gamma - sum([dl * dg * Ef for dl,dg,Ef in zip(delta,drug,E)])) * N 
    dEdt = [(s[kk] * (1 - E[kk]) * (1 - drug[kk]) - r[kk] * drug[kk]) * E[kk] for kk in range(len(E))]

    # for this application only, let's cut off the simulation if below curethreshold
    if N <= 1:
        dNdt = 0
        dEdt = [0 for kk in range(len(E))]
    
    return [dNdt,*dEdt]

######################### Code for enumerating the optimal schedule by brute force ################################

def generate_combos(gamma,ndrugs,deltahighmult=50,deltamidmult=25,deltalowmult=3/4,rhighmult=15,rlowmult=6,shighmult=45/4,slowmult=1/2,szeromult=0):
    # generate all combos of ndrugs for the given gamma
    # let's just do high/low for r and s
    # delta gets 3: ineffective, small effective, very effective. But, ineffective only gets rlow and slow (doesn't make a difference anyway)
    # need to return list with entries of this form: [gamma, deltas, rs, ss], where deltas is a ndrugs length list of the deltas for this drug, etc.
    deltas = [deltahighmult * gamma, deltamidmult * gamma]
    rs = [rhighmult * gamma, rlowmult * gamma]
    ss = [shighmult * gamma, slowmult * gamma,szeromult * gamma]
    onedrugcombos = [[dd,rr,sv] for dd in deltas for rr in rs for sv in ss] + [[deltalowmult * gamma, rlowmult * gamma, szeromult * gamma]]
    allcombos = product(onedrugcombos,repeat=ndrugs)
    #return [[gamma] + sum(list(draw),[]) for draw in allcombos]
    return [[gamma] + list(zip(*draw)) for draw in allcombos]

def num_schedules(ndrugs,nwindows):
    # returns the number of schedules with ndrugs that cannot be used together over nwindows with a required rest week in between
    return sum([ndrugs**kk * factorial(nwindows - kk + 1) / (factorial(kk) * factorial(nwindows - 2 * kk + 1)) for kk in range(1 + int(np.floor((nwindows + 1) / 2)))])

def enumerate_schedules_1(nwindows):
    # enumerate all schedules with one drug
    schedules = [[set() for kk in range(nwindows)]]
    for kk in range(1,int(np.floor((nwindows + 1) / 2)) + 1):
        # the base is what we are inserting empty weeks into
        # https://math.stackexchange.com/questions/3330180/combinatorics-problem-how-many-ways-this-row-can-be-filled
        base = [[{0}]] + [[set(),{0}] for jj in range(kk - 1)]
        # enumerate all ways to assign the remaining nwindows - 2 * kk + 1 empty blocks in the kk + 1 slots
        all_combinations = combinations_with_replacement([jj for jj in range(kk + 1)], nwindows - 2 * kk + 1)
        for combo in all_combinations:
            newaddition = []
            inserts = [[set() for mm in range(combo.count(jj))] for jj in range(kk+1)] # populate the spots to insert into
            # fold them together and append to schedules
            for jj in range(2 * kk + 1):
                if jj % 2 == 0:
                    # evens come from inserts
                    newaddition.append(inserts[int(jj / 2)])

                if jj % 2 == 1:
                    # evens come from base
                    newaddition.append(base[int((jj - 1) / 2)])

            schedules.append(sum(newaddition,[]))

    return schedules

def enumerate_schedules(ndrugs,nwindows):
    # enumerate all schedules possible for ndrugs (that cannot be used together) used in nwindows. 
    # required rest week after all drugs.
    # strategy is to enumerate all drug windows using enumerate_schedules_1, then assign drugs to each window
    fullschedules = []
    drugs = list(range(ndrugs))
    onedrugschedules = enumerate_schedules_1(nwindows)
    for schd in onedrugschedules:
        nusedwindows = schd.count({0})
        if nusedwindows == 0:
            fullschedules.append(schd)
            continue

        # if here, we use at least one drug
        indices = [ii for ii,xx in enumerate(schd) if 0 in xx] # get all indices that correspond to a used drug window
        # make all combos for nusedwindows uses of the ndrugs
        allcombos = product(drugs,repeat=nusedwindows)
        for cmb in allcombos:
            fullschedules.append([{cmb[indices.index(jj)]} if jj in indices else schd[jj] for jj in range(len(schd))])
            
    return fullschedules

def ode_sim(params,schedule,timepercycle=7):
    # simulate the GDRS ODEs for the given params and schedule
    T = [0,timepercycle * len(schedule) - 1]
    # find the number of drugs used
    ndrugs = len(params[1])
    y0 = [1e12] + [1 for kk in range(ndrugs)]
    # solve
    sol = solve_ivp(lambda t,y: GDRSode(t,y,schedule,timepercycle, params[0], params[1], params[2], params[3]),T,y0,max_step=timepercycle/100)
    return sol
    
def strategy_eval_full_sim(params,gamma,nwindows,metric='CURE',timepercycle=7):
    # generates all strategies and evaluates them on the given metric 
    # returns list of dicts with keys: (1) strategy, (2) event, (3) time, (4) value
    # for CURE metric, events can be 'MIN' (minimum value achieved was > 1) or 'CURE' (achieved value 1)
    # for PFS metric, events can be 'CURE', 'PROG' (went above 2e12), or 'NONE' (never went above progression threshold). 
    # gives the value achieved and the time that it happened.

    # find the number of drugs
    ndrugs = len(params[1])

    # generate all schedules
    allschedules = enumerate_schedules(ndrugs,nwindows)

    # create toreturn storage
    toreturn = []

    for schd in allschedules:
        # do the simulation
        #sol = ode_sim(params,schd,timepercycle=timepercycle)
        sol = Bunch(t=[],y=[])
        ndrugs = len(params[1])
        y0 = [1e12] + [1 for kk in range(ndrugs)]
        sol.t,sol.y = GDRSanalytical(y0,schd,timepercycle, *params)
        # re-order sol.y so the previous code below for evaluating the output of the ode sim solution still works for the analytical
        sol.y = [[yy[kk] for yy in sol.y] for kk in range(ndrugs+1)]
        
        # evaluate on the given metric and store the results
        if metric == 'CURE':
            # check for cure
            val = min(sol.y[0])
                
            time = sol.t[np.argmin(sol.y[0])]
            if val <= 1:
                event = 'CURE'
                val = 1
                time = next(tt for tt,xx in zip(sol.t,sol.y[0]) if xx <= 1)

            else:
                # otherwise record min
                event = 'MIN'

        elif metric =='PFS':
            # check for cure
            minval = min(sol.y[0])
                
            curetime = np.inf # need to keep the first event only
            if minval <= 1:
                cureevent = 'CURE'
                cureval = 1
                curetime = next(tt for tt,xx in zip(sol.t,sol.y[0]) if xx <= 1)
                
            # check for progression
            maxval = max(sol.y[0])
            progtime = np.inf # again, first event only
            if maxval >= 2e12:
                progevent = 'PROG'
                progval = 2e12
                progtime = next(tt for tt,xx in zip(sol.t,sol.y[0]) if xx >= 2e12)

            # check which came first
            if progtime < curetime:
                # we progressed first
                event = progevent
                time = progtime
                val = progval
                
            elif curetime < progtime:
                # we cured first
                event = cureevent
                time = curetime
                val = cureval
                
            else:
                # otherwise record none because neither happened and curetime == progtime == np.inf
                event = 'NONE'
                time = sol.t[-1]
                val = sol.y[0][-1]

        toreturn.append({'SCHEDULE':schd,'EVENT':event,'TIME':time,'VALUE':val})

    return toreturn

def print_stats(scheduleevals,metric,printout=True):
    # take the output of the previous function and print out the best on the metric and the amount of strategies that do each category
    # unpack the variables
    schedules = [dd['SCHEDULE'] for dd in scheduleevals]
    events = [dd['EVENT'] for dd in scheduleevals]
    times = [dd['TIME'] for dd in scheduleevals]
    values = [dd['VALUE'] for dd in scheduleevals]
    if metric == 'CURE':
        # print number cure vs number min
        ncure = events.count('CURE')
        nmin = events.count('MIN')
        if printout:
            print('{} total schedules tested. {} resulted in CURE. {} resulted in MIN.'.format(len(schedules),ncure,nmin))
            
            # print histogram of time to event
            fig = plt.figure(1,figsize=(8,6))
            axs = plt.gca()
            axs.hist(times,edgecolor='k')
            axs.set_ylabel('Count',fontsize=18)
            axs.set_xlabel('Time to Event',fontsize=18)            
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            #plt.legend(fontsize=14,bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            
            # print histogram of value
            fig = plt.figure(2,figsize=(8,6))
            axs = plt.gca()
            bins = np.logspace(np.log10(min(values)), np.log10(max(values)), 12)
            axs.hist(values,bins=bins,edgecolor='k')
            plt.xscale('log')
            axs.set_ylabel('Count',fontsize=18)
            axs.set_xlabel('Population at time of event',fontsize=18)            
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            axs.axvline(x=1e9,color='k',linestyle='--',label='Detection Threshold')
            plt.legend(fontsize=14,bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            
        # find the schedule with the lowest value, that's the optimum
        tomin = list(zip(values,times,[len(schd) - schd.count(set()) for schd in schedules]))
        minpair = min(tomin)
        minindx = tomin.index(minpair)
        if printout:
            print('CURE Optimal Schedule is {}, with event {}, with population {:.3E} at {}.'.format(schedules[minindx], events[minindx], values[minindx], times[minindx]))
            
        optimalschedule = schedules[minindx]

    elif metric == 'PFS':
        # print number cure vs number NONE vs number prog
        ncure = events.count('CURE')
        nnone = events.count('NONE')
        nprog = events.count('PROG')
        if printout:
            print('{} total schedules tested. {} resulted in CURE. {} resulted in PROG. {} resulted in NONE.'.format(len(schedules), ncure, nprog, nnone))
            # print histogram of time to event
            fig = plt.figure(1,figsize=(8,6))
            axs = plt.gca()
            axs.hist(times,edgecolor='k')
            axs.set_ylabel('Count',fontsize=18)
            axs.set_xlabel('Time to Event',fontsize=18)            
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            #plt.legend(fontsize=14,bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            # print histogram of value
            #fig = plt.figure(2,figsize=(8,6))
            #axs = plt.gca()
            #bins = np.logspace(np.log10(min(values)), np.log10(max(values)), 12)
            #axs.hist(values,bins=bins,edgecolor='k')
            #plt.xscale('log')
            #axs.set_ylabel('Count',fontsize=18)
            #axs.set_xlabel('Population at time of event',fontsize=18)            
            #plt.xticks(fontsize=14)
            #plt.yticks(fontsize=14)
            #axs.axvline(x=1e9,color='k',linestyle='--',label='Detection Threshold')
            #plt.legend(fontsize=14,bbox_to_anchor=(1.05, 1), loc='upper left')
            #plt.show()
            
        if ncure == 0:
            # find the schedule with the longest time with the lowest value, that's the optimum
            tomax = list(zip(times,[1/vv for vv in values],[schd.count(set()) for schd in schedules]))
            maxpair = max(tomax)
            maxindx = tomax.index(maxpair)
            if printout:
                print('PFS Optimal Schedule is {}, with event {}, with population {:.3E} at {}.'.format(schedules[maxindx], events[maxindx], values[maxindx], times[maxindx]))
                
            optimalschedule = schedules[maxindx]
        else:
            # we cured, so find the best cure
            tomin = list(zip(values,times,[len(schd) - schd.count(set()) for schd in schedules]))
            minpair = min(tomin)
            minindx = tomin.index(minpair)
            if printout:
                print('PFS Optimal Schedule is {}, with event {}, with population {:.3E} at {}.'.format(schedules[minindx], events[minindx], values[minindx], times[minindx]))
            
            optimalschedule = schedules[minindx]
        
    return optimalschedule

def full_sim(params,schedule,timepercycle=7,detectionthresh=False,legendon=True, curethresh=1, progthresh=2e12):
    # Simulates the ODE and plots it with the schedule shaded
    sol = ode_sim(params,schedule,timepercycle=timepercycle)
    # make sure the numerical solution stops when curethresh or progthresh is hit
    if min(sol.y[0]) <= curethresh:
        for indx in range(len(sol.y[0])):
            if sol.y[0][indx] <= curethresh:
                break

        for jj in range(len(sol.y)):
            if jj == 0:
                sol.y[jj] = [sol.y[jj][kk] if kk < indx else curethresh for kk in range(len(sol.y[jj]))]
            else:
                sol.y[jj] = [sol.y[jj][kk] if kk < indx else (sol.y[jj][indx] + sol.y[jj][indx-1]) / 2 for kk in range(len(sol.y[jj]))] # average this and previous value as an approximation since the threshold trip happened between the time steps

    if max(sol.y[0]) >= progthresh:
        for indx in range(len(sol.y[0])):
            if sol.y[0][indx] >= progthresh:
                break

        for jj in range(len(sol.y)):
            if jj == 0:
                sol.y[jj] = [sol.y[jj][kk] if kk < indx else progthresh for kk in range(len(sol.y[jj]))]
            else:
                sol.y[jj] = [sol.y[jj][kk] if kk < indx else (sol.y[jj][indx] + sol.y[jj][indx-1]) / 2 for kk in range(len(sol.y[jj]))]# average this and previous value as an approximation since the threshold trip happened between the time steps

    # plot
    fig = plt.figure(1,figsize=(8,6))
    axs = plt.gca()
    if detectionthresh:
        axs.axhline(y=1e9,color='k',linestyle='--',label='Detection Threshold')
        
    axs.set_ylabel('Population',fontsize=18)
    axs.set_xlabel('Time',fontsize=18)
    drugsused = set.union(*schedule)
    if len(drugsused) == 0:
        nlabels = 1
    else:
        nlabels = max(drugsused) + 1
        
    dlabels = ['Drug {}'.format(jj+1) for jj in range(nlabels)]
    dcolors = []
    seed(139499) # this seed is only for the plotting color
    for i in range(len(dlabels)):
        dcolors.append('#%06X' % randint(0, 0xFFFFFF))
    
    for kk in range(len(schedule)):
        tstart = kk * timepercycle
        tend = (kk + 1) * timepercycle
        
        if len(schedule[kk]) > 0:
            axs.axvspan(tstart,tend,color=dcolors[list(schedule[kk])[0]],alpha=0.2,label=dlabels[list(schedule[kk])[0]])
            dlabels[list(schedule[kk])[0]] = None

    # make sure sol.y stops at 
    axs.plot(sol.t,sol.y[0],linewidth=6,label='Numerical Solution')
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if legendon:
        plt.legend(fontsize=14,bbox_to_anchor=(1.05, 1), loc='upper left')
        
    axs.set_yscale('log')
    return fig

def full_sim_efficacy(params,schedule,timepercycle=7,detectionthresh=False,legendon=True):
    # Simulates the ODE and plots it with the schedule shaded
    sol = ode_sim(params,schedule,timepercycle=timepercycle)

    # plot
    fig = plt.figure(1,figsize=(8,6))
    axs = plt.gca()
    axs.set_ylabel('Efficacy',fontsize=18)
    axs.set_xlabel('Time',fontsize=18)
    drugsused = set.union(*schedule)
    if len(drugsused) == 0:
        nlabels = 1
    else:
        nlabels = max(drugsused) + 1
        
    dlabels = ['Drug {}'.format(jj+1) for jj in range(nlabels)]
    dcolors = []
    seed(139499) # this seed is only for the plotting color
    for i in range(len(dlabels)):
        dcolors.append('#%06X' % randint(0, 0xFFFFFF))
    
    for kk in range(len(schedule)):
        tstart = kk * timepercycle
        tend = (kk + 1) * timepercycle
        
        if len(schedule[kk]) > 0:
            axs.axvspan(tstart,tend,color=dcolors[list(schedule[kk])[0]],alpha=0.2,label=dlabels[list(schedule[kk])[0]])
            dlabels[list(schedule[kk])[0]] = None

    for kk in range(1,len(sol.y)):
        axs.plot(sol.t,sol.y[kk],linewidth=6,label='Numerical Solution, E_{}'.format(kk))
        
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if legendon:
        plt.legend(fontsize=14,bbox_to_anchor=(1.05, 1), loc='upper left')
        
    #axs.set_yscale('log')
    return fig

######################### Analytically solve GDRS #############################

def Tnodrug(gamma,T0,t):
    return T0 * np.exp(gamma * t)

def Eoff(s,E0,t):
    return E0 / (E0 + (1 - E0) * np.exp(-s * t))

def Eon(r,E0,t):
    return E0 * np.exp(-r * t)

def Tdrug(gamma,delta,r,T0,E0,t):
    return T0 * np.exp(gamma * t) * np.exp((delta * E0 / r) * (np.exp(-r * t) - 1))

def GDRSanalytical(y0,treatmentschedule,timepercycle, gamma, delta, r, s, curethresh=1, progthresh=2e12):
    # returns the GDRS solution evaluated at all treatment endpoints, as well as at all intermediate minima
    tvals = [0]
    y = [y0]
    for kk in range(len(treatmentschedule)):
        T0 = y[-1][0]
        E0 = y[-1][1:]
        endt = timepercycle * (kk + 1) # end of the treatment window
        # if cured or progressed, just set equal to threshold
        if T0 <= curethresh:
            tvals.append(endt)
            y.append(y[-1])
            continue

        if T0 >= progthresh:
            tvals.append(endt)
            y.append(y[-1])
            continue
            
        # check to see if a drug is on:
        intermediateflag = 0 # for later, if we have an intermediate minimum then we need to check if it is below curethresh
        if treatmentschedule[kk] == set():
            # drug is off, evaluate the no drug T and Eoff for each
            drugon = 'off'
            endT = Tnodrug(gamma,T0,timepercycle)
            endy = [[endT] + [Eoff(ss,ee,timepercycle) for ee,ss in zip(E0,s)]] # the extra brackets here are for convenience
            endt = [endt]
        else:
            # drug is on, pull it
            drugon = list(treatmentschedule[kk])[0]
            # evaluate all as drug off first, then manually evaluate the drug on E
            endT = Tdrug(gamma,delta[drugon],r[drugon],T0,E0[drugon],timepercycle)
            endy = [endT] + [Eoff(ss,ee,timepercycle) for ee,ss in zip(E0,s)]
            endy[drugon + 1] = Eon(r[drugon],E0[drugon],timepercycle)
            endy = [endy] # the extra brackets here are for convenience
            endt = [endt]

            # check for intermediate minimum. If yes, then add it too
            if gamma / delta[drugon] < E0[drugon] and gamma / delta[drugon] > E0[drugon] * np.exp(-r[drugon] * timepercycle):
                # if here, then there is an intermediate minimum
                midt = np.log(delta[drugon] * E0[drugon] / gamma) / r[drugon]
                # calculate things using same strategy as before, but now at the intermediate time
                midy = [Tdrug(gamma,delta[drugon],r[drugon],T0,E0[drugon],midt)] + [Eoff(ss,ee,midt) for ee,ss in zip(E0,s)]
                midy[drugon + 1] = Eon(r[drugon],E0[drugon],midt)
                # now pre-pend to endy so we can add it all easily
                endy = [midy] + endy
                endt = [endt[0] - timepercycle + midt] + endt
                intermediateflag = 1
                midT = midy[0]

        # check for cure and prog thresholds. If there, then add them and sort the endt,endy
        # also make sure that all y after hitting thresh are set to thresh
        if T0 < progthresh and endT > progthresh:
            # we progressed
            # solve for progression time
            if drugon == 'off':
                # simple, we are exp growing
                progt = np.log(progthresh / T0) / gamma
                endy += [[progthresh] + [Eoff(ss,ee,progt) for ee,ss in zip(E0,s)]]
            else:
                # difficult, drug is on
                progt = fsolve(lambda tt: Tdrug(gamma,delta[drugon],r[drugon],T0,E0[drugon],tt) - progthresh, timepercycle * kk)[0] 
                toaddtoendy = [progthresh] + [Eoff(ss,ee,progt) for ee,ss in zip(E0,s)]
                toaddtoendy[drugon + 1] = Eon(r[drugon],E0[drugon],progt)
                endy += [toaddtoendy]

            endt += [timepercycle * kk + progt]
            # sort them 
            doublets = sorted([(tt,yy) if tt < progt + timepercycle * kk else (tt,[progthresh] + yy[1:]) for tt,yy in zip(endt,endy)]) # added timepercycle * kk here because progt is relative to window start time
            endt = [dd[0] for dd in doublets]
            endy = [dd[1] for dd in doublets]

        if T0 > curethresh and (endT < curethresh or (intermediateflag == 1 and midT < curethresh)):
            # we cured
            # solve for cure time
            # drug must be on
            curet = fsolve(lambda tt: np.log(Tdrug(gamma,delta[drugon],r[drugon],T0,E0[drugon],tt) / curethresh), 0)[0]
            toaddtoendy = [curethresh] + [Eoff(ss,ee,curet) for ee,ss in zip(E0,s)]
            toaddtoendy[drugon + 1] = Eon(r[drugon],E0[drugon],curet)
            endy += [toaddtoendy]
            endt += [timepercycle * kk + curet]
            # sort them 
            doublets = sorted([(tt,yy) if tt < curet + timepercycle * kk else (tt,[curethresh] + toaddtoendy[1:]) for tt,yy in zip(endt,endy)]) # added timepercycle * kk here because curet is relative to window start time
            endt = [dd[0] for dd in doublets]
            endy = [dd[1] for dd in doublets]

        # if here, we've calculated it all. add the values
        y += endy # we can do this because of the extra [] around endy, convenient for adding multiple y
        tvals += endt

    return tvals,y

################################################### Simulated annealing code ################################

progthresh = 2e12
curethresh = 1

# function for finding the PFS time of a given schedule
def PFStime(tvals,yvals):
    progresstimes = [tt for yy,tt in zip(yvals,tvals) if yy[0] >= progthresh]
    if len(progresstimes) == 0:
        return np.inf

    return min(progresstimes)

# the energy functions. When calculating the min population and the PFS time, it's helpful to round because numerical precision can slightly affect the results. That is, if we are picking out one single optimal schedule, and that schedule just happens to differ by 1e-10 of a day/cell from all the others that it is actually tied with, then that schedule will win, in spite of tiebreakers.
# energy starts at about 30. Gets down to about 15.
def cureE(y,t):
    return np.log(1 + min([yy[0] for yy in y])) # need to log1p this so the energy decreases to zero on a reasonable scale as we reach the best y

def PFSE(y,t):
    if min([yy[0] for yy in y]) <= curethresh:
        # we cured, so set energy = 0
        return 0
        
    return 100 / np.log(1 + min([tt for yy,tt in zip(y,t) if yy[0] >= progthresh])) # need to do the log1p and inverse here so the energy decreases to zero on a reasonable scale

def PFSEinv(E):
    # invert the above PFS energy to find the pfs
    return np.exp(100 / E) - 1

def cureEinv(E):
    # same, invert the cure E to find the minimal tumor population values
    return np.exp(E) - 1

def regularize(y0,schedule,timepercycle,gamma,delta,r,s):
    # make sure the schedule runs until cure or progression.
    # if it reaches it early, cut the number of windows
    newt,newy = GDRSanalytical(y0,schedule,timepercycle, gamma, delta, r, s)
    if newy[-1][0] != curethresh and newy[-1][0] != progthresh:
        # we need to run longer to hit progression. calculate progtime
        progtime = np.log(progthresh / newy[-1][0]) / gamma # from the end
        numnewwindows = int(progtime // timepercycle) + 1 # add this many blank windows on the end
        schedule += [set() for kk in range(numnewwindows)]
        newt,newy = GDRSanalytical(y0,schedule,timepercycle, gamma, delta, r, s)

    # now check to see if the schedule is too long
    eventtime = min([tt for yy,tt in zip(newy,newt) if yy[0] >= progthresh or yy[0] <= curethresh])
    if eventtime // timepercycle < len(schedule) - 2:
        # it doesn't happen in the second to last window
        schedule = schedule[:int(eventtime // timepercycle)+1] + [set()]

    return schedule

def acceptanceprobability(Eold,Enew,oldschedule,newschedule,beta,metric,newtime,oldtime):
    pnaive = np.exp(-beta * (Enew - Eold)) # if Enew > Eold, then we have exp(negative) < 1 as we should
    if Enew == Eold:
        # we have a tie
        tiesettled = 0
        # if cure, and one of these cures, settle it by cure time
        # also do this if PFS and energy = 0, which means that cure time has been hit
        #if metric == 'CURE' and (newtime < np.inf or oldtime < np.inf):
        if newtime < np.inf or oldtime < np.inf:
            factor = (oldtime / newtime)
            pnaive = pnaive * factor # this factor is greater than 1 if newtime cures faster
            if factor != 1:
                tiesettled = 1
                
        # if still tied (includes PFS), settle it by minimizing drug used
        if not tiesettled:
            newnumberdrugsused = len(newschedule) - newschedule.count(set()) # number of treatment windows using drug is total - number not using
            oldnumberdrugsused = len(oldschedule) - oldschedule.count(set())
            pnaive = pnaive * (oldnumberdrugsused + 1) / (newnumberdrugsused + 1) # number of drugs used in the old schedule / number of drugs used in the new schedule > 1 when new schedule uses less (e.g. when it is better). Added one to numerator and denominator so there is no divide by zero error

    return pnaive

def betafxn(beta0,t,ncycles):
    # calculate beta given that we started at beta0 and now are at step t / ncycles
    if t < ncycles / 4:
        return beta0

    # otherwise, do some exponential growing
    growthfactor = 10
    return beta0 * np.exp(growthfactor * (t / ncycles - 1/4))

def calculatecuretime(t,y):
    tomin = [tt for tt,yy in zip(t,y) if yy[0] <= curethresh]
    if len(tomin) == 0:
        return np.inf
    
    return min(tomin)

# the full optimization function
def optimize(params,metric='CURE',timepercycle=7,strategystart=None,ncycles=10000,beta0=0.01,waitperiod=True,storeallschedules=False):
    # uses simulated annealing to find the optimal strategy
    # can start with any strategy (to run again)
    # nwindows updates dynamically
    # TO DO: need a param for annealing schedule, which should be sigmoidal in temp. Need an option to return the energy, or maybe plot on/off

    # unpack params
    gamma = params[0]
    delta = params[1]
    r = params[2]
    s = params[3]
    drugoptions = [set()] + [{kk} for kk in range(len(delta))]
    
    # if strategystart is not set, make it all empty
    if not strategystart:
        # default is all {}
        # find the appropriate number of windows so that we have one more window than progression
        progt = np.log(2) / gamma
        nwindowsstart = int(progt // timepercycle) + 1
        strategystart = [set() for kk in range(nwindowsstart)]

    optimalstrategy = [kk for kk in strategystart] # store the best one we found
    currentstrategy = [kk for kk in strategystart] # current one we are mutating from
    nwindows = len(strategystart)
    if metric == 'CURE':
        Efxn = cureE
    else:
        # PFS
        Efxn = PFSE

    # initialize values
    y0 = [1e12] + [1 for kk in range(len(delta))]
    currentt,currenty = GDRSanalytical(y0,currentstrategy,timepercycle, gamma, delta, r, s)
    currentE = Efxn(currenty,currentt)
    optimalE = Efxn(currenty,currentt)
    energies = [Efxn(currenty,currentt)] # store the energies, starting with the currenty
    optimalEtime = calculatecuretime(currentt,currenty)
    currentEtime = calculatecuretime(currentt,currenty)
    allschedulestorage = [currentstrategy]
    for zzz in range(ncycles):
        # calculate beta for this cycle
        beta = betafxn(beta0,zzz,ncycles)
        # make a random legal change to the schedule.
        # if no wait period, can modify any part of the schedule except the last window, which happens after progression and is kept clear
        eligiblemodinds = list(range(len(currentstrategy) - 1))
        # if wait period, then cannot modify if there is a drug used before or after
        if waitperiod:
            eligiblemodinds = [kk for kk in eligiblemodinds if (kk == 0 and currentstrategy[1] == set()) or (currentstrategy[kk-1] == set() and currentstrategy[kk+1] == set())]

        # pick a random member of this list to modify
        tomodifyind = eligiblemodinds[np.random.randint(0,len(eligiblemodinds))]
        eligibledrugoptions = [kk for kk in drugoptions if kk != currentstrategy[tomodifyind]]
        newdrug = eligibledrugoptions[np.random.randint(0,len(eligibledrugoptions))]
        testschedule = [ss for ss in currentstrategy] # same as current, except...
        replaceddrug = testschedule[tomodifyind]
        testschedule[tomodifyind] = newdrug # for this one
        # if newdrug is set() and waiting period is on, then check to see if we can insted push the current drug forward or backward by one cycle
        # we want to do this because we need e.g. [{0},set(),set(),...,set(),{0},set()] to be adjacent to both [{0},set(),set(),...,{0},set(),set()] and [{0},set(),set(),...,set(),set(),{0}] for better searching of the space
        if newdrug == set():
            if waitperiod:
                # if there is a wait period, then have to check 2 in front and 2 behind
                leftmoveopen = 0
                rightmoveopen = 0
                havetoappend = 0
                # first check for left move
                if tomodifyind == 1 or (tomodifyind > 1 and testschedule[tomodifyind-1] == set() and testschedule[tomodifyind-2] == set()):
                    leftmoveopen = 1

                if tomodifyind in [len(testschedule) - 1,len(testschedule) - 2] or (testschedule[tomodifyind+1] == set() and testschedule[tomodifyind+2] == set()):
                    rightmoveopen = 1
                    if tomodifyind == len(testschedule):
                        havetoappend = 1

            else:
                # if there is no wait period, only have to check directly in front and behind
                leftmoveopen = 0
                rightmoveopen = 0
                havetoappend = 0
                # first check for left move
                if tomodifyind > 0 and testschedule[tomodifyind-1] == set():
                    leftmoveopen = 1

                if tomodifyind == len(testschedule) - 1 or testschedule[tomodifyind+1] == set():
                    rightmoveopen = 1
                    if tomodifyind == len(testschedule):
                        havetoappend = 1

            # now draw for it, even chance of each move. same procedure for waitperiod or not
            moves = ['remove'] # remove has already been done, so if it is drawn then we do nothing
            if leftmoveopen:
                moves.append('left')

            if rightmoveopen:
                moves.append('right')

            dart = np.random.rand()
            moveindx = int(dart * len(moves) // 1)
            move = moves[moveindx]
            if move == 'left':
                testschedule[tomodifyind-1] = replaceddrug

            if move == 'right':
                if havetoappend == 1:
                    testschedule.append(replaceddrug)
                else:
                    testschedule[tomodifyind+1] = replaceddrug
                        
            # if here, check to see if tomodifyind + 2 is full. If not, then half the time we want to make current drug move back a week.
            # this should help get out of local mins by making, e.g. [{0},set()] adjacent to [set(),{0},set()] instead of having to go through [set()].
            #if tomodifyind + 2 >= len(testschedule) or (tomodifyind + 2 < len(testschedule) and testschedule[tomodifyind + 2] == set()):
            #    # we can index here and it's empty
            #    if np.random.rand() < 1/2:
            #        # put drug one week later than it was
            #        testschedule[tomodifyind + 1] = replaceddrug
                    
        # make sure testschedule runs to cure or progression
        testschedule = regularize(y0,testschedule,timepercycle,gamma,delta,r,s)
        # calculate new schedule's energy
        newt,newy = GDRSanalytical(y0,testschedule,timepercycle, gamma, delta, r, s)
        newEtime = calculatecuretime(newt,newy)
        # if we are searching for PFS and cure is found, then switch Efxn. This is to help the optimization find the first cure, instead of have a force pushing toward longer times and one pushing toward shorter
        #if metric == 'PFS' and newEtime < np.inf:
        #    Efxn = cureE
            
        newE = Efxn(newy,newt)
        # calculate the probability differential
        changeaccepted = 0
        paccept = acceptanceprobability(currentE,newE,currentstrategy,testschedule,beta,metric,newEtime,currentEtime)
        # throw a dart for acceptance of the change, accept if < np.exp(-beta * (Enew - Eold))
        if np.random.rand() < paccept:
            # accept the change
            changeaccepted = 1
            # update currentE, currentstrategy, currentt, currenty, currentEtime
            currentE = copy.deepcopy(newE)
            currentstrategy = copy.deepcopy(testschedule)
            currentt = copy.deepcopy(newt)
            currenty = copy.deepcopy(newy)
            currentEtime = copy.deepcopy(newEtime)

        # add to energies
        energies.append(currentE + 0)
        if storeallschedules:
            allschedulestorage.append(copy.deepcopy(currentstrategy))
        # if we accepted the change, see if optimal strategy changed
        if changeaccepted and acceptanceprobability(optimalE,currentE,optimalstrategy,currentstrategy,beta,metric,currentEtime,optimalEtime) > 1:
            # it is better
            optimalE = copy.deepcopy(currentE)
            optimalstrategy = copy.deepcopy(currentstrategy)
            optimalEtime = copy.deepcopy(currentEtime)

    if storeallschedules:
        return optimalstrategy,energies,allschedulestorage
        
    return optimalstrategy,energies

def standardize_optimum(strategy,metric,params,timepercycle,waitperiod=True,beta0=0.1):
    # this only works for a single drug
    standardstrategy = [ss for ss in strategy]
    # if only one drug window (or less), then standardized strategy is just the input
    if standardstrategy.count({0}) <= 1:
        return standardstrategy
        
    y0 = [1e12] + [1 for kk in params[1]]
    analyticalt,analyticaly = GDRSanalytical(y0,strategy,timepercycle, *params)
    if metric == 'CURE':
        Efxn = cureE
    else:
        # PFS
        Efxn = PFSE

    givenE = Efxn(analyticaly,analyticalt)
    if metric == 'CURE':
        # cut all that are after nadir
        yvals = [yy[0] for yy in analyticaly]
        miny = min(yvals)
        mint = analyticalt[yvals.index(miny)]   
        mintcycleindx = int(mint // timepercycle) # // makes it so we get the index in which it happens. if the min is achieved in the first window, this gives 0, the index in strategy that this happens
        # replace with set() if after mintcycleindx
        standardstrategy = standardstrategy[:mintcycleindx+1] + [set() for kk in range(mintcycleindx+1,len(standardstrategy))]

    # identify the period where drug is used
    lastdrugindx = len(standardstrategy) - 1
    while lastdrugindx >= 0:
        if standardstrategy[lastdrugindx] != set():
            break # if here, we found a drug
            
        lastdrugindx -= 1 # if here, increment to the next last element

    if lastdrugindx < 0:
        # optimal strategy was just all set() or only ha
        return strategy

    # find first drug index too
    firstdrugindx = 0
    while firstdrugindx < len(standardstrategy):
        if standardstrategy[firstdrugindx] != set():
            break # if here, we found a drug
            
        firstdrugindx += 1 # if here, increment to the next last element

    # no need to check for no drugs, we already filtered that out
    # if here, then we know what the first and last drug indices are. 
    # calculate the average number of windows per drug
    waitperiodaddition = 0
    if waitperiod:
        waitperiodaddition = 1 # need to add the buffer if we have a waiting period, just makes this a bit easier.

    ndrugsused = standardstrategy.count({0})
    # THIS IS WRONG: need to do average of the number of windows between drugs used. This should get rid of the need to do the waitpreiod thing.
    #windowsperdrug = (lastdrugindx - firstdrugindx) / (ndrugsused + waitperiodaddition) # note that we have specified to a single drug 
    drugwindows = np.array([kk for kk in range(len(standardstrategy)) if standardstrategy[kk] == {0}]) # we have specified to a single drug, but this should be easy to generalize? find avg space between all drugs and then deal with collisions
    windowsperdrug = np.mean(np.diff(drugwindows).tolist())
    wpdplus = int(np.ceil(windowsperdrug))
    wpdminus = int(np.floor(windowsperdrug))
    # create a schedule with each
    standardstrategyplus = standardstrategy[:firstdrugindx] + sum([[{0}] + [set() for kk in range(wpdplus - 1)] for jj in range(ndrugsused-1)],[]) + standardstrategy[lastdrugindx:] # made it ndrugsused - 1 because standardstrategy[lastindx:] contains the final drug use
    standardstrategyminus = standardstrategy[:firstdrugindx] + sum([[{0}] + [set() for kk in range(wpdminus - 1)] for jj in range(ndrugsused-1)],[]) + standardstrategy[lastdrugindx:]
    standardstrategyplus = regularize(y0,standardstrategyplus,timepercycle,*params)
    standardstrategyminus = regularize(y0,standardstrategyminus,timepercycle,*params)
    sspt,sspy = GDRSanalytical(y0,standardstrategyplus,timepercycle, *params)
    ssmt,ssmy = GDRSanalytical(y0,standardstrategyminus,timepercycle, *params)
    Eplus = Efxn(sspy,sspt)
    Eminus = Efxn(ssmy,ssmt)
    Etimeplus = calculatecuretime(sspt,sspy)
    Etimeminus = calculatecuretime(ssmt,ssmy)
    # compute the acceptance probability, minus is incumbent, plus is challenger
    pacceptpluscompminus = acceptanceprobability(Eminus,Eplus,standardstrategyminus,standardstrategyplus,beta0,metric,Etimeplus,Etimeminus)
    if pacceptpluscompminus < 1:
        # - is better
        standardstrategy = standardstrategyminus
    else:
        # + is better
        standardstrategy = standardstrategyplus

    # above, we've standardized cure by eliminating extra drug windows. We've also standardized E-type and MTD by regularizing the intervals in between
    # final thing is to attempt the T-type AT strategy using the information provided. This way, if T-type is the optimum, then we are trying it out. 
    # in reality, we should add these three standard strategies as "initial checks" before doing the SA. That way, if one is optimal, it is guaranteed to be found...
    
    return standardstrategy

def get_final_optimum(strategy,metric,params,timepercycle,waitperiod=True,beta0=0.01,reporting=False):
    # find the optimal strategy from these options: Simulated annealing output, standardized version of SA output, continuous MTD, and T-type AT
    # this only works for a single drug

    y0 = [1e12] + [1 for kk in params[1]] # starting values
    # strategy is the SA optimum
    SAanalyticalt,SAanalyticaly = GDRSanalytical(y0,strategy,timepercycle, *params) # calculate what happens with this schedule
    
    # Find the standardized version
    standardizedstrategy = standardize_optimum(strategy,metric,params,timepercycle,waitperiod=waitperiod)
    standardizedanalyticalt,standardizedanalyticaly = GDRSanalytical(y0,standardizedstrategy,timepercycle, *params) # calculate what happens with this schedule
    
    # if metric is cure, then we need to peel off all doses after the minimum
    if metric == 'CURE':
        # run the strategy an additional number of times to make sure we are actually capturing the minimum...
        standardizedstrategy = sum([standardizedstrategy for kk in range(5)],[])
        standardizedanalyticalt,standardizedanalyticaly = GDRSanalytical(y0,standardizedstrategy,timepercycle, *params) # calculate what happens with this schedule
        mint = standardizedanalyticalt[standardizedanalyticaly.index(min(standardizedanalyticaly))]
        cycleindxofmin = int(mint // timepercycle)
        standardizedstrategy = standardizedstrategy[:cycleindxofmin + 1] + [set() for kk in range(len(standardizedstrategy) - cycleindxofmin - 1)]
        standardizedstrategy = regularize(y0,standardizedstrategy,timepercycle,*params) # regularize the schedule when we are done so it plots nicely
        standardizedanalyticalt,standardizedanalyticaly = GDRSanalytical(y0,standardizedstrategy,timepercycle, *params) # calculate what happens with this schedule
        # if efficacy is still high, wait a couple periods then try again
        if standardizedanalyticaly[-1][1] > 0.5:
            # find out how many times to repeat
            standardizedstrategy = standardizedstrategy[:len(standardizedstrategy) - standardizedstrategy[::-1].index({0})]
            standardizedanalyticalt,standardizedanalyticaly = GDRSanalytical(y0,standardizedstrategy,timepercycle, *params) # calculate what happens with this schedule
            logkills = np.log10(y0[0]) - np.log10(standardizedanalyticaly[-1][0])
            repetitionsneeded = int(np.log10(y0[0]) / logkills) + 10 # add some padding
            standardizedstrategy = standardizedstrategy + [set(),set(),set()]
            standardizedstrategy = sum([standardizedstrategy for kk in range(repetitionsneeded)],[])
            standardizedstrategy = regularize(y0,standardizedstrategy,timepercycle,*params) # regularize the schedule when we are done so it plots nicely
            standardizedanalyticalt,standardizedanalyticaly = GDRSanalytical(y0,standardizedstrategy,timepercycle, *params) # calculate what happens with this schedule

    # find the continuous MTD schedule
    startingnumberofwindows = 10
    if waitperiod:
        MTDstrategy = sum([[{0},set()] for kk in range(startingnumberofwindows)],[])
    else:
        MTDstrategy = [{0} for kk in range(startingnumberofwindows)]
        
    MTDanalyticalt,MTDanalyticaly = GDRSanalytical(y0,MTDstrategy,timepercycle, *params) # calculate what happens with this schedule
    while MTDanalyticaly[-1][0] < progthresh and MTDanalyticaly[-1][0] > curethresh:
        # if we haven't progressed / cured, add windows
        if waitperiod:
            MTDstrategy = MTDstrategy + [{0},set()]
        else:
            MTDstrategy = MTDstrategy + [{0}]

        MTDanalyticalt,MTDanalyticaly = GDRSanalytical(y0,MTDstrategy,timepercycle, *params) # calculate what happens with this schedule

    # if metric is cure, then we need to peel off all doses after the minimum
    if metric == 'CURE':
        mint = MTDanalyticalt[MTDanalyticaly.index(min(MTDanalyticaly))]
        cycleindxofmin = int(mint // timepercycle)
        MTDstrategy = MTDstrategy[:cycleindxofmin + 1] + [set() for kk in range(len(MTDstrategy) - cycleindxofmin - 1)]
        
    MTDstrategy = regularize(y0,MTDstrategy,timepercycle,*params) # regularize the schedule when we are done so it plots nicely
    MTDanalyticalt,MTDanalyticaly = GDRSanalytical(y0,MTDstrategy,timepercycle, *params)
    # find the T-type AT schedule
    maxnumberofwindows = 1000 # stop the AT when we hit this so we don't loop forever
    ### CHANGED THE DEFINITION HERE. AT STARTS WITH A DOSE
    #ATstrategy = [set()] # try nothing first
    if waitperiod:
        ATstrategy = [{0},set()]
    else:
        ATstrategy = [{0}]
        
    ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params) # calculate what happens with this schedule
    #if ATanalyticaly[-1][0] >= progthresh:
        ## progressed, so need to dose first
        #if waitperiod:
        #    ATstrategy = [[{0},set()]]
        #else:
        #    ATstrategy = [{0}]
            
    while (ATanalyticaly[-1][0] < progthresh and ATanalyticaly[-1][0] > curethresh) and len(ATstrategy) < maxnumberofwindows:
        # check for next window progression
        ATstrategy = ATstrategy + [set()]
        ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params) # calculate what happens with this schedule
        if ATanalyticaly[-1][0] >= progthresh:
            # progressed, so need to dose first
            if waitperiod:
                ATstrategy = ATstrategy[:-1] + [{0},set()]
            else:
                ATstrategy = ATstrategy[:-1] + [{0}]

            # calculate again with the new schedule
            ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params)

    ATstrategy = regularize(y0,ATstrategy,timepercycle,*params) # regularize the schedule when we are done so it plots nicely
    ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params)
    # compare them all
    if metric == 'CURE':
        Efxn = cureE
    else:
        # PFS
        Efxn = PFSE

    # calculate E and t vals for the comparison
    SAE = Efxn(SAanalyticaly,SAanalyticalt)
    standardizedE = Efxn(standardizedanalyticaly,standardizedanalyticalt)
    MTDE = Efxn(MTDanalyticaly,MTDanalyticalt)
    ATE = Efxn(ATanalyticaly,ATanalyticalt)
    SAEtime = calculatecuretime(SAanalyticalt,SAanalyticaly)
    standardizedEtime = calculatecuretime(standardizedanalyticalt,standardizedanalyticaly)
    MTDEtime = calculatecuretime(MTDanalyticalt,MTDanalyticaly)
    ATEtime = calculatecuretime(ATanalyticalt,ATanalyticaly)
    # first compare standardized SA (challenger) to SA (incumbent)
    pacceptstandardizedcomptoSA = acceptanceprobability(SAE,standardizedE,strategy,standardizedstrategy,beta0,metric,standardizedEtime,SAEtime)
    if pacceptstandardizedcomptoSA < 1:
        winner1 = strategy # SA better than standardized
        winner1E = SAE
        winner1Etime = SAEtime
    else:
        winner1 = standardizedstrategy # standardized better than SA
        winner1E = standardizedE
        winner1Etime = standardizedEtime

    # no matter who won, it's e type
    reporter1 = 'E'
        
    # Now MTD (incumbent) to AT (challenger)
    pacceptATcomptoMTD = acceptanceprobability(MTDE,ATE,MTDstrategy,ATstrategy,beta0,metric,ATEtime,MTDEtime)
    if pacceptATcomptoMTD < 1:
        winner2 = MTDstrategy # MTD better than AT
        winner2E = MTDE
        winner2Etime = MTDEtime
        reporter2 = 'MTD'
    else:
        winner2 = ATstrategy # AT better than MTD
        winner2E = ATE
        winner2Etime = ATEtime
        reporter2 = 'T'
        
    # now compare winner1 (incumbent) vs winner2 (challenger)
    pacceptwinners = acceptanceprobability(winner1E,winner2E,winner1,winner2,beta0,metric,winner2Etime,winner1Etime)
    if pacceptwinners < 1:
        optimum = winner1 # winner1 better than winner2
        reporter = reporter1
    else:
        optimum = winner2 # winner2 better than winner1
        reporter = reporter2

    if reporting:
        return (optimum,reporter)
        
    return optimum

# let's abstract out getting the AT strategy so we can use it elsewhere
def getATstrategy(metric,params,timepercycle,waitperiod=True):
    y0 = [1e12] + [1 for kk in params[1]] # starting values
    maxnumberofwindows = 1000 # stop the AT when we hit this so we don't loop forever
    ### CHANGED THE DEFINITION HERE. AT STARTS WITH A DOSE
    #ATstrategy = [set()] # try nothing first
    if waitperiod:
        ATstrategy = [{0},set()]
    else:
        ATstrategy = [{0}]
        
    ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params) # calculate what happens with this schedule
    #if ATanalyticaly[-1][0] >= progthresh:
        ## progressed, so need to dose first
        #if waitperiod:
        #    ATstrategy = [[{0},set()]]
        #else:
        #    ATstrategy = [{0}]
            
    while (ATanalyticaly[-1][0] < progthresh and ATanalyticaly[-1][0] > curethresh) and len(ATstrategy) < maxnumberofwindows:
        # check for next window progression
        ATstrategy = ATstrategy + [set()]
        ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params) # calculate what happens with this schedule
        if ATanalyticaly[-1][0] >= progthresh:
            # progressed, so need to dose first
            if waitperiod:
                ATstrategy = ATstrategy[:-1] + [{0},set()]
            else:
                ATstrategy = ATstrategy[:-1] + [{0}]

            # calculate again with the new schedule
            ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params)

    ATstrategy = regularize(y0,ATstrategy,timepercycle,*params) # regularize the schedule when we are done so it plots nicely
    return ATstrategy

# 2 drug adaptive therapy where only one drug is used adaptively
def getATstrategy2(metric,params,timepercycle,waitperiod=True,drugorder=[0,1]):
    # drug order is adaptive, constant
    y0 = [1e12] + [1 for kk in params[1]] # starting values
    maxnumberofwindows = 1000 # stop the AT when we hit this so we don't loop forever
    ### CHANGED THE DEFINITION HERE. AT STARTS WITH A DOSE
    #ATstrategy = [set()] # try nothing first
    if waitperiod:
        ATstrategy = [{drugorder[0]},set()]
        defaultmove = [{drugorder[1]},set()]
    else:
        ATstrategy = [{drugorder[0]}]
        defaultmove = [{drugorder[1]}]
        
    ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params) # calculate what happens with this schedule
    #if ATanalyticaly[-1][0] >= progthresh:
        ## progressed, so need to dose first
        #if waitperiod:
        #    ATstrategy = [[{0},set()]]
        #else:
        #    ATstrategy = [{0}]
            
    while (ATanalyticaly[-1][0] < progthresh and ATanalyticaly[-1][0] > curethresh) and len(ATstrategy) < maxnumberofwindows:
        # check for next window progression
        ATstrategy = ATstrategy + defaultmove
        ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params) # calculate what happens with this schedule
        if ATanalyticaly[-1][0] >= progthresh:
            # progressed, so need to dose first
            if waitperiod:
                ATstrategy = ATstrategy[:-2] + [{drugorder[0]},set()]
            else:
                ATstrategy = ATstrategy[:-1] + [{drugorder[0]}]

            # calculate again with the new schedule
            ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params)

    ATstrategy = regularize(y0,ATstrategy,timepercycle,*params) # regularize the schedule when we are done so it plots nicely
    return ATstrategy

# now write a 2 drug strategy tester. Test sequential (default in situations of ties), alternating, second-strike 
def get_final_optimum2(strategy,metric,params,timepercycle,waitperiod=True,beta0=0.01,reporting=False):
    # tests a few schedules for the 2 drug case
    y0 = [1e12] + [1 for kk in params[1]] # starting values
    # strategy is the SA optimum
    SAanalyticalt,SAanalyticaly = GDRSanalytical(y0,strategy,timepercycle, *params) # calculate what happens with this schedule

    # create the alternating schedules
    altstrategy01 = make_alternating(params,timepercycle,metric,waitperiod=waitperiod,drugorder=[0,1])
    altanalyticalt01,altanalyticaly01 = GDRSanalytical(y0,altstrategy01,timepercycle, *params)
    altstrategy10 = make_alternating(params,timepercycle,metric,waitperiod=waitperiod,drugorder=[1,0])
    altanalyticalt10,altanalyticaly10 = GDRSanalytical(y0,altstrategy10,timepercycle, *params)

    # make the sequential strategies
    seqstrategy01 = make_sequential(params,timepercycle,waitperiod=waitperiod,drugorder=[0,1])
    seqanalyticalt01,seqanalyticaly01 = GDRSanalytical(y0,seqstrategy01,timepercycle, *params)
    seqstrategy10 = make_sequential(params,timepercycle,waitperiod=waitperiod,drugorder=[0,1])
    seqanalyticalt10,seqanalyticaly10 = GDRSanalytical(y0,seqstrategy10,timepercycle, *params)

    # create the second-strike schedules
    ssstrategy01 = make_second_strike(params,timepercycle,waitperiod=waitperiod,drugorder=[0,1])
    ssanalyticalt01,ssanalyticaly01 = GDRSanalytical(y0,ssstrategy01,timepercycle, *params)
    ssstrategy10 = make_second_strike(params,timepercycle,waitperiod=waitperiod,drugorder=[1,0])
    ssanalyticalt10,ssanalyticaly10 = GDRSanalytical(y0,ssstrategy10,timepercycle, *params)
    
    # compete them
    if metric == 'CURE':
        Efxn = cureE
    else:
        # PFS
        Efxn = PFSE

    # calculate E and t vals for the comparison
    SAE = Efxn(SAanalyticaly,SAanalyticalt)
    SAEtime = calculatecuretime(SAanalyticalt,SAanalyticaly)
    altE01 = Efxn(altanalyticaly01,altanalyticalt01)
    seqE01 = Efxn(seqanalyticaly01,seqanalyticalt01)
    ssE01 = Efxn(ssanalyticaly01,ssanalyticalt01)
    altE10 = Efxn(altanalyticaly10,altanalyticalt10)
    seqE10 = Efxn(seqanalyticaly10,seqanalyticalt10)
    ssE10 = Efxn(ssanalyticaly10,ssanalyticalt10)
    altEtime01 = calculatecuretime(altanalyticalt01,altanalyticaly01)
    ssEtime01 = calculatecuretime(ssanalyticalt01,ssanalyticaly01)
    seqEtime01 = calculatecuretime(seqanalyticalt01,seqanalyticaly01)
    altEtime10 = calculatecuretime(altanalyticalt10,altanalyticaly10)
    ssEtime10 = calculatecuretime(ssanalyticalt10,ssanalyticaly10)
    seqEtime10 = calculatecuretime(seqanalyticalt10,seqanalyticaly10)
    # first, compare 01 to 10 for each case and take the winner as that strategy's representative
    # alternating, 10 (challenger) to 01 (incumbent)
    pacceptalt0110 = acceptanceprobability(altE01,altE10,altstrategy01,altstrategy10,beta0,metric,altEtime10,altEtime01)
    if pacceptalt0110 < 1: # 01 better than 10
        altstrategy = altstrategy01
        altE = altE01
        altEtime = altEtime01
    else: # 10 better than 01
        altstrategy = altstrategy10
        altE = altE10
        altEtime = altEtime10

    # seq, 10 (challenger) to 01 (incumbent)
    pacceptseq0110 = acceptanceprobability(seqE01,seqE10,seqstrategy01,seqstrategy10,beta0,metric,seqEtime10,seqEtime01)
    if pacceptseq0110 < 1: # 01 better than 10
        seqstrategy = seqstrategy01
        seqE = seqE01
        seqEtime = seqEtime01
    else: # 10 better than 01
        seqstrategy = seqstrategy10
        seqE = seqE10
        seqEtime = seqEtime10

    # ss, 10 (challenger) to 01 (incumbent)
    pacceptss0110 = acceptanceprobability(ssE01,ssE10,ssstrategy01,ssstrategy10,beta0,metric,ssEtime10,ssEtime01)
    if pacceptss0110 < 1: # 01 better than 10
        ssstrategy = ssstrategy01
        ssE = ssE01
        ssEtime = ssEtime01
    else: # 10 better than 01
        ssstrategy = ssstrategy10
        ssE = ssE10
        ssEtime = ssEtime10
        
    # Now, compare alt (challenger) to SA (incumbent)
    pacceptaltcomptoSA = acceptanceprobability(SAE,altE,strategy,altstrategy,beta0,metric,altEtime,SAEtime)
    if pacceptaltcomptoSA < 1:
        winner1 = strategy # SA better than alt
        winner1E = SAE
        winner1Etime = SAEtime
        reporter1 = 'SA'
        if SAanalyticaly[-1][0] == curethresh:
            # if we cure, keep track of that
            reporter1 = 'CURE'
            
    else:
        winner1 = altstrategy # alt better than SA
        winner1E = altE
        winner1Etime = altEtime
        reporter1 = 'ALT'
        
    # Now seq (incumbent) to ss (challenger)
    pacceptsscomptoseq = acceptanceprobability(seqE,ssE,seqstrategy,ssstrategy,beta0,metric,ssEtime,seqEtime)
    if pacceptsscomptoseq < 1:
        winner2 = seqstrategy # seq better than ss
        winner2E = seqE
        winner2Etime = seqEtime
        reporter2 = 'SEQ'
    else:
        winner2 = ssstrategy # ss better than seq
        winner2E = ssE
        winner2Etime = ssEtime
        reporter2 = 'SS'
        
    # now compare winner1 vs winner2 
    pacceptwinners = acceptanceprobability(winner1E,winner2E,winner1,winner2,beta0,metric,winner2Etime,winner1Etime)
    if pacceptwinners <= 1: 
        optimum = winner1 # winner1 better than winner2
        winner3E = winner1E
        winner3Etime = winner1Etime
        reporter = reporter1
        # this code can be used to investigate the PFS phase diagram
        #if params[3][0] == 0:
        #    print('Winner 1: {} with E = {}. Winner 2: {} with E = {}. 1 won.'.format(reporter1,winner1E,reporter2,winner2E))
    else:
        optimum = winner2 # winner2 better than winner1
        winner3E = winner2E
        winner3Etime = winner2Etime
        reporter = reporter2
        #if params[3][0] == 0:
        #    print('Winner 1: {} with E = {}. Winner 2: {} with E = {}. 2 won.'.format(reporter1,winner1E,reporter2,winner2E))

    if reporting: # return which one won
        return (optimum,reporter)

    return optimum

def make_sequential(params,timepercycle,waitperiod=True,drugorder=[0,1]):
    # create the sequential schedule
    y0 = [1e12] + [1 for kk in params[1]] # starting values
    basicunit = [{drugorder[0]}]
    if waitperiod:
        basicunit = [{drugorder[0]},set()]

    seqstrategy = [] + basicunit
    seqanalyticalt,seqanalyticaly = GDRSanalytical(y0,seqstrategy,timepercycle, *params)
    while seqanalyticaly[-1][0] > curethresh and seqanalyticaly[-1][0] < progthresh:
        seqstrategy += basicunit
        seqanalyticalt,seqanalyticaly = GDRSanalytical(y0,seqstrategy,timepercycle, *params)

    # if cured, then done
    if seqanalyticaly[-1][0] != curethresh:
        # progressed, so peel off last basic unit and go again with second drug
        basicunit[0] = {drugorder[1]}
        seqstrategy = seqstrategy[:-len(basicunit)] + basicunit
        seqanalyticalt,seqanalyticaly = GDRSanalytical(y0,seqstrategy,timepercycle, *params)
        while seqanalyticaly[-1][0] > curethresh and seqanalyticaly[-1][0] < progthresh:
            seqstrategy += basicunit
            seqanalyticalt,seqanalyticaly = GDRSanalytical(y0,seqstrategy,timepercycle, *params)

    seqstrategy = regularize(y0,seqstrategy,timepercycle,*params)
    return seqstrategy

def make_second_strike(params,timepercycle,waitperiod=True,drugorder=[0,1]):
    # create the second-strike schedule
    y0 = [1e12] + [1 for kk in params[1]] # starting values
    basicunit = [{drugorder[0]}]
    if waitperiod:
        basicunit = [{drugorder[0]},set()]

    ssstrategy = [] + basicunit
    ssanalyticalt,ssanalyticaly = GDRSanalytical(y0,ssstrategy,timepercycle, *params)
    oldmin = np.inf
    newmin = min([yy[0] for yy in ssanalyticaly])
    while newmin < oldmin and newmin > curethresh:
        oldmin = newmin + 0
        ssstrategy += basicunit
        ssanalyticalt,ssanalyticaly = GDRSanalytical(y0,ssstrategy,timepercycle, *params)
        newmin = min([yy[0] for yy in ssanalyticaly])

    # if cured, then done
    if ssanalyticaly[-1][0] != curethresh:
        # passed min, so peel off last basic unit and go again with second drug
        basicunit[0] = {drugorder[1]}
        ssstrategy = ssstrategy[:-len(basicunit)] + basicunit
        ssanalyticalt,ssanalyticaly = GDRSanalytical(y0,ssstrategy,timepercycle, *params)
        oldmin = np.inf
        newmin = min([yy[0] for yy in ssanalyticaly])
        while newmin < oldmin and newmin > curethresh:
            oldmin = newmin + 0
            ssstrategy += basicunit
            ssanalyticalt,ssanalyticaly = GDRSanalytical(y0,ssstrategy,timepercycle, *params)
            newmin = min([yy[0] for yy in ssanalyticaly])

        # remove any doses after the min
        mint = ssanalyticalt[ssanalyticaly.index(min(ssanalyticaly))]
        cycleindxofmin = int(mint // timepercycle)
        ssstrategy = ssstrategy[:cycleindxofmin + 1] + [set() for kk in range(len(ssstrategy) - cycleindxofmin - 1)]

    ssstrategy = regularize(y0,ssstrategy,timepercycle,*params)
    return ssstrategy

def make_alternating(params,timepercycle,metric,waitperiod=True,drugorder=[0,1]):
    # create the alternating schedule
    y0 = [1e12] + [1 for kk in params[1]] # starting values
    basicunit = [{drugorder[0]},{drugorder[1]}]
    if waitperiod:
        basicunit = [{drugorder[0]},set(),{drugorder[1]},set()]

    altstrategy = [] + basicunit
    altanalyticalt,altanalyticaly = GDRSanalytical(y0,altstrategy,timepercycle, *params)
    while altanalyticaly[-1][0] > curethresh and altanalyticaly[-1][0] < progthresh:
        altstrategy += basicunit
        altanalyticalt,altanalyticaly = GDRSanalytical(y0,altstrategy,timepercycle, *params)

    # if metric is cure, then we need to peel off all doses after the minimum
    if metric == 'CURE':
        mint = altanalyticalt[altanalyticaly.index(min(altanalyticaly))]
        cycleindxofmin = int(mint // timepercycle)
        altstrategy = altstrategy[:cycleindxofmin + 1] + [set() for kk in range(len(altstrategy) - cycleindxofmin - 1)]
        
    altstrategy = regularize(y0,altstrategy,timepercycle,*params)
    altanalyticalt,altanalyticaly = GDRSanalytical(y0,altstrategy,timepercycle, *params)
    return altstrategy

    # auxiliary function for the plotting

def lintransform(x,x1,x2,y1,y2):
    # mapping x1 -> y1, x2 -> y2 linearly
    # return y <- x
    pct = (x - x1) / (x2 - x1)
    return pct * (y2 - y1) + y1

# do the test for if the given parameters should be adaptive therapy or not
# Find PFS time when doing AT for 2 doses. Compare to PFS time when doing CTS for 2 windows
# if AT gives longer PFS time, then AT is favored. Otherwise, CTS.
def isATpredictedoptimum(params,timepercycle,waitperiod=True):
    y0 = [1e12] + [1 for kk in params[1]] # starting values
    maxnumberofwindows = 1000 # stop the AT when we hit this so we don't loop forever
    ### CHANGED THE DEFINITION HERE. AT STARTS WITH A DOSE
    #ATstrategy = [set()] # try nothing first
    if waitperiod:
        ATstrategy = [{0},set()]
    else:
        ATstrategy = [{0}]
        
    ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params) # calculate what happens with this schedule
    ndoses = 1
    #if ATanalyticaly[-1][0] >= progthresh:
        ## progressed, so need to dose first
        #if waitperiod:
        #    ATstrategy = [[{0},set()]]
        #else:
        #    ATstrategy = [{0}]
            
    while (ATanalyticaly[-1][0] < progthresh and ATanalyticaly[-1][0] > curethresh) and len(ATstrategy) < maxnumberofwindows and ndoses < 2:
        # check for next window progression
        ATstrategy = ATstrategy + [set()]
        ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params) # calculate what happens with this schedule
        if ATanalyticaly[-1][0] >= progthresh:
            # progressed, so need to dose first
            if waitperiod:
                ATstrategy = ATstrategy[:-1] + [{0},set()]
            else:
                ATstrategy = ATstrategy[:-1] + [{0}]

            # calculate again with the new schedule
            ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params)
            ndoses += 1

    ATstrategy = regularize(y0,ATstrategy,timepercycle,*params) # regularize the schedule when we are done so it plots nicely
    ATanalyticalt,ATanalyticaly = GDRSanalytical(y0,ATstrategy,timepercycle, *params)
    # find AT PFS time
    ATPFStime = PFStime(ATanalyticalt,ATanalyticaly)
    # move last drug to the second or third place (depending on wait time) to make CTS schedule
    if waitperiod:
        CTSstrategy = copy.deepcopy(ATstrategy)
        CTSstrategy[2] = {0}
    else:
        CTSstrategy = copy.deepcopy(ATstrategy)
        CTSstrategy[1] = {0}

    indx = len(CTSstrategy) - 1
    while indx > 0:
        if CTSstrategy[indx] == {0}:
            CTSstrategy[indx] = set()
            break

        indx -= 1
        
    # find CTS PFS time
    CTSanalyticalt,CTSanalyticaly = GDRSanalytical(y0,CTSstrategy,timepercycle, *params)
    CTSPFStime = PFStime(CTSanalyticalt,CTSanalyticaly)
    # compare. If CTS >= AT, then CTS should be optimal. Otherwise, AT should be.
    if np.round(ATPFStime,4) > np.round(CTSPFStime,4) and ATstrategy != CTSstrategy:
        return 1
    else:
        return 0
        
    return 0

# code for plotting the phase diagram
def plotphasediagram(storage,phasespace,storagereport,timepercycle,rvals,svals,rvalsX,svalsY,gamma,delta,metric):
    deltar = (rvals[1] - rvals[0]) / gamma * 0.9
    deltas = (svals[1] - svals[0]) / gamma * 0.9
    fig = plt.figure(777,figsize=(56,32))
    ax = plt.gca()
    for kk in range(len(storage)):
        # loop through the schedule and plot drug on/off
        nwindows = len(storage[kk])
        squarewidth = deltar / nwindows
        squareheight = deltas 
        relative_bottom_left_xs = [-deltar / 2 + kk * squarewidth for kk in range(nwindows)]
        relative_bottom_left_ys = [-deltas / 2 for kk in range(nwindows)]
        rcenter = rvalsX[kk]
        scenter = svalsY[kk]
        for jj in range(len(storage[kk])):
            if storage[kk][jj] == {0}:
                # plot filled rectangle
                square = patches.Rectangle((rcenter + relative_bottom_left_xs[jj], scenter + relative_bottom_left_ys[jj]), squarewidth, squareheight, linewidth=1, facecolor='cornflowerblue')#, edgecolor='k',)
                ax.add_patch(square)
        
        # plot the solution
        sol = ode_sim(phasespace[kk],storage[kk],timepercycle=timepercycle)
        # scale t values to the x values of the rectangle. sol.t[0] = rcenter - deltar / 2, sol.t[-1] = rcenter + deltar / 2
        # scale log10(y) values to the y values of the rectangle. y = 1 is scenter - deltas / 2, y = 2e12 is scenter + deltas / 2
        # all y values less than 1 get set to 1, all > 2e12 get set to 2e12
        tX = [lintransform(tt,sol.t[0],sol.t[-1],rcenter - deltar / 2,rcenter + deltar / 2) for tt in sol.t]
        censoredy = [curethresh if yy < curethresh else progthresh if yy > progthresh else yy for yy in sol.y[0]]
        yY = [lintransform(np.log10(yy),np.log10(curethresh),np.log10(progthresh),scenter - deltas / 2,scenter + deltas / 2) for yy in censoredy]
        ax.plot(tX,yY,'k-',linewidth=4)
        # plot big rectangle around the whole thing
        square = patches.Rectangle((rcenter + relative_bottom_left_xs[0], scenter + relative_bottom_left_ys[0]), deltar, deltas, linewidth=3, edgecolor='k', facecolor='none')
        ax.add_patch(square)

    #plt.plot(rvalsX,svalsY,'k.')
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=70)
    plt.yticks(fontsize=70)
    plt.xlabel('r / gamma',fontsize=75)
    plt.ylabel('s / gamma',fontsize=75)
    if metric == 'PFS':
        plt.title('Maximize PFS',fontsize=84)
    elif metric == 'CURE':
        plt.title('Minimize tumor burden',fontsize=84)
    
    plt.show()

    # do the same thing, but plot colored rectangles instead depending on what the overall optimum was reported as to show the phase boundaries
    fig = plt.figure(888,figsize=(15,8))
    ax = plt.gca()
    for kk in range(len(storage)):
        # loop through the schedule and plot drug on/off
        nwindows = len(storage[kk])
        squarewidth = deltar / nwindows
        squareheight = deltas 
        relative_bottom_left_xs = [-deltar / 2 + kk * squarewidth for kk in range(nwindows)]
        relative_bottom_left_ys = [-deltas / 2 for kk in range(nwindows)]
        rcenter = rvalsX[kk]
        scenter = svalsY[kk]
        # determine the color to plot based on which schedule is returned as optimal
        if storagereport[kk] == 'E':
            squarecolor = '#6699CC'

        if storagereport[kk] == 'T':
            squarecolor = '#7fc97f'

        if storagereport[kk] == 'MTD':
            squarecolor = '#fdc086'
        
        # plot big rectangle around the whole thing
        square = patches.Rectangle((rcenter + relative_bottom_left_xs[0], scenter + relative_bottom_left_ys[0]), deltar, deltas, linewidth=3, edgecolor='k', facecolor=squarecolor)
        ax.add_patch(square)

    #plt.plot(rvalsX,svalsY,'k.')
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel('r / gamma',fontsize=22)
    plt.ylabel('s / gamma',fontsize=22)
    if metric == 'PFS':
        plt.title('Maximize PFS',fontsize=27)
    elif metric == 'CURE':
        plt.title('Minimize tumor burden',fontsize=27)

    # Next, find when our analytical metrics predict that the schedule should be T type or E type
    plt.figure(349827349,figsize=(56,32))
    ax = plt.gca()
    for kk in range(len(storage)):
        # loop through the schedule and plot drug on/off
        nwindows = len(storage[kk])
        squarewidth = deltar / nwindows
        squareheight = deltas 
        relative_bottom_left_xs = [-deltar / 2 + kk * squarewidth for kk in range(nwindows)]
        relative_bottom_left_ys = [-deltas / 2 for kk in range(nwindows)]
        rcenter = rvalsX[kk]
        scenter = svalsY[kk]
        
        # plot the function that needs to be zero for E type to be possible
        rr = rcenter * gamma
        ss = scenter * gamma
        dt1 = timepercycle # dose time
        fxntoeval = lambda x: gamma * dt1 + delta * x / rr * (np.exp(-rr * dt1) - 1) + gamma / ss * np.log((1 - x * np.exp(-rr * dt1)) / ((1 - x) * np.exp(-rr * dt1)))
        xevalvals = np.array([xx/100 for xx in range(1,100)])
        yevalvals = np.array([fxntoeval(xx) for xx in xevalvals])
        # scale t values to the x values of the rectangle. sol.t[0] = rcenter - deltar / 2, sol.t[-1] = rcenter + deltar / 2
        # scale log10(y) values to the y values of the rectangle. y = 1 is scenter - deltas / 2, y = 2e12 is scenter + deltas / 2
        # all y values less than 1 get set to 1, all > 2e12 get set to 2e12
        tX = [lintransform(tt,xevalvals[0],xevalvals[-1],rcenter - deltar / 2,rcenter + deltar / 2) for tt in xevalvals]
        censoredy = [0 if yy < 0 else 1 if yy > 1 else yy for yy in yevalvals]
        yY = [lintransform(yy,0,1,scenter - deltas / 2,scenter + deltas / 2) for yy in censoredy]
        if 0 in censoredy:
            # E type is possible, make it blue
            ax.plot(tX,yY,'-',linewidth=4,color='#6699CC')
        else:
            # not possible, make it red
            ax.plot(tX,yY,'-',linewidth=4,color='#fdc086')
            
        # plot big rectangle around the whole thing. Color of edge is the categorization our method for parsing into T, E, cts gave
        if storagereport[kk] == 'E':
            edgecolor = '#6699CC'

        if storagereport[kk] == 'T':
            edgecolor = '#7fc97f'

        if storagereport[kk] == 'MTD':
            edgecolor = '#fdc086'
            
        square = patches.Rectangle((rcenter + relative_bottom_left_xs[0], scenter + relative_bottom_left_ys[0]), deltar, deltas, linewidth=3, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(square)
        predictedAT = isATpredictedoptimum(phasespace[kk],timepercycle,waitperiod=True)
        if predictedAT:
            # AT is possible, mark it with green
            plt.plot(rr/gamma,ss/gamma,'*',markersize=30,color='#7fc97f')

    #plt.plot(rvalsX,svalsY,'k.')
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=70)
    plt.yticks(fontsize=70)
    plt.xlabel('r / gamma',fontsize=75)
    plt.ylabel('s / gamma',fontsize=75)
    if metric == 'PFS':
        plt.title('Maximize PFS',fontsize=84)
    elif metric == 'CURE':
        plt.title('Minimize tumor burden',fontsize=84)
    
    plt.show()
    return

# code for plotting the phase diagram
def plotphasediagram2(storage,phasespace,storagereport,timepercycle,rvals,svals,rvalsX,svalsY,gamma,metric):
    deltar = (rvals[1] - rvals[0]) / gamma * 0.9
    deltas = (svals[1] - svals[0]) / gamma * 0.9
    fig = plt.figure(777,figsize=(56,32))
    ax = plt.gca()
    for kk in range(len(storage)):
        # loop through the schedule and plot drug on/off
        nwindows = len(storage[kk])
        squarewidth = deltar / nwindows
        squareheight = deltas 
        relative_bottom_left_xs = [-deltar / 2 + kk * squarewidth for kk in range(nwindows)]
        relative_bottom_left_ys = [-deltas / 2 for kk in range(nwindows)]
        rcenter = rvalsX[kk]
        scenter = svalsY[kk]
        for jj in range(len(storage[kk])):
            if storage[kk][jj] == {0}:
                # plot filled rectangle
                square = patches.Rectangle((rcenter + relative_bottom_left_xs[jj], scenter + relative_bottom_left_ys[jj]), squarewidth, squareheight, linewidth=1, facecolor='cornflowerblue')#, edgecolor='k',)
                ax.add_patch(square)

            if storage[kk][jj] == {1}:
                # plot filled rectangle
                square = patches.Rectangle((rcenter + relative_bottom_left_xs[jj], scenter + relative_bottom_left_ys[jj]), squarewidth, squareheight, linewidth=1, facecolor='springgreen')#, edgecolor='k',)
                ax.add_patch(square)
        
        # plot the solution
        sol = ode_sim(phasespace[kk],storage[kk],timepercycle=timepercycle)
        # scale t values to the x values of the rectangle. sol.t[0] = rcenter - deltar / 2, sol.t[-1] = rcenter + deltar / 2
        # scale log10(y) values to the y values of the rectangle. y = 1 is scenter - deltas / 2, y = 2e12 is scenter + deltas / 2
        # all y values less than 1 get set to 1, all > 2e12 get set to 2e12
        tX = [lintransform(tt,sol.t[0],sol.t[-1],rcenter - deltar / 2,rcenter + deltar / 2) for tt in sol.t]
        censoredy = [curethresh if yy < curethresh else progthresh if yy > progthresh else yy for yy in sol.y[0]]
        yY = [lintransform(np.log10(yy),np.log10(curethresh),np.log10(progthresh),scenter - deltas / 2,scenter + deltas / 2) for yy in censoredy]
        ax.plot(tX,yY,'k-',linewidth=4)
        # plot big rectangle around the whole thing
        square = patches.Rectangle((rcenter + relative_bottom_left_xs[0], scenter + relative_bottom_left_ys[0]), deltar, deltas, linewidth=3, edgecolor='k', facecolor='none')
        ax.add_patch(square)

    #plt.plot(rvalsX,svalsY,'k.')
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=70)
    plt.yticks(fontsize=70)
    plt.xlabel('r / gamma',fontsize=75)
    plt.ylabel('s / gamma',fontsize=75)
    if metric == 'PFS':
        plt.title('Maximize PFS',fontsize=84)
    elif metric == 'CURE':
        plt.title('Minimize tumor burden',fontsize=84)
    
    plt.show()

    # do the same thing, but plot colored rectangles instead depending on what the overall optimum was reported as to show the phase boundaries
    fig = plt.figure(888,figsize=(15,8))
    ax = plt.gca()
    for kk in range(len(storage)):
        # loop through the schedule and plot drug on/off
        nwindows = len(storage[kk])
        squarewidth = deltar / nwindows
        squareheight = deltas 
        relative_bottom_left_xs = [-deltar / 2 + kk * squarewidth for kk in range(nwindows)]
        relative_bottom_left_ys = [-deltas / 2 for kk in range(nwindows)]
        rcenter = rvalsX[kk]
        scenter = svalsY[kk]
        # determine the color to plot based on which schedule is returned as optimal
        if storagereport[kk] == 'CURE':
            squarecolor = '#6699CC'

        if storagereport[kk] == 'SA':
            squarecolor = '#8AE378' # slight perturbation of the green
        
        if storagereport[kk] == 'ALT':
            squarecolor = '#beaed4'

        if storagereport[kk] == 'SEQ':
            squarecolor = '#fdc086'

        if storagereport[kk] == 'SS':
            squarecolor = '#ffff99'
        
        # plot big rectangle around the whole thing
        square = patches.Rectangle((rcenter + relative_bottom_left_xs[0], scenter + relative_bottom_left_ys[0]), deltar, deltas, linewidth=3, edgecolor='k', facecolor=squarecolor)
        ax.add_patch(square)

    #plt.plot(rvalsX,svalsY,'k.')
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel('r / gamma',fontsize=22)
    plt.ylabel('s / gamma',fontsize=22)
    if metric == 'PFS':
        plt.title('Maximize PFS',fontsize=27)
    elif metric == 'CURE':
        plt.title('Minimize tumor burden',fontsize=27)

    return

def infinitesimalphaseboundaryplot(storage,phasespace,storagereport,timepercycle,rvals,svals,rvalsX,svalsY,gamma,delta,metric,smin,smax):
    # Next, find when our analytical metrics predict that the schedule should be T type or E type
    deltar = (rvals[1] - rvals[0]) / gamma * 0.9
    deltas = (svals[1] - svals[0]) / gamma * 0.9
    plt.figure(349827349,figsize=(56,32))
    ax = plt.gca()
    for kk in range(len(storage)):
        # loop through the schedule and plot drug on/off
        nwindows = len(storage[kk])
        squarewidth = deltar / nwindows
        squareheight = deltas 
        relative_bottom_left_xs = [-deltar / 2 + kk * squarewidth for kk in range(nwindows)]
        relative_bottom_left_ys = [-deltas / 2 for kk in range(nwindows)]
        rcenter = rvalsX[kk]
        scenter = svalsY[kk]
        
        # plot the function that needs to be zero for E type to be possible
        rr = rcenter * gamma
        ss = scenter * gamma
        dt1 = timepercycle # dose time
        fxntoeval = lambda x: gamma * dt1 + delta * x / rr * (np.exp(-rr * dt1) - 1) + gamma / ss * np.log((1 - x * np.exp(-rr * dt1)) / ((1 - x) * np.exp(-rr * dt1)))
        xevalvals = np.array([xx/100 for xx in range(1,100)])
        yevalvals = np.array([fxntoeval(xx) for xx in xevalvals])
        # scale t values to the x values of the rectangle. sol.t[0] = rcenter - deltar / 2, sol.t[-1] = rcenter + deltar / 2
        # scale log10(y) values to the y values of the rectangle. y = 1 is scenter - deltas / 2, y = 2e12 is scenter + deltas / 2
        # all y values less than 1 get set to 1, all > 2e12 get set to 2e12
        tX = [lintransform(tt,xevalvals[0],xevalvals[-1],rcenter - deltar / 2,rcenter + deltar / 2) for tt in xevalvals]
        censoredy = [0 if yy < 0 else 1 if yy > 1 else yy for yy in yevalvals]
        yY = [lintransform(yy,0,1,scenter - deltas / 2,scenter + deltas / 2) for yy in censoredy]
        if 0 in censoredy:
            # E type is possible, make it blue
            ax.plot(tX,yY,'-',linewidth=4,color='#6699CC')
        else:
            # not possible, make it red
            ax.plot(tX,yY,'-',linewidth=4,color='#fdc086')
            
        # plot big rectangle around the whole thing. Color of edge is the categorization our method for parsing into T, E, cts gave
        if storagereport[kk] == 'E':
            edgecolor = '#6699CC'

        if storagereport[kk] == 'T':
            edgecolor = '#7fc97f'

        if storagereport[kk] == 'MTD':
            edgecolor = '#fdc086'
            
        square = patches.Rectangle((rcenter + relative_bottom_left_xs[0], scenter + relative_bottom_left_ys[0]), deltar, deltas, linewidth=3, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(square)
        predictedAT = isATpredictedoptimum(phasespace[kk],timepercycle,waitperiod=True)
        if predictedAT:
            # AT is possible, mark it with green
            plt.plot(rr/gamma,ss/gamma,'*',markersize=30,color='#7fc97f')

    # plot the infinitesimal boundary
    svalsboundary = [(smin + kk/100 *(smax - smin))for kk in range(101)]
    rvalsboundary = [(ssval * (delta * (1 + gamma / delta)**2 / (4 * gamma) - 1)) / gamma  for ssval in svalsboundary]
    svalsboundary = [ssval / gamma for ssval in svalsboundary]
    # get the bounds for x and y so we can keep the plot the same
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    plt.plot(rvalsboundary,svalsboundary,'k-',linewidth=8)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    # other plot formatting
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.xlabel('r / gamma',fontsize=60)
    plt.ylabel('s / gamma',fontsize=60)
    if metric == 'PFS':
        plt.title('Maximize PFS',fontsize=84)
    elif metric == 'CURE':
        plt.title('Minimize tumor burden',fontsize=84)
    
    plt.show()
    return

############## Alternative methods for the two drug PFS phase diagram #############

# now write a 2 drug strategy tester. Test sequential (default in situations of ties), alternating, second-strike 
def get_final_optimum2_alternate(strategy,metric,params,timepercycle,waitperiod=True,beta0=0.01,reporting=False):
    # tests a few schedules for the 2 drug case
    y0 = [1e12] + [1 for kk in params[1]] # starting values
    # strategy is the SA optimum
    SAanalyticalt,SAanalyticaly = GDRSanalytical(y0,strategy,timepercycle, *params) # calculate what happens with this schedule

    # create the alternating schedules
    altstrategy01 = make_alternating(params,timepercycle,metric,waitperiod=waitperiod,drugorder=[0,1])
    altanalyticalt01,altanalyticaly01 = GDRSanalytical(y0,altstrategy01,timepercycle, *params)
    altstrategy10 = make_alternating(params,timepercycle,metric,waitperiod=waitperiod,drugorder=[1,0])
    altanalyticalt10,altanalyticaly10 = GDRSanalytical(y0,altstrategy10,timepercycle, *params)

    # make the sequential strategies
    seqstrategy01 = make_sequential(params,timepercycle,waitperiod=waitperiod,drugorder=[0,1])
    seqanalyticalt01,seqanalyticaly01 = GDRSanalytical(y0,seqstrategy01,timepercycle, *params)
    seqstrategy10 = make_sequential(params,timepercycle,waitperiod=waitperiod,drugorder=[0,1])
    seqanalyticalt10,seqanalyticaly10 = GDRSanalytical(y0,seqstrategy10,timepercycle, *params)

    # create the second-strike schedules
    ssstrategy01 = make_second_strike(params,timepercycle,waitperiod=waitperiod,drugorder=[0,1])
    ssanalyticalt01,ssanalyticaly01 = GDRSanalytical(y0,ssstrategy01,timepercycle, *params)
    ssstrategy10 = make_second_strike(params,timepercycle,waitperiod=waitperiod,drugorder=[1,0])
    ssanalyticalt10,ssanalyticaly10 = GDRSanalytical(y0,ssstrategy10,timepercycle, *params)

    # create one-drug-like adaptive therapy
    ATstrategy01 = getATstrategy2(metric,params,timepercycle,waitperiod=waitperiod,drugorder=[0,1])
    ATanalyticalt01,ATanalyticaly01 = GDRSanalytical(y0,ATstrategy01,timepercycle, *params)
    ATstrategy10 = getATstrategy2(metric,params,timepercycle,waitperiod=waitperiod,drugorder=[1,0])
    ATanalyticalt10,ATanalyticaly10 = GDRSanalytical(y0,ATstrategy10,timepercycle, *params)
    
    # compete them
    if metric == 'CURE':
        Efxn = cureE
    else:
        # PFS
        Efxn = PFSE

    # calculate E and t vals for the comparison
    SAE = Efxn(SAanalyticaly,SAanalyticalt)
    SAEtime = calculatecuretime(SAanalyticalt,SAanalyticaly)
    altE01 = Efxn(altanalyticaly01,altanalyticalt01)
    seqE01 = Efxn(seqanalyticaly01,seqanalyticalt01)
    ssE01 = Efxn(ssanalyticaly01,ssanalyticalt01)
    ATE01 = Efxn(ATanalyticaly01,ATanalyticalt01)
    altE10 = Efxn(altanalyticaly10,altanalyticalt10)
    seqE10 = Efxn(seqanalyticaly10,seqanalyticalt10)
    ssE10 = Efxn(ssanalyticaly10,ssanalyticalt10)
    ATE10 = Efxn(ATanalyticaly10,ATanalyticalt10)
    altEtime01 = calculatecuretime(altanalyticalt01,altanalyticaly01)
    ssEtime01 = calculatecuretime(ssanalyticalt01,ssanalyticaly01)
    seqEtime01 = calculatecuretime(seqanalyticalt01,seqanalyticaly01)
    ATEtime01 = calculatecuretime(ATanalyticalt01,ATanalyticaly01)
    altEtime10 = calculatecuretime(altanalyticalt10,altanalyticaly10)
    ssEtime10 = calculatecuretime(ssanalyticalt10,ssanalyticaly10)
    seqEtime10 = calculatecuretime(seqanalyticalt10,seqanalyticaly10)
    ATEtime10 = calculatecuretime(ATanalyticalt10,ATanalyticaly10)
    # first, compare 01 to 10 for each case and take the winner as that strategy's representative
    # alternating, 10 (challenger) to 01 (incumbent)
    pacceptalt0110 = acceptanceprobability(altE01,altE10,altstrategy01,altstrategy10,beta0,metric,altEtime10,altEtime01)
    if pacceptalt0110 < 1: # 01 better than 10
        altstrategy = altstrategy01
        altE = altE01
        altEtime = altEtime01
    else: # 10 better than 01
        altstrategy = altstrategy10
        altE = altE10
        altEtime = altEtime10

    # seq, 10 (challenger) to 01 (incumbent)
    pacceptseq0110 = acceptanceprobability(seqE01,seqE10,seqstrategy01,seqstrategy10,beta0,metric,seqEtime10,seqEtime01)
    if pacceptseq0110 < 1: # 01 better than 10
        seqstrategy = seqstrategy01
        seqE = seqE01
        seqEtime = seqEtime01
    else: # 10 better than 01
        seqstrategy = seqstrategy10
        seqE = seqE10
        seqEtime = seqEtime10

    # ss, 10 (challenger) to 01 (incumbent)
    pacceptss0110 = acceptanceprobability(ssE01,ssE10,ssstrategy01,ssstrategy10,beta0,metric,ssEtime10,ssEtime01)
    if pacceptss0110 < 1: # 01 better than 10
        ssstrategy = ssstrategy01
        ssE = ssE01
        ssEtime = ssEtime01
    else: # 10 better than 01
        ssstrategy = ssstrategy10
        ssE = ssE10
        ssEtime = ssEtime10

    # AT, 10 (challenger) to 01 (incumbent)
    pacceptAT0110 = acceptanceprobability(ATE01,ATE10,ATstrategy01,ATstrategy10,beta0,metric,ATEtime10,ATEtime01)
    if pacceptAT0110 < 1: # 01 better than 10
        ATstrategy = ATstrategy01
        ATE = ATE01
        ATEtime = ATEtime01
    else: # 10 better than 01
        ATstrategy = ATstrategy10
        ATE = ATE10
        ATEtime = ATEtime10
        
    # Now, compare alt (challenger) to SA (incumbent)
    pacceptaltcomptoSA = acceptanceprobability(SAE,altE,strategy,altstrategy,beta0,metric,altEtime,SAEtime)
    if pacceptaltcomptoSA < 1:
        winner1 = strategy # SA better than alt
        winner1E = SAE
        winner1Etime = SAEtime
        reporter1 = 'SA'
    else:
        winner1 = altstrategy # alt better than SA
        winner1E = altE
        winner1Etime = altEtime
        reporter1 = 'ALT'
        
    # Now seq (incumbent) to ss (challenger)
    pacceptsscomptoseq = acceptanceprobability(seqE,ssE,seqstrategy,ssstrategy,beta0,metric,ssEtime,seqEtime)
    if pacceptsscomptoseq < 1:
        winner2 = seqstrategy # seq better than ss
        winner2E = seqE
        winner2Etime = seqEtime
        reporter2 = 'SEQ'
    else:
        winner2 = ssstrategy # ss better than seq
        winner2E = ssE
        winner2Etime = ssEtime
        reporter2 = 'SS'
        
    # now compare winner1 vs winner2 
    pacceptwinners = acceptanceprobability(winner1E,winner2E,winner1,winner2,beta0,metric,winner2Etime,winner1Etime)
    if pacceptwinners <= 1: 
        winner3 = winner1 # winner1 better than winner2
        winner3E = winner1E
        winner3Etime = winner1Etime
        reporter3 = reporter1
    else:
        winner3 = winner2 # winner2 better than winner1
        winner3E = winner2E
        winner3Etime = winner2Etime
        reporter3 = reporter2

    # now compare winner3 vs AT
    pacceptwinners2 = acceptanceprobability(winner3E,ATE,winner3,ATstrategy,beta0,metric,ATEtime,winner3Etime)
    if pacceptwinners2 <= 1: 
        optimum = winner3 # winner3 better than AT
        reporter = reporter3
    else:
        optimum = ATstrategy # AT better than winner3
        reporter = 'AT'

    if reporting: # return which one won
        return (optimum,reporter)

    return optimum

# code for plotting the phase diagram of the above and putting the drug 1 phase boundary on
def plotphasediagram2_alternate(storage,phasespace,storagereport,timepercycle,rvals,svals,rvalsX,svalsY,gamma,metric,delta,smin,smax):
    deltar = (rvals[1] - rvals[0]) / gamma * 0.9
    deltas = (svals[1] - svals[0]) / gamma * 0.9
    fig = plt.figure(777,figsize=(56,32))
    ax = plt.gca()
    for kk in range(len(storage)):
        # loop through the schedule and plot drug on/off
        nwindows = len(storage[kk])
        squarewidth = deltar / nwindows
        squareheight = deltas 
        relative_bottom_left_xs = [-deltar / 2 + kk * squarewidth for kk in range(nwindows)]
        relative_bottom_left_ys = [-deltas / 2 for kk in range(nwindows)]
        rcenter = rvalsX[kk]
        scenter = svalsY[kk]
        for jj in range(len(storage[kk])):
            if storage[kk][jj] == {0}:
                # plot filled rectangle
                square = patches.Rectangle((rcenter + relative_bottom_left_xs[jj], scenter + relative_bottom_left_ys[jj]), squarewidth, squareheight, linewidth=1, facecolor='cornflowerblue')#, edgecolor='k',)
                ax.add_patch(square)

            if storage[kk][jj] == {1}:
                # plot filled rectangle
                square = patches.Rectangle((rcenter + relative_bottom_left_xs[jj], scenter + relative_bottom_left_ys[jj]), squarewidth, squareheight, linewidth=1, facecolor='springgreen')#, edgecolor='k',)
                ax.add_patch(square)
        
        # plot the solution
        sol = ode_sim(phasespace[kk],storage[kk],timepercycle=timepercycle)
        # scale t values to the x values of the rectangle. sol.t[0] = rcenter - deltar / 2, sol.t[-1] = rcenter + deltar / 2
        # scale log10(y) values to the y values of the rectangle. y = 1 is scenter - deltas / 2, y = 2e12 is scenter + deltas / 2
        # all y values less than 1 get set to 1, all > 2e12 get set to 2e12
        tX = [lintransform(tt,sol.t[0],sol.t[-1],rcenter - deltar / 2,rcenter + deltar / 2) for tt in sol.t]
        censoredy = [curethresh if yy < curethresh else progthresh if yy > progthresh else yy for yy in sol.y[0]]
        yY = [lintransform(np.log10(yy),np.log10(curethresh),np.log10(progthresh),scenter - deltas / 2,scenter + deltas / 2) for yy in censoredy]
        ax.plot(tX,yY,'k-',linewidth=4)
        # plot big rectangle around the whole thing
        square = patches.Rectangle((rcenter + relative_bottom_left_xs[0], scenter + relative_bottom_left_ys[0]), deltar, deltas, linewidth=3, edgecolor='k', facecolor='none')
        ax.add_patch(square)

    #plt.plot(rvalsX,svalsY,'k.')
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.xlabel('r / gamma',fontsize=60)
    plt.ylabel('s / gamma',fontsize=60)
    if metric == 'PFS':
        plt.title('Maximize PFS',fontsize=84)
    elif metric == 'CURE':
        plt.title('Minimize tumor burden',fontsize=84)
    
    plt.show()

    # do the same thing, but plot colored rectangles instead depending on what the overall optimum was reported as to show the phase boundaries
    fig = plt.figure(888,figsize=(14,8))
    ax = plt.gca()
    for kk in range(len(storage)):
        # loop through the schedule and plot drug on/off
        nwindows = len(storage[kk])
        squarewidth = deltar / nwindows
        squareheight = deltas 
        relative_bottom_left_xs = [-deltar / 2 + kk * squarewidth for kk in range(nwindows)]
        relative_bottom_left_ys = [-deltas / 2 for kk in range(nwindows)]
        rcenter = rvalsX[kk]
        scenter = svalsY[kk]
        # determine the color to plot based on which schedule is returned as optimal
        if storagereport[kk] == 'SA':
            squarecolor = '#6699CC'

        if storagereport[kk] == 'AT':
            squarecolor = '#7fc97f'
        
        if storagereport[kk] == 'ALT':
            squarecolor = '#beaed4'

        if storagereport[kk] == 'SEQ':
            squarecolor = '#fdc086'

        if storagereport[kk] == 'SS':
            squarecolor = '#ffff99'
        
        # plot big rectangle around the whole thing
        square = patches.Rectangle((rcenter + relative_bottom_left_xs[0], scenter + relative_bottom_left_ys[0]), deltar, deltas, linewidth=3, edgecolor='k', facecolor=squarecolor)
        ax.add_patch(square)

    #plt.plot(rvalsX,svalsY,'k.')
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('r / gamma',fontsize=14)
    plt.ylabel('s / gamma',fontsize=14)
    if metric == 'PFS':
        plt.title('Maximize PFS',fontsize=18)
    elif metric == 'CURE':
        plt.title('Minimize tumor burden',fontsize=18)

    # plot the infinitesimal boundary
    svalsboundary = [(smin + kk/100 *(smax - smin))for kk in range(101)]
    rvalsboundary = [(ssval * (delta * (1 + gamma / delta)**2 / (4 * gamma) - 1)) / gamma  for ssval in svalsboundary]
    svalsboundary = [ssval / gamma for ssval in svalsboundary]
    # get the bounds for x and y so we can keep the plot the same
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    plt.plot(rvalsboundary,svalsboundary,'k-',linewidth=8)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    return