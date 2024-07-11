import ast
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import StringVar
import matplotlib
import openpyxl
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import style
from matplotlib.figure import Figure
import pandas as pd
import scipy as sp
from scipy.fft import fft, fftfreq
import colorednoise as cn
from scipy.stats import expon,uniform,logistic
import pyabf
from _datetime import datetime


style.use("ggplot")
px = 1 / plt.rcParams['figure.dpi']
count = 0
COLOUR = 'DarkBlue'
set_ax_to_time = False

default_values = {'numpulses' : 25,
                   'mincurr' :-3.5,
                    'maxcurr' : -10,
                    'minpwd' :1,
                    'maxpwd' : 100,
                    'sparse_fac' : 0.1,
                    'vshift':35,
                    'sampfreq' : 250000,
                    'colornoiseorders' : [1,2],
                    'nsigma' : 5,
                    'numconcs' : 6,
                    'maxamposin' : 1,
                    'nstepwins' : 5,
                    'driftmaxmag' : 3,
                    'maxnsteps' : 3,
                    'dist' : 'exp',
                    'multilevel' :'Mix',
                    'mixratio' : 0.30,
                    'resistance':8.5e9,
                    'capacitance':1e-15,
                    'sequence' : ['A','T','G','C'],
                    'currents' : [-10,-8,-7,-5],
                    'pulsewidths' : [10,12,3,7],
                    'shuffle': 'True'}
def dummy():
    return

def Toggletimencounts():
    global set_ax_to_time
    # Determine is on or off
    if set_ax_to_time == True:
        #timecounttoggle.config(image=on)
        set_ax_to_time = False
    else:
        #timecounttoggle.config(image=off)
        set_ax_to_time = True
cdt = datetime.now()


def multilevelsignalgen(sequence,currents,pulsewidths):
    seqsignal = []
    for i,bp in enumerate(sequence):
        ht = currents[i]
        pw = pulsewidths[i]
        bppulse = [ht]*pw
        seqsignal.extend(bppulse)
    return seqsignal,sequence,currents,pulsewidths

def gen_events(numpulses,mincurr,maxcurr,minpwd,maxpwd,sparse_fac,dist, multilevel = 'False',mixratio = 0.31,sequence = ['A','T','G','C'],currents = [-8,-5,-3,-10],pulsewidths = [10,10,10,10],shuffle = 'true'):
    amps = []
    for i in range(0,numpulses):
        amp = random.uniform(mincurr,maxcurr)
        amps.append(amp)
    pwdlist = []
    pulses = []
    lvlnmlist,lvlcurrlist,lvlwidthlist = [],[],[]
    for i in range(0,numpulses):
        if multilevel == 'false':
            pwd = random.randint(minpwd,maxpwd)
            pulse = [amps[i]]*pwd
            pulses.append(pulse)
        elif multilevel == 'true':
            if shuffle == 'true':
                seqinds = np.arange(0,len(sequence))
                random.shuffle(seqinds)
                sequence = [sequence[i] for i in seqinds]
                currents = [currents[i] for i in seqinds]
                pulsewidths = [pulsewidths[i] for i in seqinds]
                pulse,lvlnms,lvlcurrs,lvlwdths = multilevelsignalgen(sequence=sequence,currents = currents,pulsewidths = pulsewidths)
            elif shuffle == 'false':
                pulse, lvlnms, lvlcurrs, lvlwdths = multilevelsignalgen(sequence=sequence, currents=currents,pulsewidths=pulsewidths)
            pulses.append(pulse)
            lvlnmlist.append(lvlnms)
            lvlcurrlist.append(lvlcurrs)
            lvlwidthlist.append(lvlwdths)
        else:
            break
    if multilevel == 'mix':
        num_multilvlpulses = int(numpulses*mixratio)
        num_singlelvlpulses = numpulses - num_multilvlpulses
        for i in range(0,num_multilvlpulses):
            if shuffle == 'true':
                seqinds = np.arange(0,len(sequence))
                random.shuffle(seqinds)
                sequence = [sequence[i] for i in seqinds]
                currents = [currents[i] for i in seqinds]
                pulsewidths = [pulsewidths[i] for i in seqinds]
                pulse, lvlnms, lvlcurrs, lvlwdths = multilevelsignalgen(sequence=sequence, currents=currents,pulsewidths=pulsewidths)
            elif shuffle == 'false':
                pulse, lvlnms, lvlcurrs, lvlwdths = multilevelsignalgen(sequence=sequence, currents=currents,pulsewidths=pulsewidths)
                #print(pulse)
            pulses.append(pulse)
            lvlnmlist.append(lvlnms)
            lvlcurrlist.append(lvlcurrs)
            lvlwidthlist.append(lvlwdths)
        for i in range(0,num_singlelvlpulses):
            pwd = random.randint(int(minpwd), int(maxpwd))
            pulse = [amps[i]] * pwd
            pulses.append(pulse)
            lvlnmlist.append(['Single Level'])
            lvlwidthlist.append([pwd])

        inds = np.arange(0,len(pulses))
        #print(list(inds))
        random.shuffle(inds)
        #print(list(inds))
        #print('lvlnmlist', lvlnmlist)
        #print('lvlwidthlist', lvlwidthlist)
        pulses = [pulses[i] for i in inds]
        lvlnmlist = [lvlnmlist[i] for i in inds]
        lvlwidthlist = [lvlwidthlist[i] for i in inds]
        #print('pulses',pulses)
    #print('lvlnmlist',lvlnmlist)
    #print('lvlwidthlist',lvlwidthlist)



    concpulses = np.concatenate(pulses)
    locop = len(concpulses)
    appendch = []
    baseuni = []
    startinds = []
    stopinds = []
    los = 0
    #plt.plot(concpulses)
    #plt.show()
    #print(dist)
    if dist == 'exp':
        expdist = expon.rvs(scale=locop / sparse_fac, size=numpulses).astype(int)
        evdist = expdist
        #print(list(evdist))
    elif dist == 'uni':
        unidist = uniform.rvs(loc=0, scale=locop / (0.5 * sparse_fac), size=numpulses).astype(int)
        evdist = unidist
    elif dist == 'log':
        logdist = logistic.rvs(loc=0, scale=locop / (3*1.33 * sparse_fac), size=numpulses).astype(int)
        minlg = min(logdist)
        logdist = abs(minlg)+logdist
        evdist = logdist
    #plt.plot(np.sort(evdist))
    #plt.show()
    eventlocs = np.array([])
    if multilevel == 'false':
        for i in range(0,len(pulses)):
            zers = np.zeros(evdist[i])
            baseuni.append(zers)
            appendch.append(baseuni[i])
            appendch.append(pulses[i])
            startinds.append(los+len(zers))
            stopinds.append(los+len(zers)+len(pulses[i]))
            signal = np.concatenate(appendch)
            los=len(signal)
        eventlocs = [startinds,stopinds]
        #print(eventlocs)
    elif multilevel == 'true':
        lolvlsps = []    #list of level startpoints in a single event
        for i in range(0,len(pulses)-1):
            zers = np.zeros(evdist[i])
            baseuni.append(zers)
            appendch.append(baseuni[i])
            appendch.append(pulses[i])
            startinds.append(los + len(zers))
            stopinds.append(los + len(zers) + len(pulses[i]))
            lvlwdth = lvlwidthlist[i]
            lvlnms = lvlnmlist[i]
            lvlsp = los + len(zers)
            lvlsps = []
            for j,base in enumerate(lvlnms):
                lvlsp = lvlsp + lvlwdth[j]
                lvlsps.append(lvlsp)
            lolvlsps.append(lvlsps)
            lolvlspsar = np.array(lolvlsps) - 1
            signal = np.concatenate(appendch)
            los = len(signal)
        currlvls = list(signal[lolvlspsar])
        #print('levl starts',len(lolvlsps), lolvlsps)
        #print('levl currs',len(currlvls),list(currlvls))
        #print('level widths',lvlwidthlist)
        eventlocs = [startinds,lolvlspsar,lvlwidthlist,stopinds]
    elif multilevel == 'mix':
        mostlevels = np.max([len(nm) for nm in lvlnmlist]) #max number of levels in an event
        #print('mostlevels',mostlevels)
        lolvlsps,mxlvlwdthlist = [],[]  # list of level startpoints in a single event
        for i,lvlnms in enumerate(lvlnmlist):
            #print(len(lvlnms),lvlnms)
            if len(lvlnms) != mostlevels:
                zers = np.zeros(evdist[i])
                baseuni.append(zers)
                appendch.append(baseuni[i])
                appendch.append(pulses[i])
                startinds.append(los + len(zers))
                stopinds.append(los + len(zers) + len(pulses[i]))
                signal = np.concatenate(appendch)
                los = len(signal)
                lvlsp = np.array([los]*mostlevels)
                lvlsp = np.pad(lvlsp,(0,abs(len(lvlsp)-mostlevels)),'constant',constant_values=0)
                #print(lvlsp)
                lvlwdths = lvlwidthlist[i]
                lvlwdths = np.pad(lvlwdths,(0,abs(len(lvlwdths)-mostlevels)),'constant',constant_values=0)
                lolvlsps.append(lvlsp)
                mxlvlwdthlist.append(lvlwdths)
                #print(lolvlsps)
                lolvlspsar = np.array(lolvlsps) - 1
                signal = np.concatenate(appendch)
                los = len(signal)
                currlvls = list(signal[lolvlspsar])
                #print('levl currs', len(currlvls), list(currlvls))
            elif len(lvlnms) == mostlevels:
                #print('len(lvlnms)',len(lvlnms))
                zers = np.zeros(evdist[i])
                baseuni.append(zers)
                appendch.append(baseuni[i])
                appendch.append(pulses[i])
                startinds.append(los + len(zers))
                stopinds.append(los + len(zers) + len(pulses[i]))
                lvlwdth = lvlwidthlist[i]
                lvlnms = lvlnmlist[i]
                lvlsp = los + len(zers)
                mxlvlwdthlist.append(lvlwdth)
                lvlsps = []
                for j, base in enumerate(lvlnms):
                    lvlsp = lvlsp + lvlwdth[j]
                    lvlsps.append(lvlsp)
                lolvlsps.append(lvlsps)
                #print(lolvlsps)
                lolvlspsar = np.array(lolvlsps) - 1
                signal = np.concatenate(appendch)
                los = len(signal)
                currlvls = list(signal[lolvlspsar])

            #print('levl starts', len(lolvlsps), lolvlsps)
            #print('levl currs', len(currlvls), list(currlvls))
            #print('level widths', len(mxlvlwdthlist),mxlvlwdthlist)
            #print(len(startinds), len(lolvlspsar), len(mxlvlwdthlist), len(stopinds))
            eventlocs = [startinds, lolvlspsar, mxlvlwdthlist, stopinds]
    print(len(lolvlsps),lolvlsps)
    stopcork = np.array([0]*int(los/(numpulses+10)))
    signal = np.concatenate([signal,stopcork])
    los = len(signal)
    #plt.plot(signal)
    #plt.plot(signal)
    #plt.show()
    #print('startinds',len(startinds),startinds)
    #print('stopinds',len(stopinds),stopinds)
    #intereventtimes = (np.sort(np.diff(startinds)))
    #plt.hist(intereventtimes,50)
    #plt.show()
    #print(len(startinds),len(stopinds))
    print('Num points part of events and len of sig:',locop,len(signal))
    print('Num points part of events, len of baseline,percent:', locop, len(signal)-locop,(locop/(len(signal)-locop))*100)
    #plt.figure(figsize=(10,5))
    #plt.plot(signal,linewidth = '0.6',color = 'DarkBlue')
    #plt.show()
    return signal,concpulses,eventlocs,lvlnmlist

def add_white_noise(sig,concpulses):
    size = len(sig)
    low = 0 #np.std(concpulses)/3
    high = np.std(concpulses)/5
    #low = 0
    #high = np.mean(concpulses)

    noise = np.random.uniform(low,high,size)
    noised = sig+noise

    nstd = np.std(noised)
    print('Noised STD:',nstd)
    where = np.where(abs(noised) >= 5*nstd)
    wherehigh = np.squeeze(where)
    print('Npoints above and Indices:',len(wherehigh),wherehigh)
    #plt.plot(noised,linewidth = '0.6',color='DarkGreen')
    #plt.scatter(wherehigh,noised[wherehigh],)
    #plt.show()
    return noised,noise


def add_colornoise(sig,order):
    ln = len(sig)
    colornoise = np.zeros(ln)
    for od in order:
        colornoisef = cn.powerlaw_psd_gaussian(od,ln)
        colornoise = colornoise +colornoisef
    colornoised = colornoise + sig

    return colornoised,colornoise

def add_ac_noise(sig,acfrq = 50,harmonic = 3):
    ln = int(len(sig))
    acnoise = [0]*ln
    ampl = 0.01*max(sig)
    harmonics = np.arange(1,harmonic*2,2)
    print(harmonics)

    for harmonic in harmonics:
        acpwdt = ln/(harmonic*acfrq)
        ampl = ampl/(harmonic**2)
        acpulse = ampl * np.sin((np.arange(0,ln)) * (2*np.pi/acpwdt))
        acnoise = acnoise + acpulse
    acnoised = sig + acnoise
    return acnoised,acnoise

def sinudrifter(signal, numconcs, maxamp):
    ln = len(signal)
    try:
        los = int(ln / numconcs)  # length of segment
        print('ln,los:',ln,los)
        dom = np.arange(0, los, 1)
        sigbas = []
        subsigsum = []
        for w in range(0, numconcs):
            amp = random.random() * maxamp
            har = random.randint(1, 2) #harmonics
            sig = amp * np.sin(har * dom * 2 * np.pi / los)
            sigbas = sigbas + list(sig)
            loss = int(ln / random.randint(1, numconcs))  # length of subsegment
            doms = np.arange(0, loss, 1)
            subsig = amp * np.sin(doms * 2 * np.pi / loss)
            subsigsum = subsigsum + list(subsig)
            subsigsum = subsigsum[:ln]
        if len(sigbas) < ln:  # rounding off to integers may reduce the number of elements to equate for the same length.
            diff = ln - len(sigbas)
            driftsig = sigbas + [0] * diff
        else:
            driftsig = sigbas[:ln]

        print(len(signal), len(driftsig), ln)
        netdrift = np.array(driftsig)
        sinudrifted_sig = np.array(signal)+np.array(netdrift)
    except:
        sinudrifted_sig = signal
        netdrift = np.zeros(ln)
        print('NO SINUSOIDAL DRIFT ADDED')

    return sinudrifted_sig,netdrift


def abrupt_drifter(sig, nstepwins, driftmaxmag, maxnsteps):
    ln = len(sig)
    try:
        los = ln // nstepwins
        stepdrifts = []
        for stepwin in range(0, nstepwins):
            stepinwin = []
            nsteps = random.randint(2, maxnsteps)
            for steps in range(1, nsteps):
                driftmag = random.randrange(-driftmaxmag,driftmaxmag)
                stepwidth = random.randint(2, los // steps)
                step = [driftmag] * stepwidth
                stepdrifts.append(step)
        stepdrifts = np.concatenate(stepdrifts)
        print(len(stepdrifts))
        diff = abs(ln - len(stepdrifts))
        if diff != 0:
            stepdrifts = np.pad(stepdrifts, pad_width=diff, mode='constant', constant_values=0)

        stepdrifts = stepdrifts[:ln]
        stepdrifted = sig + stepdrifts
        print(len(stepdrifts))
    except:
        stepdrifted = sig
        stepdrifts = np.zeros(ln)
        print('NO ABRUPT DRIFTS ADDED')
    return stepdrifted,stepdrifts

def rcdeform(signal,sampfreq,R = 8.5e9 ,C = 2e-15,):
    DT = 1/sampfreq
    tau = R*C
    print('DT','tau','DT/tau',DT,tau,DT/tau)
    out = np.zeros(len(signal)+1)
    out[0] = signal[0]
    for i in range(0,len(signal)):
        out[i+1] = (signal[i] - out[i])*(DT/(R*C)) + out[i]
    return out

def rms(sig):
    return np.sqrt(np.mean(np.square(sig)))

def save_dict_to_txt(my_dict, filename):
    with open(filename, "w") as f:
        for key, value in my_dict.items():
            f.write(f"{key}: {value}\n")


def generate_sim_signal(numpulses,mincurr,maxcurr,minpwd,maxpwd,sparse_fac,sampfreq,colornoiseorders,nsigma,numconcs,maxamposin,nstepwins,driftmaxmag,maxnsteps,vshift = 10,dist = 'exp',multilevel = 'False',mixratio = 0.31,resistance = 8.5e9,capacitance = 2e-15,sequence = ['A','T','G','C'],currents = [-8,-5,-3,-10],pulsewidths = [10,10,10,10],shuffle = 'true'):
    p,concpulses,eventlocs,lvlnmlist = gen_events(numpulses,mincurr,maxcurr,minpwd,maxpwd,sparse_fac,dist = dist,multilevel = multilevel,mixratio=mixratio,sequence = sequence,currents = currents,pulsewidths = pulsewidths,shuffle = shuffle)
    w,jw = add_white_noise(p,concpulses)
    c,cj = add_colornoise(w,colornoiseorders)
    a,aj = add_ac_noise(c)


    all_the_noise = jw+cj+aj
    prcdeformed = rcdeform(p, sampfreq, R=resistance, C=capacitance)
    prcdeformed = prcdeformed[:len(all_the_noise)]

    stdatn = np.std(all_the_noise)
    print('STDATN,:', stdatn)

    noise_scaling_factor = abs(abs(mincurr/stdatn)-nsigma)
    noise_scaling_factor2 = abs(abs(maxcurr / stdatn)-nsigma)
    noise_scaling_factor3 = abs(abs(abs(np.mean(concpulses))/stdatn)-nsigma)

    print('MMINCURR','MAXCURR','Nsigma',mincurr,maxcurr,nsigma)
    print('NOISE SCALING FACTOR:',noise_scaling_factor)
    print('NOISE SCALING FACTOR:', noise_scaling_factor2)

    scaled_noise = all_the_noise/noise_scaling_factor
    scaled_noise2 = all_the_noise/noise_scaling_factor2
    scaled_noise3 = all_the_noise/noise_scaling_factor3

    print('NEW STDS',np.std(scaled_noise),np.std(scaled_noise2),np.std(scaled_noise3),rms(scaled_noise),rms(scaled_noise2),rms(scaled_noise3))
    print('RATIOS',abs(mincurr/np.std(scaled_noise)),abs(maxcurr/np.std(scaled_noise2)),abs(maxcurr/np.std(scaled_noise3)))

    scalingfactor = abs((nsigma*stdatn)/mincurr)
    scalingfactor2 = abs((nsigma * stdatn) / maxcurr)
    scalingfactor3 = abs(nsigma *stdatn)/abs(np.mean(concpulses))
    print('SCALING FACTOR:', scalingfactor,scalingfactor2,scalingfactor3)

    scaled_sig = scalingfactor*prcdeformed
    scaled_sig2 = scalingfactor2*prcdeformed
    scaled_sig3 = scalingfactor3*prcdeformed

    scaled_sig4 = scaled_noise+prcdeformed
    scaled_sig5 = scaled_noise2+prcdeformed
    scaled_sig6 = scaled_noise3+prcdeformed
    noisy_final_sig = scaled_sig2 + all_the_noise

    '''fig, axs = plt.subplots(3,1, sharex=True, figsize=(7, 7), )

    axs[0,].plot(scaled_sig4, color='#676785', linewidth='0.3', )
    axs[0,].set_title('1. Ideal Signal')

    axs[1,].plot(scaled_sig5, color='#676895', linewidth='0.3')
    axs[1,].set_title('2. Ideal Signal + White Noise')


    axs[2,].plot(scaled_sig6, color='Maroon', linewidth='0.3')
    axs[2,].set_title('3. Ideal Signal + White Noise + 1/f Noise +1/f^2')



    fig.tight_layout()
    plt.show()'''
    '''plt.plot(scaled_sig4, linewidth='0.4')
    plt.plot(scaled_)sig5,linewidth = '0.4')
    #plt.plot(noisy_final_sig,linewidth = '0.4')
    #plt.scatter(np.arange(0, len(p)), p, s=1.2, c='Red')
    #plt.plot(p, linewidth='0.4')
    #plt.scatter(np.arange(0, len(p)), p, s=1.2, c='DarkBlue')
    plt.show()
    '''

    #plt.plot(p,color = 'DarkBlue', linewidth = '0.6')
    #plt.plot(scaled_sig, color = 'DarkRed',linewidth = '0.5')
    #plt.plot(scaled_sig2,color = 'DarkGreen',linewidth = '0.4')
    #plt.plot(scaled_sig3,color = 'Yellow',linewidth = '0.8')
    #plt.show()

    finalundriftedsignal = scaled_sig4
    los = len(finalundriftedsignal)
    vshifted = finalundriftedsignal + np.array([vshift]*los)

    fisig,drift = sinudrifter(vshifted,numconcs,maxamposin)  #signal with sinusoidal drifts and the drifting signal
    stepdrifted,stepdrift = abrupt_drifter(fisig,nstepwins,driftmaxmag,maxnsteps) #signal with abrupt drifts added and the signal which initiated abrupt drifts
    #rcdeformed2 = rcdeform(stepdrifted,sampfreq)
    netdrift = stepdrift + drift  #Total Drift
    noiseless = netdrift + p + np.array([vshift]*los)

    print('RMS NOISELESS:',rms(noiseless))
    print("RMS ALL THE NOISE:", rms(scaled_noise))
    print('RMS ENTIRE SIGNAL:',rms(stepdrifted))
    print('RMS undrifted:', rms(scaled_sig4))


    if multilevel == 'false':
        currvals = noiseless[np.array(eventlocs[0])]
        currvalsbl = noiseless[np.array(eventlocs[0])-1]
        curramp = abs(currvals-currvalsbl)
        print(currvals)
        eventwidth = np.array(eventlocs[1])-np.array(eventlocs[0])
        print(list(eventwidth))

        #Dicts to be written as .csv files
        evdetailsdict = {'Event Start Points': eventlocs[0],
                        'Event End Points': eventlocs[1],
                        'Event Width': eventwidth,
                         'Current level- Event  (nA)':currvals,
                         'Current level - Baseline (nA)':currvalsbl,
                         'Event Amplitude (nA)':curramp}
    elif multilevel == 'true' or 'mix':
        numlevels = len(sequence)
        lvlcurkeys,lvlwidkeys = [],[]
        levwdict,levcdict = {},{}
        for i in range(0,numlevels):
            lvlcurkey = f'Level {i} Current (nA)'
            lvlwidkey = f'Level {i} Width (dPoints)'

            lvlcurkeys.append(lvlcurkey)
            lvlwidkeys.append(lvlwidkey)
        levcdict = {lvlcurkey:[] for lvlcurkey in lvlcurkeys}
        levwdict = {lvlwidkey:[] for lvlwidkey in lvlwidkeys}
        meancurrentlist = []
        for i in range(0,len(eventlocs[0])):
            lolvlsps = eventlocs[1][i]
            lvlwidths = eventlocs[2][i]
            if lvlnmlist[i][0] != 'Single Level':
                meancurr = np.sum(np.array(noiseless[lolvlsps])*np.array(lvlwidths)/(len(lolvlsps)*np.array(lvlwidths)))
            else:
                meancurr = np.array(noiseless[lolvlsps])[0]
            print('meancurr', meancurr)
            meancurrentlist.append(meancurr)
            for j,lvl in enumerate(lolvlsps):
                if lvlnmlist[j] == 'Single Level':
                    if j == 0:
                        levcdict[lvlcurkeys[0]].append(noiseless[lvl])
                    else:
                        levcdict[lvlcurkeys[j]].append(0)
                else:
                    levcdict[lvlcurkeys[j]].append(noiseless[lvl])
                    levwdict[lvlwidkeys[j]].append(lvlwidths[j])


        eventwidth = np.array(eventlocs[3])-np.array(eventlocs[0])

        spepdict = {'Event Start Points': eventlocs[0],
                    'Event End Points': eventlocs[3],
                    'Event Width': eventwidth,
                    'Event Mean Current level (nA)':meancurrentlist}
        evdetailsdict = {**spepdict,**levcdict,**levwdict}
        print(len(spepdict['Event Start Points']),len(levcdict['Level 0 Current (nA)']),len(levwdict['Level 0 Width (dPoints)']))

    Time = np.arange(0,len(stepdrifted)/sampfreq,1/sampfreq)
    time = Time[:len(stepdrifted)]
    print('len(Time)',len(Time),'len(time)',len(time),'len(Current)',len(stepdrifted),'len(scaled_sig)',len(scaled_sig))
    sigdict = {'Time':time*1000,     #multiplied by 1000 since clampfit converts time in ms
               'Current':stepdrifted}
    time2 = np.arange(0,len(scaled_sig)/sampfreq,1/sampfreq)
    time2m = time2[:len(netdrift)]
    print(len(time2),len(time2m),len(netdrift))
    just_theeventsdict = {'Time':time2m*1000,
                          'Current':scaled_sig+netdrift}

    #Writing .csv files
    evdetaildf = pd.DataFrame(evdetailsdict)
    paramfilename = 'Sim_Sig_Param_log' + str(numpulses) + 'mpw' + str(maxpwd) + str(int(sampfreq / 1000)) + str(sparse_fac) + str(dist) + str(cdt.day) + str(cdt.month) + str(cdt.hour) + str(cdt.minute) + '.txt'
    save_dict_to_txt(default_values,paramfilename)
    evdetaildf.to_csv('Sim_Sig Event Details' + str(numpulses) + 'mpw' + str(maxpwd) + str(int(sampfreq / 1000)) + str(
        sparse_fac) + str(dist) + str(cdt.day) + str(cdt.month) + str(cdt.hour) + str(cdt.minute) + '.csv', index=False,
                      mode='w')

    sigdf = pd.DataFrame(sigdict)
    tdf = pd.DataFrame(just_theeventsdict)
    sigdf.to_csv('SimSig'+str(numpulses)+'mpw'+str(maxpwd)+str(int(sampfreq/1000))+str(sparse_fac)+str(dist)+str(cdt.day)+str(cdt.month)+str(cdt.hour)+str(cdt.minute)+'.csv',index=False)
    tdf.to_csv('SimSigEvonly' + str(numpulses)+'mpw'+str(maxpwd)+str(int(sampfreq/1000))+str(sparse_fac)+str(dist)+str(cdt.day)+str(cdt.month)+str(cdt.hour)+str(cdt.minute)+'.csv', index=False)


    display_dataframe(evdetaildf,canvas4,excel_tab)
    return stepdrifted,stepdrift,fisig,netdrift,p,scaled_noise,time

def PSD(currentsig,sampfreq):
    currfft = fft(currentsig)
    psd = np.abs(currfft)**2/sampfreq
    frq = fftfreq(len(currentsig),d=(1/sampfreq))
    return frq,psd


def plotanywhere(fignm, x, y, canvasnm, toolbarnm, master):
    plotn = fignm.add_subplot(111)
    if set_ax_to_time == True:
        plotn.plot(y, color=COLOUR, linewidth=0.5)
        plotn.set_xlabel('Counts')
        plotn.set_ylabel('Current (nA)')
    elif set_ax_to_time == False:
        plotn.plot(x, y, color=COLOUR, linewidth=0.5)
        plotn.set_xlabel('Time(s)')
        plotn.set_ylabel('Current (nA)')
    try:
        canvasnm.get_tk_widget().pack_forget()
    except AttributeError:
        pass
    canvasnm = FigureCanvasTkAgg(fignm, master=master)
    canvasnm.draw()
    toolbarnm.pack_forget()
    canvasnm.get_tk_widget().pack(side=TOP, fill=BOTH, expand=TRUE)
    toolbarnm = NavigationToolbar2Tk(canvasnm, Output_signal_tab)
    toolbarnm.update()


def submit_values(entries):
    # Retrieve the input values from the entry
    global default_values,canvas1,def_values,toolbar1,canvas3,toolbar3,canvas2,toolbar2,set_ax_to_time
    values = []
    count = 0
    for entry in entries:
        try:
            value = float(entry.get())
            values.append(value)
        except ValueError:
            try:
                value = entry.get()
                value = tuple(map(float, value.strip("()").split(",")))
                values.append(value)
            except:
                value = entry.get()
                #value = tuple(map(float, value.strip("()").split(",")))
                values.append(value)


    updated_values = {'numpulses': int(values[0]), 'mincurr': values[1], 'maxcurr': values[2],
                      'minpwd': values[3], 'maxpwd': values[4], 'sparse_fac': values[5],'vshift':float(values[6]), 'sampfreq': int(values[7]),
                      'colornoiseorders': ast.literal_eval(values[8]), 'nsigma': int(values[9]), 'numconcs': int(values[10]), 'maxamposin': values[11],
                      'nstepwins': int(values[12]), "driftmaxmag": values[13],'maxnsteps': int(values[14]), 'dist': values[15],
                      'multilevel':values[16].lower(),'mixratio':float(values[17]),'resistance':float(values[18]),'capacitance':float(values[19]),
                      'sequence':ast.literal_eval(values[20]), 'currents':ast.literal_eval(values[21]),'pulsewidths':ast.literal_eval(values[22]),
                      'shuffle':values[23].lower()}
    default_values.update(updated_values)
    print(default_values)
    numpulses = default_values['numpulses']
    mincurr = default_values['mincurr']
    maxcurr = default_values['maxcurr']
    minpwd = default_values['minpwd']
    maxpwd = default_values['maxpwd']
    sparse_fac = default_values['sparse_fac']
    vshift = default_values['vshift']
    sampfreq = default_values['sampfreq']

    colornoiseorders = default_values['colornoiseorders']
    nsigma = default_values['nsigma']
    numconcs = default_values['numconcs']

    maxamposin = default_values['maxamposin']

    nstepwins = default_values['nstepwins']
    driftmaxmag = default_values['driftmaxmag']
    maxnsteps = default_values['maxnsteps']
    dist = default_values['dist']
    multilevel = default_values['multilevel']
    mixratio = default_values['mixratio']
    resistance = default_values['resistance']
    capacitance = default_values['capacitance']
    sequence = default_values['sequence']
    currents = default_values['currents']
    pulsewidths = default_values['pulsewidths']
    shuffle = default_values['shuffle']
    stepdrifted, stepdrift, fisig, netdrift, p,scaled_noise,time = generate_sim_signal(numpulses, mincurr, maxcurr, minpwd,
                                                                              maxpwd, sparse_fac, sampfreq,
                                                                              colornoiseorders, nsigma, numconcs,
                                                                              maxamposin, nstepwins, driftmaxmag,
                                                                              maxnsteps,vshift, dist, multilevel,mixratio,resistance,capacitance,sequence,
                                                                              currents, pulsewidths,shuffle)


    #PLOTTING THE OUTPUT IN THE FIRST TAB
    fig = Figure(figsize=(700 * px, 400 * px))
    #plotanywhere(fignm = fig, x = time, y=stepdrifted, canvasnm=canvas1, toolbarnm=toolbar1, master=Output_signal_tab) FUNCTION TESTING
    plot1 = fig.add_subplot(111)
    if set_ax_to_time == True:
        plot1.plot(stepdrifted, color=COLOUR, linewidth=0.5)
        plot1.set_xlabel('Counts')
        plot1.set_ylabel('Current (nA)')
    elif set_ax_to_time == False:
        plot1.plot(time,stepdrifted, color=COLOUR, linewidth=0.5)
        plot1.set_xlabel('Time(s)')
        plot1.set_ylabel('Current (nA)')

    try:
        canvas1.get_tk_widget().pack_forget()
    except AttributeError:
        pass
    canvas1 = FigureCanvasTkAgg(fig, master=Output_signal_tab)
    canvas1.draw()
    toolbar1.pack_forget()
    canvas1.get_tk_widget().pack(side=TOP, fill=BOTH, expand=TRUE)
    toolbar1 = NavigationToolbar2Tk(canvas1, Output_signal_tab)
    toolbar1.update()

    #PLOTTING COMPONENTS IN THE SECOND SUBTAB
    fig2 = Figure(figsize=(700 * px, 400 * px))
    plot2 = fig2.add_subplot(111)
    plot2.plot(scaled_noise, label='Noise', color='#AB241D', linewidth='0.5')
    plot2.plot(p,label = 'Signal without Noise',color = '#352BB2',linewidth = '0.6')
    plot2.plot(netdrift,label = 'Total Drift',color = '#17BB55',linewidth = '0.8')
    plot2.legend(loc = 'lower right')
    try:
        canvas2.get_tk_widget().pack_forget()
    except AttributeError:
        pass
    canvas2 = FigureCanvasTkAgg(fig2, master=Comp_tab, )
    canvas2.draw()
    canvas2._tkcanvas.pack(side=BOTTOM, expand=True)
    canvas2.get_tk_widget().pack(side=BOTTOM, expand=True)
    toolbar2.pack_forget()
    toolbar2 = NavigationToolbar2Tk(canvas2, Comp_tab)
    toolbar2.update()
    #PLOTTING PSD IN THE SECOND TAB

    fig3 = Figure(figsize=(1200*px,500*px))
    plot3 = fig3.add_subplot(111)
    frq,psd = PSD(fisig,sampfreq)
    plot3.loglog(frq,psd,color = 'DarkGreen',linewidth = '0.5')
    plot3.set_xlabel('log(Frequency (Hz))')
    plot3.set_ylabel('log(Power/Frequency (nA^2/Hz))')
    plot3.set_title('PSD')
    try:
        canvas3.get_tk_widget().pack_forget()
    except AttributeError:
        pass
    canvas3 = FigureCanvasTkAgg(fig3, master=framepsd)
    canvas3.draw()
    toolbar3.pack_forget()
    canvas3._tkcanvas.pack(side=BOTTOM, expand=True)
    canvas3.get_tk_widget().pack(side=BOTTOM, expand=True)
    toolbar3 = NavigationToolbar2Tk(canvas3,framepsd)
    toolbar3.update()

def display_dataframe(df,canvas,window):
    # Create Tkinter window
    #canvas.delete('all')
    clear_frame(window)
    canvas = Canvas(window)
    vscrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
    hscrollbar = ttk.Scrollbar(window, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=vscrollbar.set)
    canvas.configure(xscrollcommand=hscrollbar.set)

    # Place the scrollbar on the right side and fill the canvas
    vscrollbar.pack(side="right", fill="y")
    hscrollbar.pack(side = 'bottom',fill='x')
    canvas.pack(side="left", fill="both", expand=True)

    # Create a Treeview widget
    tree = ttk.Treeview(canvas)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"



    # Configure Treeview columns
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)  # Adjust column width as needed

    # Insert data rows into the Treeview
    for index, row in df.iterrows():
        values = [str(row[col]) for col in df.columns]
        tree.insert("", "end", values=values)

    # Add Treeview to the Tkinter window
    tree.pack(fill="both", expand=True)

    # Update the canvas scroll region after adding content
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

if __name__ == '__main__':
    window = Tk()
    # get the screen size
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # calculate the window size
    window_width = int(screen_width / 2)
    window_height = int(screen_height / 2)
    print(window_height, window_width)

    # set the window size
    window.geometry(f"{window_width}x{window_height}")
    #making tabs within the main window
    notebook_main = ttk.Notebook(window)
    Input_and_sig_tab = Frame(notebook_main) #main tab1
    psd_tab = Frame(notebook_main) #main tab2
    excel_tab = Frame(notebook_main)

    #filling tab1
    notebook_main.add(Input_and_sig_tab, text="Inputs and Signal")
    notebook_main.add(psd_tab, text = 'PSD')
    notebook_main.add(excel_tab,text='Event Details')
    notebook_main.pack(fill="both", expand=True)

    #frame containers within the tab and sub frames
    frame = Frame(master=Input_and_sig_tab)
    frame.pack(expand=True, fill=None, anchor='center')
    frame1 = Frame(master=frame)
    frame2 = Frame(master=frame)
    frame3 = Frame(master=frame)
    frame4 = Frame(master=frame)

    frame1.grid(row=0, column=0, )
    frame2.grid(row=0, column=1, )
    frame3.grid(row=1, column=0, )
    frame4.grid(row=1, column=1, )

    #Mechanism for recieving inputs

    numpulses = default_values['numpulses']
    mincurr = default_values['mincurr']
    maxcurr = default_values['maxcurr']
    minpwd = default_values['minpwd']
    maxpwd = default_values['maxpwd']
    sparse_fac = default_values['sparse_fac']
    vshift = default_values['vshift']
    sampfreq = default_values['sampfreq']

    colornoiseorders = default_values['colornoiseorders']
    nsigma = default_values['nsigma']
    numconcs = default_values['numconcs']

    maxamposin = default_values['maxamposin']

    nstepwins = default_values['nstepwins']
    driftmaxmag = default_values['driftmaxmag']
    maxnsteps = default_values['maxnsteps']
    dist = default_values['dist']
    multilevel = default_values['multilevel']
    mixratio = default_values['mixratio']
    resistance = default_values['resistance']
    capacitance = default_values['capacitance']
    sequence = default_values['sequence']
    currents = default_values['currents']
    pulsewidths = default_values['pulsewidths']
    shuffle = default_values['shuffle']

    var_names = ['Number of Translocations','Minimum Current (nA)','Maximum Current (nA)','Minimum Event Width (Datapoints)','Maximum Event Width (Datapoints)',
                 'Event Density Factor (%)','Baseline Shift (nA)','Sampling Frequency','Order of Colornoises (List)','N sigma','Num of Sines for Drift',
                 'Maximum Amplitude of Sine Drift (nA)','Number of Square waves in Drift','Maximum Amplitude of Square wave (nA)','Maximum number of steps in sq drift',
                 'Type of Distribution','Multilevel Events? (True/False/Mix)','Fraction of events, Multilevel','Resistance (Ohms)','Capacitance(Farads)',
                 'Sequence of Multilevels (list of Strings)','Current values of each level (list (nA))','Width of each level (Datapoints)','Shuffle']
    def_values = [numpulses,mincurr,maxcurr,minpwd,maxpwd,sparse_fac,vshift,sampfreq,colornoiseorders,
                  nsigma,numconcs,maxamposin,nstepwins,driftmaxmag,maxnsteps,dist,multilevel,mixratio,
                  resistance,capacitance,sequence,currents,pulsewidths,shuffle]
    print(def_values)
    entries = []
    ttkstyle = ttk.Style()
    ttkstyle.configure('TLabel',font = ('Tahoma',10))
    #populating fields to recieve inputs
    for i, var_name in enumerate(var_names):
        #row = i // 2
        #col = i % 2
        row = i
        col = 0

        label = ttk.Label(frame1, text=var_name,style = 'TLabel')
        if isinstance(def_values[i], tuple):
            # Convert tuple to string with parentheses
            def_value_str = f"({', '.join(map(str, def_values[i]))})"
        else:
            def_value_str = str(def_values[i])
        entry_text = StringVar(value=def_value_str)

        entry = Entry(frame1, textvariable=entry_text)

        label.grid(row=row, column=col, padx=5, pady=5)
        entry.grid(row=row, column=col + 1, padx=5, pady=5)

        entries.append(entry)

    # Create a button to submit the input values
    print('Entries:', entries)

    # Buttons
    Submitbutton = ttk.Button(master=frame3, text="Submit", command=lambda:submit_values(entries))
    #timecounttoggle = ttk.Button(master=frame2, text='Time-Count Toggle', command=Toggletimencounts)
    Submitbutton.pack(side=RIGHT)
    #timecounttoggle.pack(side=TOP,anchor='ne')

    #Window title
    window.title('Nanopore Signal Simulator v2.1')



    notebookf2 = ttk.Notebook(frame2)

    Output_signal_tab = Frame(notebookf2)
    notebookf2.add(Output_signal_tab, text="Output Signal")
    notebookf2.pack(fill="both", expand=True)

    fig = Figure(figsize=(700 * px, 400 * px))
    canvas1 = FigureCanvasTkAgg(fig, master=Output_signal_tab, )
    canvas1.draw()
    canvas1._tkcanvas.pack(side=BOTTOM, expand=True)
    canvas1.get_tk_widget().pack(side=BOTTOM, expand=True)
    toolbar1 = NavigationToolbar2Tk(canvas1, Output_signal_tab)
    toolbar1.update()

    Comp_tab = Frame(master=notebookf2)
    notebookf2.add(Comp_tab,text = 'Signal Components')
    fig2 = Figure(figsize=(700 * px, 400 * px))
    canvas2 = FigureCanvasTkAgg(fig2, master=Comp_tab, )
    canvas2.draw()
    canvas2._tkcanvas.pack(side=BOTTOM, expand=True)
    canvas2.get_tk_widget().pack(side=BOTTOM, expand=True)
    toolbar2 = NavigationToolbar2Tk(canvas2, Comp_tab)
    toolbar2.update()


    #Label for Displaying Details and Information
    Info_label = ttk.Label(master = frame4,text = 'Nanopore Group at IISc')
    Info_label.pack()

    #SECOND TAB: PSD
    framepsd = Frame(master=psd_tab)
    framepsd.pack(expand=True, fill=None, anchor='center')
    fig3 = Figure(figsize=(1200 * px, 400 * px))
    canvas3 = FigureCanvasTkAgg(fig3, master=framepsd,)
    canvas3.draw()
    canvas3._tkcanvas.pack(side=BOTTOM, expand=True)
    canvas3.get_tk_widget().pack(side=BOTTOM, expand=True)
    toolbar3 = NavigationToolbar2Tk(canvas3, framepsd)
    toolbar3.update()

    #THIRD TAB : EVENT DETAILS
    canvas4 = Canvas(master=excel_tab)
    vscrollbar = ttk.Scrollbar(excel_tab, orient="vertical", command=canvas4.yview)
    hscrollbar = ttk.Scrollbar(excel_tab, orient="horizontal", command=canvas4.xview)
    canvas4.configure(yscrollcommand=vscrollbar.set)
    canvas4.configure(xscrollcommand=hscrollbar.set)

    # Place the scrollbar on the right side and fill the canvas
    vscrollbar.pack(side="right", fill="y")
    hscrollbar.pack(side='bottom', fill='x')
    canvas4.pack(side="left", fill="both", expand=True)

    # Update the canvas scroll region after adding content
    canvas4.update_idletasks()
    canvas4.config(scrollregion=canvas4.bbox("all"))
    window.mainloop()