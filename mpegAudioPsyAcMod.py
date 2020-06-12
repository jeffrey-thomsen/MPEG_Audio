# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:03:41 2020

@author: Jeffrey
"""
import numpy as np

import scipy.fft as fft
import scipy.signal
import scipy.signal.windows as windowfun

"""
VERY IMPORTANT I STILL NEED TO GET THE CORRECT TIME SHIFTING IN HERE TO MATCH
THE SUBBAND FRAMES
"""
layer=1

scaleFactorVal = np.ones(32)*0.015625

sampleRate = 44100

bitrate = 128

#x = np.transpose(np.array([np.sin(2*np.pi*9470*np.linspace(0,2,88201))]))
#x = np.transpose(np.array([np.sin(2*np.pi*9470*np.linspace(0,2,88201))+np.sin(2*np.pi*947*np.linspace(0,2,88201))+np.sin(2*np.pi*200*np.linspace(0,2,88201))]))

import scipy.io.wavfile as wav
filename = 'data/audio/smooth_audio.wav'
sampleRate, x=wav.read(filename)
x = np.expand_dims(np.mean(x[176400:264600,:], axis=1), axis = 1)
x = x/32768


#%% Step 1: FFT Analysis

if layer==1:
    winLen = 512
elif layer==2:
    winLen = 1024

hannWindow = windowfun.hann(winLen,False)

sig = hannWindow * x[0:512,0]
X = fft.rfft(sig)
XFreq = fft.rfftfreq(winLen, d=1./sampleRate)

# y[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n)/n))

# power spectrum
Lk = 96 + 10 * np.log10(4/winLen**2 * np.abs(X)**2 * 8/3)


#%% Step 2: Determination of the sound pressure level

# map spectral lines to subbands
pqmfFreq = sampleRate/64 *(np.arange(0,32)+0.5)
pqmfFreqCutoff = sampleRate/64 *(np.arange(0,32)+1)

subbands = np.zeros(len(XFreq))

for iFreq in range(len(XFreq)):
    subbands[iFreq]=np.argmax(XFreq[iFreq]<=pqmfFreqCutoff)

# determine maximum level per subband
Lkmax = np.zeros(32)
for iBand in range(32):
    Lkmax[iBand] = max(Lk[subbands==iBand])
    
# calculate scalefactor equivalent SPL
if layer==1:
    Lscf = 20 * np.log10(scaleFactorVal * 32768) - 10
elif layer==2:
    # NEED TO PICK MAX OF THREE SCALEFACTORS PER FRAME
    Lscf = 20 * np.log10(scaleFactorVal * 32768) - 10

# max operation to compare which is bigger and use that
Lsb = np.zeros(32)
Lsb[Lkmax>=Lscf] = Lkmax[Lkmax>=Lscf]
Lsb[Lkmax< Lscf] =  Lscf[Lkmax< Lscf]


#%% Step 3: Considering the threshold in quiet

"""
read tables D.1a to D.1f depending on Layer and sampling rate
apply offset of -12 dB if bit rate is >= 96kbit/s per channel
"""
threshQuietFilename = 'D1_layer'+str(layer)+'_fs'+str(sampleRate)

threshQuiet = np.load('data/'+threshQuietFilename+'.npy')

LTqFreq = threshQuiet[:,1]
LTq = threshQuiet[:,3]
if (bitrate>=96):
    LTq -= 12
    
    
#%% Step 4: Finding of tonal and non-tonal components

# find tonal components

peaks, _ = scipy.signal.find_peaks(Lk)#, threshold=7)

tonalComp= []

LT = np.zeros(len(Lk))

if layer==1:
    nAreas = 3 # no of distinct areas of different critical bandwidth
    jRange = [[2],[2,3],[2,3,4,5,6]] # corresponding number of adjacent freq bins to evaluate
    jMax = [2,3,6]
    currPks = [(2<peaks)&(peaks<63), (63<=peaks)&(peaks<127), (127<=peaks)&(peaks<=250)] # indices of local peaks in the corresponding freq ranges
elif layer==2:
    nAreas = 4
    jRange = [[2],[2,3],[2,3,4,5,6],[2,3,4,5,6,7,8,9,10,11,12]]
    jMax = [2,3,6,12]
    currPks = [(2<peaks)&(peaks<63), (63<=peaks)&(peaks<127), (127<=peaks)&(peaks<255),(255<=peaks)&(peaks<=500)]
 

for iBandArea in range(nAreas): # loop through the areas of different critical bandwidth
    
    pks = peaks[currPks[iBandArea]]
    for k in range(len(pks)): # loop thrugh all local maxima in that freq range

        for j in jRange[iBandArea]: # loop through the corresponding number of adjacent freq bins to evaluate

            if ((Lk[pks[k]]-Lk[pks[k]-j]) < 7):
                tonal=False
                break
            if((Lk[pks[k]]-Lk[pks[k]+j]) < 7):
                tonal=False
                break
            tonal=True
        
        if tonal: # local maximum identified as peak, do the honors
            tonalComp.append(pks[k])
            
tonalComp = np.array(tonalComp)            
LT[tonalComp] = 10 * np.log10(10**(0.1*Lk[tonalComp-1]) + 10**(0.1*Lk[tonalComp]) + 10**(0.1*Lk[tonalComp+1]))
# probably wrong and replaced Lk[pks[k]-jMax[iBandArea]:pks[k]+jMax[iBandArea]+1] = -np.inf
Lk[tonalComp-1] = -np.inf
Lk[tonalComp] = -np.inf
Lk[tonalComp+1] = -np.inf

# sum remaining non-tonal components

# map spectral lines to critical bands
critBandsFilename = 'D2_layer'+str(layer)+'_fs'+str(sampleRate)
critBands = np.load('data/'+critBandsFilename+'.npy')
critBandsCutoff = np.append(critBands[:,2],24000)
critBandsCutoff = critBands[:,2]

cbands = np.zeros(len(XFreq))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

nonTonalComp = np.zeros(len(critBandsCutoff),dtype=int)
nonTonalComp[0] = find_nearest(XFreq, np.sqrt(1*critBandsCutoff[0]))
for i in range(1,len(critBandsCutoff)):
    nonTonalComp[i] = find_nearest(XFreq, np.sqrt(critBandsCutoff[i]*critBandsCutoff[i-1]))

for iFreq in range(len(XFreq)):
    cbands[iFreq]=np.argmax(XFreq[iFreq]<=critBandsCutoff)
    
# determine maximum level per subband
LN = np.zeros(len(Lk))
for iBand in range(len(critBandsCutoff)):
    LN[nonTonalComp[iBand]] = 10 * np.log10( np.sum(10**(0.1*Lk[cbands==iBand])) )
    
#%% Step 5: Decimation of tonal and non-tonal masking components

#%% Step 6: Calculation of individual masking thresholds

#%% Step 7: Calculation of the global masking threshold LTg

#%% Step 8: Determination of the minimum masking threshold

#%% Step 9: Calculation of the signal-to-mask-ratio
