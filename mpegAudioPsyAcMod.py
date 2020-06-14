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

#signal = np.transpose(np.array([np.sin(2*np.pi*9470*np.linspace(0,2,88201))]))
#signal = np.transpose(np.array([np.sin(2*np.pi*9470*np.linspace(0,2,88201))+np.sin(2*np.pi*947*np.linspace(0,2,88201))+np.sin(2*np.pi*200*np.linspace(0,2,88201))]))

import scipy.io.wavfile as wav
filename = 'data/audio/smooth_audio.wav'
sampleRate, signal=wav.read(filename)
signal = np.expand_dims(np.mean(signal[176400:264600,:], axis=1), axis = 1)
signal = signal/32768


#%% Step 1: FFT Analysis

if layer==1:
    winLen = 512
elif layer==2:
    winLen = 1024

hannWindow = windowfun.hann(winLen,False)

signal = hannWindow * signal[0:512,0]
X = fft.rfft(signal)
XFreq = fft.rfftfreq(winLen, d=1./sampleRate)

# y[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n)/n))

# calculate power spectrum
Lk = 96 + 10 * np.log10(4/winLen**2 * np.abs(X)**2 * 8/3)


#%% Step 2: Determination of the sound pressure level

# map FFT spectral lines to subbands
pqmfFreq = sampleRate/64 *(np.arange(0,32)+0.5)
pqmfFreqCutoff = sampleRate/64 *(np.arange(0,32)+1)

subbandsSPL = np.zeros(len(XFreq),dtype=int)

for iFreq in range(len(XFreq)):
    subbandsSPL[iFreq]=np.argmax(XFreq[iFreq]<=pqmfFreqCutoff)

# determine maximum level per subband
Lkmax = np.zeros(32)
for iBand in range(32):
    Lkmax[iBand] = max(Lk[subbandsSPL==iBand])
    
# calculate scalefactor equivalent SPL
if layer==1:
    Lscf = 20 * np.log10(scaleFactorVal * 32768) - 10
elif layer==2:
    """
    NEED TO PICK MAX OF THREE SCALEFACTORS PER FRAME
    """
    Lscf = 20 * np.log10(scaleFactorVal * 32768) - 10

# max operation to compare which is bigger and use that as SPL value
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
LTqBark = threshQuiet[:,2]
LTq = threshQuiet[:,3]
if (bitrate>=96):
    LTq -= 12
    
    
#%% Step 4: Finding of tonal and non-tonal components

# find tonal components


peakInd, _ = scipy.signal.find_peaks(Lk)#, threshold=7)

LtonalInd= []



# initialize counters and band areas
if layer==1:
    nAreas = 3 # no of distinct areas of different critical bandwidth
    jRange = [[2],[2,3],[2,3,4,5,6]] # corresponding number of adjacent freq bins to evaluate
    jMax = [2,3,6]
    currPks = [(2<peakInd)&(peakInd<63), (63<=peakInd)&(peakInd<127), (127<=peakInd)&(peakInd<=250)] # indices of local peaks in the corresponding freq ranges
elif layer==2:
    nAreas = 4
    jRange = [[2],[2,3],[2,3,4,5,6],[2,3,4,5,6,7,8,9,10,11,12]]
    jMax = [2,3,6,12]
    currPks = [(2<peakInd)&(peakInd<63), (63<=peakInd)&(peakInd<127), (127<=peakInd)&(peakInd<255),(255<=peakInd)&(peakInd<=500)]
 
# collect peak indices that qualify as tonal components
for iBandArea in range(nAreas): # loop through the areas of different critical bandwidth
    
    pks = peakInd[currPks[iBandArea]]
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
            LtonalInd.append(pks[k])
LtonalInd = np.array(LtonalInd)
            
# calculate tonal component levels   
Ltonal = np.zeros(len(Lk))                    
Ltonal[LtonalInd] = 10 * np.log10(10**(0.1*Lk[LtonalInd-1]) + 10**(0.1*Lk[LtonalInd]) + 10**(0.1*Lk[LtonalInd+1]))
LtonalList= Ltonal[np.nonzero(Ltonal)]

# remove tonal components from power spectrum
# probably wrong and replaced Lk[pks[k]-jMax[iBandArea]:pks[k]+jMax[iBandArea]+1] = -np.inf
Lk[LtonalInd-1] = -np.inf
Lk[LtonalInd] = -np.inf
Lk[LtonalInd+1] = -np.inf

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

LnoiseInd = np.zeros(len(critBandsCutoff),dtype=int)
LnoiseInd[0] = find_nearest(XFreq, np.sqrt(1*critBandsCutoff[0]))
for i in range(1,len(critBandsCutoff)):
    LnoiseInd[i] = find_nearest(XFreq, np.sqrt(critBandsCutoff[i]*critBandsCutoff[i-1]))

for iFreq in range(len(XFreq)):
    cbands[iFreq]=np.argmax(XFreq[iFreq]<=critBandsCutoff)
    
# sum up noise levels in every subband
Lnoise = np.zeros(len(Lk))
for iBand in range(len(critBandsCutoff)):
    Lnoise[LnoiseInd[iBand]] = 10 * np.log10( np.sum(10**(0.1*Lk[cbands==iBand])) )
LnoiseList = Lnoise[np.nonzero(Lnoise)]

# remove values that end up -inf
LnoiseInd =  LnoiseInd[np.isfinite(LnoiseList)]
LnoiseList = LnoiseList[np.isfinite(LnoiseList)]

#%% Step 5: Decimation of tonal and non-tonal masking components

# map tonal and noise spectral values to subsampled frequency indices
# and remove values smaller than theshold in quiet LTq
for i in range(len(LtonalInd)):
    LtonalInd[i] = find_nearest(LTqFreq, XFreq[LtonalInd[i]])    
    if LtonalList[i]<LTq[LtonalInd[i]]:
        LtonalList[i] = 0
for i in range(len(LnoiseInd)):
    LnoiseInd[i] = find_nearest(LTqFreq, XFreq[LnoiseInd[i]])
    if LnoiseList[i]<LTq[LnoiseInd[i]]:
        LnoiseList[i] = 0
  
# remove new zero entries from the tonal and noise component lists
LtonalInd  = LtonalInd[np.nonzero(LtonalList)]
LtonalList = LtonalList[np.nonzero(LtonalList)]
LnoiseInd  = LnoiseInd[np.nonzero(LnoiseList)]
LnoiseList = LnoiseList[np.nonzero(LnoiseList)]
  

# decimate tonal components within distance of less than 0.5 Bark
for i in range(len(LtonalInd)):
    if LTqBark[LtonalInd[i]]-LTqBark[LtonalInd[i-1]]<0.5:
        if LtonalList[i]>LtonalList[i-1]:
            LtonalList[i-1] = 0
        else:
            LtonalList[i] = 0
         
# remove new zero entries from the tonal component list        
LtonalInd =  LtonalInd[np.nonzero(LtonalList)]
LtonalList = LtonalList[np.nonzero(LtonalList)]


#%% Step 6: Calculation of individual masking thresholds

# masking index function for tonal maskers
def avtm(z):
    avtm = -1.525 - 0.275*z - 4.5
    return avtm

# masking index function for noise maskers
def avnm(z):
    avnm = -1.525 - 0.175*z - 0.5
    return avnm

# masking function
def vf(dz,L):
    
    if -3<=dz<-1:
        vf = 17 * (dz+1) - (0.4*L + 6)
    elif -1<=dz<0:
        vf = (0.4*L + 6) * dz
    elif 0<=dz<1:
        vf = -17 * dz
    elif 1<=dz<8:
        vf= -(dz - 1) * (17 - 0.15*L) - 17
    else:
        vf = -np.inf
    
    return vf

# individual masking threshold calculation according to Annex D.1 Step 6 of the MPEG standard
LTtm = np.zeros((len(LtonalInd),len(LTq)))
for j in range(len(LtonalInd)):
    for i in range(len(LTq)):
        LTtm[j,i] = LtonalList[j] + avtm(LTqBark[LtonalInd[j]]) + vf((LTqBark[i]-LTqBark[LtonalInd[j]]), LtonalList[j] )
        if np.isnan(LTtm[j,i]):
            print('LTtm',j,i)
LTnm = np.zeros((len(LnoiseInd),len(LTq)))
for j in range(len(LnoiseInd)):
    for i in range(len(LTq)):
        LTnm[j,i] = LnoiseList[j] + avtm(LTqBark[LnoiseInd[j]]) + vf((LTqBark[i]-LTqBark[LnoiseInd[j]]), LnoiseList[j] )
        if np.isnan(LTnm[j,i]):
            print('LTnm',j,i)


#%% Step 7: Calculation of the global masking threshold LTg

LTg = 10 * np.log10( 10**(0.1*LTq) + np.sum(10**(0.1*LTtm),axis=0) + np.sum(10**(0.1*LTnm),axis=0))

#%% Step 8: Determination of the minimum masking threshold

# map masking threshold to subbands
subbandsMask = np.zeros(len(LTq),dtype=int)

for iFreq in range(len(LTq)):
    subbandsMask[iFreq]=np.argmax(LTqFreq[iFreq]<=pqmfFreqCutoff)

# determine minimum masking threshold per subband
LTmin = np.zeros(32)
for iBand in range(max(subbandsMask)):
    LTmin[iBand] = max(LTg[subbandsMask==iBand])

# added this to avoid unnecessarily high SMRs at bands above the defined range
LTmin[max(subbandsMask):]=LTmin[max(subbandsMask)-1]

#%% Step 9: Calculation of the signal-to-mask-ratio

SMRa = Lsb - LTmin