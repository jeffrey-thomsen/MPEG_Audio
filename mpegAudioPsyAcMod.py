# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:03:41 2020

@author: Jeffrey
"""
import numpy as np

import scipy.fft as fft
import scipy.signal.windows as windowfun

"""
VERY IMPORTANT I STILL NEED TO GET THE CORRECT TIME SHIFTING IN HERE TO MATCH
THE SUBBAND FRAMES
"""
layer=1

scaleFactorVal = np.ones(32)*0.015625

sampleRate = 44100
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

bands = np.zeros(len(XFreq))
for iFreq in range(len(XFreq)):
    bands[iFreq]=np.argmax(XFreq[iFreq]<=pqmfFreqCutoff)

# determine maximum level per subband
Lkmax = np.zeros(32)
for iBand in range(32):
    Lkmax[iBand] = max(Lk[bands==iBand])
    
# calculate scalefactor equivalent SPL
if layer==1:
    Lscf = 20 * np.log10(scaleFactorVal * 32768) - 10
elif layer==2:
    # NEED TO PICK MAX OF THREE SCALEFACTORS PER FRAME
    Lscf = 20 * np.log10(scaleFactorVal * 32768) - 10

# max operation to compare which is bigger and use that
Lsb = np.zeros(32)
Lsb[Lkmax>=Lscf] =    Lkmax[Lkmax>=Lscf]
Lsb[Lkmax< Lscf] = Lscf[Lkmax< Lscf]


#%% Step 3: Considering the threshold in quiet

"""
read tables D.1a to D.1f depending on Layer and sampling rate
apply offset of -12 dB if bit rate is >= 96kbit/s per channel
"""

#%% Step 4: Finding of tonal and non-tonal components

#%% Step 5: Decimation of tonal and non-tonal masking components

#%% Step 6: Calculation of individual masking thresholds

#%% Step 7: Calculation of the global masking threshold LTg

#%% Step 8: Determination of the minimum masking threshold

#%% Step 9: Calculation of the signal-to-mask-ratio
