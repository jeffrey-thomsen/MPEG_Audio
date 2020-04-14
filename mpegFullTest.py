# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:31:56 2020

@author: Jeffrey
"""

"""
cleaner, top-to-bottom test script for my MPEG Audio coder
"""

import mpegAudioFunctions as mpeg

import scipy.io.wavfile as wav


#%% load audio
filename = 'data/audio/watermelonman_audio.wav'
sampleRate, x=wav.read(filename)


#%% calculate polyphase filterbank output

subSamples = mpeg.feedCoder(x)

#%% initialize and push subband samples into a subbandFrame object

subFrame = mpeg.subbandFrame()

subSamples = subFrame.pushFrame(subSamples)

#%% calculate scalefactors for current frame

scaleFactorVal, scaleFactorInd = mpeg.calcScaleFactors(subFrame)

# right now no conversion to binary yet
#mpeg.codeScaleFactor(scaleFactorIndex)

#%% bit allocation for current frame

nBitsSubband, bscf, bspl, adb = mpeg.assignBits(subFrame)


# In[14]:
# simple test for quantizer

#mpeg.quantizeSubbandFrame(subFrame,scaleFactorVal,nBitsSubband)