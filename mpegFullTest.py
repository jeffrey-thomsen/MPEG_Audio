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
#x = x[20000:,:]
x = x[3400000:,:]
x = x/32768 # normalize values between -1 and 1, I suppose that's the values the coder wants to work with

#%% calculate polyphase filterbank output

subSamples = mpeg.feedCoder(x)

#%% initialize and push subband samples into a subbandFrame object

subFrame = mpeg.subbandFrame()

transmitFrames=[]
while len(subSamples)>=12:
    subSamples = subFrame.pushFrame(subSamples)
    
    #calculate scalefactors for current frame
    scaleFactorVal, scaleFactorInd = mpeg.calcScaleFactors(subFrame)
    
    # right now no conversion to binary yet
    #mpeg.codeScaleFactor(scaleFactorIndex)
    
    #bit allocation for one frame
    nBitsSubband, bscf, bspl, adb = mpeg.assignBits(subFrame)
    
    #quantize subband samples of one frame
    transmit = mpeg.quantizeSubbandFrame(subFrame,scaleFactorInd,nBitsSubband)
    transmitFrames.append(transmit)

#%% Decoding

decodedSignal = mpeg.decoder(transmitFrames)
#%%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(x[0:32160,0])
plt.plot(decodedSignal[480:])

plt.figure()
plt.plot(x[0:32160,0]-decodedSignal[480:])


#%% reveal error of quantization

# import numpy as np
# import matplotlib.pyplot as plt
# decodedSubband = np.zeros((32,12))
# for i in range(32):
#     decodedSubband[i,:]=transmitSubband[i]*transmitScalefactorVal[i]
    
# decErr=decodedSubband/subFrame.frame

# print(np.mean(decErr))