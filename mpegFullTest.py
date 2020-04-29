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

import numpy as np

import time
#%% load audio
filename = 'data/audio/watermelonman_audio.wav'
sampleRate, x=wav.read(filename)
#x = x[20000:,:]
#x = x[3400000:,:]
x = x[20000:64100,:]
x = x/32768 # normalize values between -1 and 1, I suppose that's the values the coder wants to work with



x = np.transpose(np.array([np.sin(2*np.pi*918.75*np.linspace(0,0.5,22051))]))

#x = np.transpose(np.array([2*(0.5-np.random.uniform(size=22050))]))


#%% calculate polyphase filterbank output
start = time.time()
subSamples = mpeg.feedCoder(x)
end = time.time()
print("Subband samples calculated in")
print(end - start)

subSamplesArray=np.array(subSamples)
#%% initialize and push subband samples into a subbandFrame object

start = time.time()

subFrame = mpeg.subbandFrame()

transmitFrames=[]
while len(subSamples)>=12:
    subSamples = subFrame.pushFrame(subSamples)
    
    #calculate scalefactors for current frame
    scaleFactorVal, scaleFactorInd = mpeg.calcScaleFactors(subFrame)
    
    # right now no conversion to binary yet
    #mpeg.codeScaleFactor(scaleFactorIndex)
    
    #bit allocation for one frame
    nBitsSubband, bscf, bspl, adb = mpeg.assignBits(subFrame,scaleFactorVal)
    
    #quantize subband samples of one frame
    transmit = mpeg.quantizeSubbandFrame(subFrame,scaleFactorInd,nBitsSubband)
    transmitFrames.append(transmit)
    print("+1 frame")

end = time.time()
print("Scalefactor calculation, bit allocation and quantization in")
print(end - start)

#%% Decoding
start = time.time()

decodedSignal = mpeg.decoder(transmitFrames)

end = time.time()
print("Decoded signal in")
print(end - start)

#%% Error evaluation


import matplotlib.pyplot as plt

plt.figure()
plt.plot(x[:,0])
plt.plot(decodedSignal[480:])

plt.figure()
plt.title('squared error')
plt.plot((x[0:len(decodedSignal)-480,0]-decodedSignal[480:])**2)

# spectral
import numpy as np
import scipy.signal as scisig
f, Px = scisig.welch(x[0:len(decodedSignal)-480,0], fs=sampleRate, window='hanning', nperseg=16384, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py = scisig.welch(decodedSignal[480:],           fs=sampleRate, window='hanning', nperseg=16384, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

plt.figure()
plt.title('spectral error')
plt.plot(f,Px)
plt.plot(f,Py)
plt.plot(f,-np.abs(Px-Py))
plt.xscale('log')

plt.figure()
plt.title('spectral error')
plt.plot(f,10*np.log10(Px))
plt.plot(f,10*np.log10(Py))
#plt.plot(f,-np.abs(Px-Py))
#plt.xscale('log')

#%%
wav.write('test_source.wav', 44100, x)
wav.write('test_recons.wav', 44100, decodedSignal[480:])
#%% reveal error of quantization

# import numpy as np
# import matplotlib.pyplot as plt
# decodedSubband = np.zeros((32,12))
# for i in range(32):
#     decodedSubband[i,:]=transmitSubband[i]*transmitScalefactorVal[i]
    
# decErr=decodedSubband/subFrame.frame

# print(np.mean(decErr))