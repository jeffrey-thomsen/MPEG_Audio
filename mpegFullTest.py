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
x = x[20000:108200,:]
x = x/32768 # normalize values between -1 and 1, I suppose that's the values the coder wants to work with



#x = np.transpose(np.array([np.sin(2*np.pi*918.75*np.linspace(0,0.5,22051))]))

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

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# plt.figure()
# plt.title('time signal comparison')
# plt.plot(x[:,0])
# plt.plot(decodedSignal[481:])

axes[0, 0].set_title('time signal squared error')
axes[0, 0].plot((x[0:len(decodedSignal)-481,0]-decodedSignal[481:])**2)
axes[0, 0].grid(axis='x')

# spectral
import numpy as np
import scipy.signal as scisig
f, Px = scisig.welch(x[0:len(decodedSignal)-481,0], fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py = scisig.welch(decodedSignal[481:],           fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

axes[0, 1].set_title('spectral error (absolute difference, max-normalized)')
axes[0, 1].plot(f,np.abs(Px-Py)/np.max(Px))
axes[0, 1].set_xscale('log')
axes[0, 1].grid(b=True,which='both')
axes[0, 1].set_xlim((10, 20000))

axes[1, 0].set_title('spectral error (ratio in dB)')
axes[1, 0].plot(f,10*np.log10(Py/Px))
axes[1, 0].set_xscale('log')
axes[1, 0].grid(b=True,which='both')
axes[1, 0].set_xlim((10, 20000))

axes[1, 1].set_title('spectral comparison in dB')
axes[1, 1].plot(f,10*np.log10(Px))
axes[1, 1].plot(f,10*np.log10(Py))
axes[1, 1].set_xscale('log')
axes[1, 1].grid(b=True,which='both')
axes[1, 1].set_xlim((10, 20000))

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