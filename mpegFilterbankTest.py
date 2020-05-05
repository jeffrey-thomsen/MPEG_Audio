# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:21:11 2020

@author: Jeffrey
"""

"""
only for testing the analysis and synthesis filterbanks of the MPEG audio coder
"""

import mpegAudioFunctions as mpeg

import scipy.io.wavfile as wav

import numpy as np

import time

import matplotlib.pyplot as plt

import scipy.signal as scisig

#%% load audio
filename = 'data/audio/watermelonman_audio.wav'
sampleRate, x=wav.read(filename)
#x = x[20000:,:]
#x = x[3400000:,:]
x = x[20000:108200,:]
x = x/32768 # normalize values between -1 and 1, I suppose that's the values the coder wants to work with


# apparent filter center frequencies that do not lead to aliasing
#689.0625
#1378.125
#2067.1875
#2756.25

#x = np.transpose(np.array([np.cos(2*np.pi*(750)*np.linspace(0,2,88201))]))
#x = np.transpose(np.array([np.sin(2*np.pi*4823.4375*np.linspace(0,2,88201))+np.cos(2*np.pi*2756.25*np.linspace(0,2,88201))]))

#x = np.transpose(np.array([2*(0.5-np.random.uniform(size=88200))]))


#%% calculate polyphase filterbank output
start = time.time()
subSamples = mpeg.feedCoder(x)
end = time.time()
print("Subband samples calculated in")
print(end - start)

subSamplesArray=np.array(subSamples)
sSArray = np.array(subSamples)
#%% Decoding
start = time.time()

synBuff = mpeg.synthesisBuffer()
decodedSignal=[]
    
while subSamplesArray[:,0].size>0:
    subSamplesArray = synBuff.pushBlock(subSamplesArray)
    decodedSignal+= list(mpeg.synthesisFilterbank(synBuff))
decodedSignal = np.array(decodedSignal)

end = time.time()
print("Decoded signal in")
print(end - start)

#%% Error evaluation

plt.figure()
plt.plot(x[:,0])
plt.plot(decodedSignal[481:])

plt.figure()
plt.title('squared error')
plt.plot((x[0:len(decodedSignal)-481,0]-decodedSignal[481:])**2)

# spectral
f, Px = scisig.welch(x[0:len(decodedSignal)-481,0], fs=sampleRate, window='hamming', nperseg=32768, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py = scisig.welch(decodedSignal[481:],           fs=sampleRate, window='hamming', nperseg=32768, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

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
plt.xscale('log')

#%%
wav.write('test_source.wav', 44100, x)
wav.write('test_recons.wav', 44100, decodedSignal[480:])