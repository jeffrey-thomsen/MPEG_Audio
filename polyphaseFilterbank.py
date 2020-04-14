#!/usr/bin/env python
# coding: utf-8

# # Implementation of the Polyphase filterbank

# In[1]:


import numpy as np

import mpegaudiofunctions as mpeg


# Simple test for analysisBuffer and polyphaseFilterbank

# In[2]:


# test pushBlock in Buffer and Filterbank just with random values
myBuffer = mpeg.analysisBuffer(np.random.rand(512))
myBuffer.pushBlock(np.random.rand(32))

myst = mpeg.polyphaseFilterbank(myBuffer)
print(myst)


# In[3]:


# test with a sine tone at 10kHz, assuming roughly 48kHz sampling rate
# --> maximum value in filterbank output seen in 13th entry

myBuffer = mpeg.analysisBuffer(np.sin(2*np.pi*10000*np.linspace(0,0.01,512)))
myst = mpeg.polyphaseFilterbank(myBuffer)
print(myst)


# # Loading audio files

# In[4]:


# this works

import scipy.io.wavfile as wav
filename = 'data/audio/watermelonman_audio.wav'
sampleRate, x=wav.read(filename)

xLeft  = x[:,0]
xRight = x[:,1]


# import routine which outputs bytes objects

#import wave
#chunk = 1024  
#open a wav format music
#filename = 'data/audio/lull_audio.wav'
#filename = 'data/audio/cometogether_audio.wav'
#f = wave.open(filename)

#f.getnchannels()
#f.getframerate()
#f.getnframes()
#f.getparams()
#f.readframes(1)

#f.close()

# more sophisticated audio library, not installed

#import librosa
#x, sample_rate = librosa.load('data/audio/lull_audio.wav', mono=True)


# Routine that takes blocks of audio and feeds it to the buffer, then groups subband samples into frames

# In[8]:


# simple test for feedCoder function

subbandSamples = mpeg.feedCoder(x)

# show that output of feedCoder is a numerical array
print(len(subbandSamples))
print(subbandSamples[0])
print(type(subbandSamples[0]))
print(type(subbandSamples[0][0]))


# # Implementation of Calculation of scale factors

# In[6]:


# simple test for subbandFrame object
mySubFrame = mpeg.subbandFrame(layer=1) # initialize
subbandSamples = mySubFrame.pushFrame(subbandSamples) # need to return popped samples list
#mySubFrame.pushFrame(subbandSamples[0:12])

print(mySubFrame.frame.shape)


# In[10]:


# simple test for codeScaleFactors

print(mpeg.calcScaleFactors(mySubFrame))

# simple test to show binary representation os scalefactors
scaleFactor, scaleFactorIndex = mpeg.calcScaleFactors(mySubFrame)

mpeg.codeScaleFactor(scaleFactorIndex)




# In[12]:
# bit allocation

nBitsSubband = mpeg.assignBits(mySubFrame)


# In[14]:
# simple test for quantizer

mpeg.quantizeSubbandFrame(mySubFrame,scaleFactor,nBitsSubband)




# In[16]:


# simple test of RL  to MS stereo

msx = mpeg.convRLtoMS(x)

print(x[1,:])
print(msx[1,:])


