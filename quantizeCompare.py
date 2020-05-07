# -*- coding: utf-8 -*-
"""
Created on Fri May  8 00:10:10 2020

@author: Jeffrey
"""


"""
This creates comparison audio files to show what simple uniform quantizing 
at certain bit per sample rates sounds like, as opposed to spending the same
number of bits in the MPEG audio coder
"""


import mpegAudioFunctions as mpeg

import scipy.io.wavfile as wav

import numpy as np

import time

#%%

filename = 'data/audio/traffic_audio.wav'
sampleRate, x=wav.read(filename)
x = x[0:573300,1]
x = x[:,1]

# filename = 'data/audio/smooth_audio.wav'
# sampleRate, x=wav.read(filename)
# x = x[176400:264600,0]


#%%
# quantize from 16 down to 8 bit
xround8 = np.floor(x/256)*256
xroundInt8=np.round(xround8).astype(np.int16)

# quantize from 16 down to 4 bit
xround4 = np.floor(x/4096)*4096
xroundInt4=np.round(xround4).astype(np.int16)

# quantize from 16 down to 3 bit
xround3 = np.floor(x/8192)*8192
xroundInt3=np.round(xround3).astype(np.int16)

# quantize from 16 down to 2 bit
xround2 = np.floor(x/16384)*16384
xroundInt2=np.round(xround2).astype(np.int16)


wav.write('test_source.wav', 44100, x)
wav.write('test_quant08.wav', 44100, xroundInt8)
wav.write('test_quant04.wav', 44100, xroundInt4)
wav.write('test_quant03.wav', 44100, xroundInt3)
wav.write('test_quant02.wav', 44100, xroundInt2)