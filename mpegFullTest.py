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

import scipy.signal as scisig

import time

import matplotlib.pyplot as plt

#%% define bitrate

nTotalBits = 768 # 768 equals 2bps

#%% load audio

# filename = 'data/audio/fixingahole_audio.wav'
# sampleRate, x=wav.read(filename)
# x = np.expand_dims(np.mean(x[0:441000,:], axis=1), axis = 1)

# filename = 'data/audio/smooth_audio.wav'
# sampleRate, x=wav.read(filename)
# # smooth: 176400:264600 9238950:9327150
# # x = x[176400:264600,:]
# x = np.expand_dims(np.mean(x[176400:264600,:], axis=1), axis = 1)

# filename = 'data/audio/nara_audio.wav'
# sampleRate, x=wav.read(filename)
# # nara: 9238950:9327150
# # x = np.expand_dims(np.mean(x, axis=1), axis = 1)
# x = np.expand_dims(np.mean(x[9238950:9327150,:], axis=1), axis = 1)

filename = 'data/audio/traffic_audio.wav'
sampleRate, x=wav.read(filename)
# traffic: 0:573300
# x = np.expand_dims(x[0:88200,1], axis = 1)
# x = np.expand_dims(x[0:573300,1], axis = 1)
x = np.expand_dims(x[:,1], axis = 1)
# x = np.expand_dims(np.mean(x[0:573300,:], axis=1), axis = 1)

# filename = 'data/audio/watermelonman_audio.wav'
# sampleRate, x=wav.read(filename)
# x = np.expand_dims(np.mean(x[20000:64100,:], axis=1), axis = 1)
# x = np.expand_dims(np.mean( x[2593080:2637180,:], axis=1), axis = 1)


#%% 

x = x/32768 # normalize values between -1 and 1, I suppose that's the values the coder wants to work with

# pure tones and noise
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
    
    # push next 12 subband samples into the subbandFrame object
    
    subSamples = subFrame.pushFrame(subSamples)
    
    # calculate scalefactors for current frame
    
    #start = time.time()
    scaleFactorVal, scaleFactorInd = mpeg.calcScaleFactors(subFrame)
    #end = time.time()
    #print("scalefactor calculation in")
    #print(end - start)
    

    # bit allocation for current frame
    
    #start = time.time()
    nBitsSubband, bscf, bspl, adb = mpeg.assignBits(subFrame,scaleFactorVal,nTotalBits)
    #end = time.time()
    #print("bit allocation in")
    #print(end - start)


    # quantize subband samples of current frame and store in transmitFrame object
    
    #start = time.time()
    transmit = mpeg.quantizeSubbandFrame(subFrame,scaleFactorInd,nBitsSubband)
    transmitFrames.append(transmit)
    #end = time.time()
    #print("quantization in")
    #print(end - start)
    
    
end = time.time()
print("scaling, bit allocation and quantizing in")
print(end - start)




# create matrix just for checking bit allocation
nBitsAllocated = []
scaleFactorInd = []
quantSubbandSamples = np.zeros((len(transmitFrames)*12,32))
rescaledSubSamples = np.zeros((len(transmitFrames)*12,32))

scaleFactorTable = np.load('data/mpeg_scale_factors.npy') # scale factors defined by MPEG

for iFrame in range(len(transmitFrames)):
    nBitsAllocated.append(transmitFrames[iFrame].nBitsSubband)
    scaleFactorInd.append(transmitFrames[iFrame].scalefactorInd)
    
    for iSubband in range(len(transmitFrames[iFrame].nSubbands)):
       quantSubbandSamples[12*iFrame:12*iFrame+12,transmitFrames[iFrame].nSubbands[iSubband]] = np.array(transmitFrames[iFrame].quantSubbandSamples)[iSubband]
       rescaledSubSamples[12*iFrame:12*iFrame+12,transmitFrames[iFrame].nSubbands[iSubband]]  = np.array(transmitFrames[iFrame].quantSubbandSamples)[iSubband]*scaleFactorTable[transmitFrames[iFrame].scalefactorInd[iSubband]]
     
scaleFactorInd = [item for sublist in scaleFactorInd for item in sublist]
#%% Decoding

start = time.time()

decodedSignal = mpeg.decoder(transmitFrames)

end = time.time()
print("Decoded signal in")
print(end - start)





#%% Evaluation

# mean squared error

mse = np.mean((x[0:len(decodedSignal)-481,0]-decodedSignal[481:])**2)
print("MSE =",mse)

# avg. squared value of output
mso = np.mean(decodedSignal[481:]**2)

snr = 10*np.log10(mso/mse)
print("SNR =",snr,"dB")

# operational rate-distortion
opratedist = 0.5*np.log2(np.var(x)/mse)
print("Operational rate-distortion =",opratedist,"\n")

# code length

inputlength = len(x)*16

nSampleBits = np.sum(nBitsAllocated)*12
nScaleFactorBits = np.count_nonzero(nBitsAllocated)*6
nMiscBits=(128+32)*len(transmitFrames) 
codelength = nSampleBits + nScaleFactorBits + nMiscBits

compressionfactor = inputlength/codelength
print("compression factor =",compressionfactor,"\n")


# Entropy

# ideal adaptive code length
# def idealadaptivecodelength(x,context):
#     yay=0
#     nay=0
#     phat=[]
#     for i in range(context,len(x)):
#         value,counts = np.unique(x[i-context:i-1],return_counts=True)
#         if x[i] in value:
#             icount = counts[np.where(value==x[i])[0][0]]
#             yay+=1
#         else:
#             icount = 0
#             nay+=1
#         phat.append((icount+1) / (np.sum(counts)+len(counts)*1))
        
#     idealadaptivecodelength=-np.sum(np.log2(phat))
#     print("Yay to nay",yay,nay,"=",yay/nay)
    
#     return idealadaptivecodelength

# source
value,counts = np.unique(x,return_counts=True)
normcounts = counts/len(x)
sourceentropy = -np.mean(normcounts*np.log2(normcounts))
empiricalselfentropy = -np.mean(np.log2(normcounts))
bayesestimate = (counts+0.5) / (len(x)+len(counts)*0.5)
idealadaptivecodelength = -np.sum(np.log2(bayesestimate))
print("Source")
print("Empirical self-entropy =",empiricalselfentropy)
print("< True entropy <")
print("Ideal adaptive code length =",idealadaptivecodelength,"\n")

# subband samples
value,counts = np.unique(subSamplesArray,return_counts=True)
normcounts = counts/len(x)
sourceentropy = -np.mean(normcounts*np.log2(normcounts))
empiricalselfentropy = -np.mean(np.log2(normcounts))
bayesestimate = (counts+0.5) / (len(x)+len(counts)*0.5)
idealadaptivecodelength = -np.sum(np.log2(bayesestimate))
print("Source subband samples")
print("Empirical self-entropy =",empiricalselfentropy)
print("< True entropy <")
print("Ideal adaptive code length =",idealadaptivecodelength,"\n")

# quantized samples
value,counts = np.unique(quantSubbandSamples,return_counts=True)
normcounts = counts/len(x)
sourceentropy = -np.mean(normcounts*np.log2(normcounts))
empiricalselfentropy = -np.mean(np.log2(normcounts))
bayesestimate = (counts+0.5) / (len(x)+len(counts)*0.5)
idealadaptivecodelength = -np.sum(np.log2(bayesestimate))
print("Quantized subband samples")
print("Empirical self-entropy =",empiricalselfentropy)
print("< True entropy <")
print("Ideal adaptive code length =",idealadaptivecodelength,"\n")

# scale factors
value,counts = np.unique(scaleFactorInd,return_counts=True)
normcounts = counts/len(x)
sourceentropy = -np.mean(normcounts*np.log2(normcounts))
empiricalselfentropy = -np.mean(np.log2(normcounts))
bayesestimate = (counts+0.5) / (len(x)+len(counts)*0.5)
idealadaptivecodelength = -np.sum(np.log2(bayesestimate))
print("Scale factors")
print("Empirical self-entropy =",empiricalselfentropy)
print("< True entropy <")
print("Ideal adaptive code length =",idealadaptivecodelength,"\n")

# output signal
value,counts = np.unique(decodedSignal,return_counts=True)
normcounts = counts/len(x)
sourceentropy = -np.mean(normcounts*np.log2(normcounts))
empiricalselfentropy = -np.mean(np.log2(normcounts))
bayesestimate = (counts+0.5) / (len(x)+len(counts)*0.5)
idealadaptivecodelength = -np.sum(np.log2(bayesestimate))
print("Output signal")
print("Empirical self-entropy =",empiricalselfentropy)
print("< True entropy <")
print("Ideal adaptive code length =",idealadaptivecodelength,"\n")





#%%

# power spectra
f, Px = scisig.welch(x[0:len(decodedSignal)-481,0], fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py = scisig.welch(decodedSignal[481:],           fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

axes[0, 0].set_title('time signal squared error')
axes[0, 0].plot((x[0:len(decodedSignal)-481,0]-decodedSignal[481:])**2)
axes[0, 0].grid(axis='x')

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

# # show bit allocation
# plt.figure(figsize=(3, 7))
# plt.imshow(nBitsAllocated,vmin=0,vmax=15)
# plt.colorbar()
# plt.title('bit allocation')

#%% save audio files

# bring data back into int16 format
xInt = np.round(x[:,0]*32768).astype(np.int16)
decodedInt=np.round(decodedSignal[480:]*32768).astype(np.int16)

wav.write('test_source.wav', 44100, xInt)
wav.write('test_recons.wav', 44100, decodedInt)

#%% reveal error of quantization

# import numpy as np
# import matplotlib.pyplot as plt
# decodedSubband = np.zeros((32,12))
# for i in range(32):
#     decodedSubband[i,:]=transmitSubband[i]*transmitScalefactorVal[i]
    
# decErr=decodedSubband/subFrame.frame

# print(np.mean(decErr))