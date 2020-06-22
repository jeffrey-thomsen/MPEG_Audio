# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:31:56 2020

@author: Jeffrey
"""

"""
Test script for testing my MPEG Audio coder
"""



import scipy.io.wavfile as wav

import numpy as np

import scipy.signal as scisig

import time

import matplotlib.pyplot as plt


global layer
global bitrate
global sampleRate

#%% load audio
"""
Loading a WAV file, reducing it to one channel and making sure the array has
the dimensions (n,1). Consider using only a short portion of maybe 1 second
x[0:44100,1] tot test the coder, as it is very slow
"""

# filename = 'data/audio/cupid_audio.wav'
# sampleRate, x=wav.read(filename)
# x = np.expand_dims(x[:,1], axis = 1)

# filename = 'data/audio/traffic_audio.wav'
# sampleRate, x=wav.read(filename)
# # traffic: 0:573300
# x = np.expand_dims(x[0:88200,1], axis = 1)
# # x = np.expand_dims(x[0:573300,1], axis = 1)
# # x = np.expand_dims(x[:,1], axis = 1)
# # x = np.expand_dims(np.mean(x[0:573300,:], axis=1), axis = 1)

filename = 'data/audio/nara_audio.wav'
sampleRate, x=wav.read(filename)
# nara: 9238950:9327150
# x = np.expand_dims(np.mean(x, axis=1), axis = 1)
x = np.expand_dims(np.mean(x[9238950:9327150,:], axis=1), axis = 1)
# x = np.expand_dims(x[:,1], axis = 1)

# filename = 'data/audio/watermelonman_audio.wav'
# sampleRate, x=wav.read(filename)
# # x = np.expand_dims(np.mean(x[20000:64100,:], axis=1), axis = 1)
# # x = np.expand_dims(np.mean( x[2593080:2637180,:], axis=1), axis = 1)
# x = np.expand_dims(np.mean( x[2593080:2681280,:], axis=1), axis = 1)
# # x = np.expand_dims(x[:,1], axis = 1)

# filename = 'data/audio/soulfinger_audio.wav'
# sampleRate, x=wav.read(filename)
# x = np.expand_dims(x[:,0], axis = 1)

# filename = 'data/audio/fixingahole_audio.wav'
# sampleRate, x=wav.read(filename)
# x = np.expand_dims(np.mean(x[0:441000,:], axis=1), axis = 1)

# filename = 'data/audio/smooth_audio.wav'
# sampleRate, x=wav.read(filename)
# # smooth: 176400:264600 9238950:9327150
# # x = x[176400:264600,:]
# x = np.expand_dims(np.mean(x[176400:264600,:], axis=1), axis = 1)

# filename = 'data/audio/tomsdiner_audio.wav'
# sampleRate, x=wav.read(filename)
# # x = x[176400:264600,:]
# x = np.expand_dims(np.mean(x[44100:136170,:], axis=1), axis = 1)


#%% 

# normalize values between -1 and 1, I suppose that's the values the coder wants to work with
x = x/32768 


# # pure tones and noise as alternative test signals
# x = np.transpose(np.array([np.sin(2*np.pi*918.75*np.linspace(0,0.5,22051))]))
# x = np.transpose(np.array([2*(0.5-np.random.uniform(size=22050))]))


#%% define bitrate
"""
The bitrate is the quality control parameter of this coder. The number
specifies the number of bits available to code one frame, representing
384 input samples
"""

layer = 2 # 1, 2  
bitrate = 64
smrModel = 'scf' #'psy' 'scf' 'spl' 'zero'
Aweighting = True


if layer==1:
    bpf = 384
    frameLength = 12
elif layer==2:
    bpf = 1152
    frameLength = 36
    import B2Loader as b2
    nbal, quantLevels, sbLimit, sumNbal, maxBitsSubband = b2.B2Loader(sampleRate,bitrate)
nTotalBits = int(np.floor(bitrate*1000/sampleRate*bpf))
bps = nTotalBits/bpf
#nTotalBits = 3072 # 768 equals 2bps

file = open("layer", "wb")
np.save(file, layer)
file.close
file = open("bitrate", "wb")
np.save(file, bitrate)
file.close
file = open("sampleRate", "wb")
np.save(file, sampleRate)
file.close

print("\n")
print("Test track:",filename)
print("Layer",layer,", bitrate",bitrate,"kbit/s or %.2f"%bps,"bps")
print("SMR model:",smrModel,", Aweighting:",Aweighting,"\n")

import mpegAudioFunctions as mpeg
#%%

"""
The following lines represent the encoding and decoding process, divided up
into three parts of time-to-frequency mapping, bit allocation/quantizing and
decoding (i.e. frequency-to-time mapping)
"""
#%% calculate polyphase filterbank output
start = time.time()

subSamples = mpeg.feedCoder(x)

end = time.time()
print("Subband samples calculated in\n %.2f"%(end - start),"seconds")

#%% Encoding

start = time.time()
    
transmitFrames = mpeg.encoder(subSamples,nTotalBits,x,sampleRate,smrModel,Aweighting)
    
end = time.time()
print("scaling, bit allocation and quantizing in\n %.2f"%(end - start),"seconds")


#%% create matrix just for checking bit allocation
nBitsAllocated = []
scaleFactorInd = []
quantSubbandSamples = np.zeros((len(transmitFrames)*frameLength,32))
rescaledSubSamples = np.zeros((len(transmitFrames)*frameLength,32))

scaleFactorTable = np.load('data/mpeg_scale_factors.npy') # scale factors defined by MPEG

for iFrame in range(len(transmitFrames)):
    nBitsAllocated.append(transmitFrames[iFrame].nBitsSubband)
    scaleFactorInd.append(transmitFrames[iFrame].scalefactorInd)
    
    for iSubband in range(len(transmitFrames[iFrame].nSubbands)):
       quantSubbandSamples[frameLength*iFrame:frameLength*iFrame+frameLength,transmitFrames[iFrame].nSubbands[iSubband]] = np.array(transmitFrames[iFrame].quantSubbandSamples)[iSubband]
       #rescaledSubSamples[frameLength*iFrame:frameLength*iFrame+frameLength,transmitFrames[iFrame].nSubbands[iSubband]]  = np.array(transmitFrames[iFrame].quantSubbandSamples)[iSubband]*scaleFactorTable[transmitFrames[iFrame].scalefactorInd[iSubband]]
     
scaleFactorInd = [item for sublist in scaleFactorInd for item in sublist]
scaleFactorInd = np.array(scaleFactorInd).flatten()


#%% Decoding

start = time.time()

decodedSignal = mpeg.decoder(transmitFrames)

end = time.time()
print("Decoded signal in\n %.2f"%(end - start),"seconds\n")


"""
The following lines represent the evaluation stage used for the project report
"""
#%% Evaluation

# rate-distortion

# mean squared error

if ((len(decodedSignal)-len(x)-481)>0):
    mse = np.mean((x[0:len(decodedSignal)-481,0]-decodedSignal[481:-(len(decodedSignal)-len(x)-481)])**2)
else:
    mse = np.mean((x[0:len(decodedSignal)-481,0]-decodedSignal[481:])**2)
print("MSE = %1.5e"%mse)

# avg. squared value of output
mso = np.mean(decodedSignal[481:]**2)

snr = 10*np.log10(mso/mse)
print("SNR = %.5f"%snr,"dB")


# code length
def mapScfSelLayerII(selInfo):
    nScf = np.zeros(len(selInfo))
    for iBand in range(len(selInfo)):
        if selInfo[iBand]==0:
            nScf[iBand] = 3
        elif selInfo[iBand]==1:
            nScf[iBand] = 2
        elif selInfo[iBand]==2:
            nScf[iBand] = 1
        elif selInfo[iBand]==3:
            nScf[iBand] = 2
    return nScf


inputlength = len(x)*16

if layer==1:
    nSampleBits = np.sum(nBitsAllocated)*12
    nScaleFactorBits = np.count_nonzero(nBitsAllocated)*6
    nMiscBits=(128+32)*len(transmitFrames) 
elif layer==2:
    nSampleBits = int(np.sum(nBitsAllocated)*36)
    nScaleFactorBits = 0
    for iFrame in range(len(transmitFrames)):
        nScaleFactorBits += 6*np.sum(mapScfSelLayerII(transmitFrames[iFrame].scfSelectionInfo[np.nonzero(nBitsAllocated[iFrame])]))
    nMiscBits=(sumNbal+32)*len(transmitFrames)
    
codelength = nSampleBits + nScaleFactorBits + nMiscBits

compressionfactor = inputlength/codelength
print("compression factor = %.2f"%compressionfactor,"")

informationRate = nSampleBits/len(x)
print("information rate = %.2f"%informationRate,"\n")

# entropy estimate

def idealadaptivecodelength(x):
    # x - dataset
    
    # returns the dictionary A and a corresponding array counts with the number
    # of occurances
    A, counts = np.unique(x,return_counts=True)
    
    lenA = len(A)
    lenX = len(x)
    
    counter = np.zeros(lenA,dtype=int)
    phat=np.zeros(lenX)
    
    for i in range(len(x)):
        
        ind = A==x[i]
    
        phat[i]=((counter[ind]+1) / (i+lenA*1))
        
        counter[ind]+=1
        
    idealadaptivecodelength=-np.sum(np.log2(phat)) / lenX
    
    assert (np.all(counts==counter)), "Counting mistake!"
    
    return idealadaptivecodelength   
    

def empiricalselfentropy(x):
    
    lenX = len(x)
    A,counts = np.unique(x,return_counts=True)
    normcounts = counts/lenX
    
    assert(np.all(normcounts!=0)),"0 encountered!"
    
    phat=np.zeros(lenX)
    
    for i in range(lenX):
        
        ind = A==x[i]
    
        phat[i] = counts[ind]/lenX

    
    ie = -np.mean(np.log2(phat))
    #ie = -np.mean(np.log2((counts+1)/(len(x)+len(counts))))
    
    return ie

# scale factors
ese = empiricalselfentropy(scaleFactorInd)
iacl = idealadaptivecodelength(scaleFactorInd)
print("Scale factors")
print("Empirical self-entropy = %.5f"%ese)
print("< H(X) <")
print("Ideal adaptive code length = %.5f"%iacl,"\n")


#%% Comparison plots

# power spectra
f, Px = scisig.welch(x[0:len(decodedSignal)-481,0], fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py = scisig.welch(decodedSignal[481:],           fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

endSource = len(decodedSignal)-482
if len(decodedSignal-481)>len(x):
    endDecode = len(x)-(len(decodedSignal)-481)
else:
    endDecode = -1

axes[0, 0].set_title('time signal squared error')
axes[0, 0].plot((x[0:endSource,0]-decodedSignal[481:endDecode])**2)
axes[0, 0].grid(axis='x')
#
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

#%% save audio files

# bring data back into int16 format
xInt = np.round(x[:,0]*32768).astype(np.int16)
decodedInt=np.round(decodedSignal[480:]*32768).astype(np.int16)

wav.write('test_source.wav', 44100, xInt)
wav.write('test_recons.wav', 44100, decodedInt)


#%% misc. for evaluation

# scaleFactorIndArray=np.array(scaleFactorInd)
nBitsAllocated=np.array(nBitsAllocated)
plt.figure()
plt.imshow(np.transpose(nBitsAllocated))
plt.colorbar()
