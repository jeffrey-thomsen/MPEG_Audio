# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:12:40 2020

@author: Jeffrey
"""
import numpy as np

import matplotlib.pyplot as plt

import scipy.signal as scisig

#%% 
sampleRate = 44100

x = np.load('eval_cupid/0192bpf/cupid_x.npy')
decodedSignal = np.load('eval_cupid/0192bpf/cupid_dS.npy')
nBitsAllocated = np.load('eval_cupid/0192bpf/cupid_nBAA.npy')
f, Py1 = scisig.welch(decodedSignal[481:], fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

# decodedSignal = np.load('eval_cupid/0288bpf/cupid_dS.npy')
# nBitsAllocated = np.load('eval_cupid/0288bpf/cupid_nBAA.npy')
decodedSignal = np.load('eval_cupid/0384bpf/cupid_dS.npy')
nBitsAllocated = np.load('eval_cupid/0384bpf/cupid_nBAA.npy')
f, Py2 = scisig.welch(decodedSignal[481:], fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

# decodedSignal = np.load('eval_cupid/0576bpf/cupid_dS.npy')
# nBitsAllocated = np.load('eval_cupid/0576bpf/cupid_nBAA.npy')
decodedSignal = np.load('eval_cupid/0768bpf/cupid_dS.npy')
nBitsAllocated = np.load('eval_cupid/0768bpf/cupid_nBAA.npy')
f, Py3 = scisig.welch(decodedSignal[481:], fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

# decodedSignal = np.load('eval_cupid/0960bpf/cupid_dS.npy')
# nBitsAllocated = np.load('eval_cupid/0960bpf/cupid_nBAA.npy')

# decodedSignal = np.load('eval_cupid/1152bpf/cupid_dS.npy')
# nBitsAllocated = np.load('eval_cupid/1152bpf/cupid_nBAA.npy')

# decodedSignal = np.load('eval_cupid/1344bpf/cupid_dS.npy')
# nBitsAllocated = np.load('eval_cupid/1344bpf/cupid_nBAA.npy')

decodedSignal = np.load('eval_cupid/1536bpf/cupid_dS.npy')
nBitsAllocated = np.load('eval_cupid/1536bpf/cupid_nBAA.npy')
f, Py4 = scisig.welch(decodedSignal[481:], fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)


# x = np.load('eval_0768bpf/nara_x.npy')
# decodedSignal = np.load('eval_0768bpf/nara_decodedSignal.npy')
# nBitsAllocated = np.load('eval_0768bpf/nara_nBitsAllocatedArray.npy')
# decodedSignal = np.load('eval_1536bpf/nara_decodedSignal.npy')
# nBitsAllocated = np.load('eval_1536bpf/nara_nBitsAllocatedArray.npy')


# x = np.load('eval_0768bpf/traffic_x.npy')
# decodedSignal = np.load('eval_0768bpf/traffic_decodedSignal.npy')
# nBitsAllocated = np.load('eval_0768bpf/traffic_nBitsAllocatedArray.npy')
# decodedSignal = np.load('eval_1536bpf/traffic_decodedSignal.npy')
# nBitsAllocated = np.load('eval_1536bpf/traffic_nBitsAllocatedArray.npy')


# mean squared error
mse = np.mean((x[0:len(decodedSignal)-481,0]-decodedSignal[481:])**2)
print("MSE =",mse)

# avg. squared value of output
mso = np.mean(decodedSignal[481:]**2)

snr = 10*np.log10(mso/mse)
print("SNR =",snr,"dB")


# code length
inputlength = len(x)*16

nSampleBits = np.sum(nBitsAllocated)*12
nScaleFactorBits = np.count_nonzero(nBitsAllocated)*6
nMiscBits=(128+32)*len(nBitsAllocated) 
codelength = nSampleBits + nScaleFactorBits + nMiscBits

compressionfactor = inputlength/codelength

print("compression factor =",compressionfactor)
print(codelength/len(x),"bits per sample","\n")


#%% Spectral comparison
sampleRate = 44100
# power spectra
f, Px = scisig.welch(x[0:len(decodedSignal)-481,0], fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py = scisig.welch(decodedSignal[481:],           fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

plt.close()
plt.figure()
plt.plot(f,10*np.log10(Py1/Px),label='0.5 bps')
plt.plot(f,10*np.log10(Py2/Px),label='1.0 bps')
plt.plot(f,10*np.log10(Py3/Px),label='2.0 bps')
plt.plot(f,10*np.log10(Py4/Px),label='4.0 bps')
plt.xscale('log')
plt.grid()
plt.xlim((10, 20000))
#plt.ylim((-4,4))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral deviation (dB)')
plt.legend()

plt.close()
plt.figure()
plt.plot(f,10*np.log10(Px))
plt.xscale('log')
plt.grid()
plt.xlim((10, 20000))
#plt.ylim((-4,4))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectrum (dB)')


#%% Rate-distortion plot

rate = [0.5, 0.625, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0]
dist = [5.54, 10.49, 14.61, 18.03, 21.67, 25.10, 28.78, 32.41, 35.71, 38.63, 63.61]

plt.close()
plt.figure()
plt.plot(rate,dist)
plt.scatter(rate,dist)
plt.xlabel('Rate (bits per sample)')
plt.ylabel('SNR (dB)')
plt.grid()
plt.xticks(np.array([0.5, 1, 2,  3, 4, 8]),['0.5','1','2','3','4','8']) 

#%% Entropy estimate

# scaleFactorInd = np.load('eval_cupid/0192bpf/cupid_sFIA.npy')
scaleFactorInd = np.load('eval_cupid/0768bpf/cupid_sFIA.npy')
# scaleFactorInd = np.load('eval_cupid/1536bpf/cupid_sFIA.npy')

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

ese = empiricalselfentropy(scaleFactorInd)
iacl = idealadaptivecodelength(scaleFactorInd)
print("Scale factors")
print("Empirical self-entropy =",ese)
print("< Source entropy <")
print("Ideal adaptive code length =",iacl,"\n")