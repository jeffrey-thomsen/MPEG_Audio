# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:58:26 2020

@author: Jeffrey
"""
import numpy as np
import matplotlib.pyplot as plt


#%% Perceptual models test
"""
variable names: p-bitrate-song name-Layer
columns: bps,smrModel,SNR,compr,inforate,entrlo,entrhi
"""

p064dinerI=np.array([[1.45,'none',6.74,11.17,0.81,4.8596,4.8715],
[1.45,'Psy. Mod.',14.06,11.34,0.84,4.7568,4.7711],
[1.45,'SPL (A)',17.42,11.35,0.84,4.7019,4.7169],
[1.45,'Scalef. (A)',17.32,11.34,0.84,4.6836,4.6980],
[1.45,'SPL',17.21,11.33,0.84,4.6817,4.6958],
[1.45,'Scalef.',17.16,11.33,0.84,4.6661,4.6797],])

p064dinerII=np.array([[1.45,'none',6.70,11.45,1.12,4.7803,4.7882],
[1.45,'Psy. Mod.',16.20,11.32,1.20,4.7571,4.7683],
[1.45,'SPL (A)',20.16,11.31,1.20,4.7271,4.7394],
[1.45,'Scalef. (A)',19.92,11.31,1.20,4.7310,4.7431],
[1.45,'SPL',19.76,0,0,0,0],
[1.45,'Scalef.',19.53,0,0,0,0]])

p128wmmI=np.array([[2.90,'none',6.86,5.36,1.94,4.77610,4.7822],
[2.90,'Psy. Mod.',27.36,5.60,2.13,4.7927,4.8011],
[2.90,'SPL (A)',30.96,0,0,0,0],
[2.90,'Scalef. (A)',30.96,0,0,0,0],
[2.90,'SPL',30.19,0,0,0,0],
[2.90,'Scalef.',30.29,0,0,0,0]])

p128wmmII=np.array([[2.90,'none',13.46,5.64,2.51,4.7695,4.7757],
[2.90,'Psy. Mod.',30.56,5.61,2.58,4.7667,4.8050],
[2.90,'SPL (A)',33.31,5.61,2.59,4.7895,4.7977],
[2.90,'Scalef. (A)',33.19,0,0,0,0],
[2.90,'SPL',32.65,0,0,0,0],
[2.90,'Scalef.',32.71,0,0,0,0]])


p032cupidI=np.array([[0.73,'none',6.58,24.56,0.19,4.3065,4.3405],
[0.73,'Psy. Mod.',10.09,23.32,0.22,4.4186,4.4544],
[0.73,'SPL (A)',14.05,23.20,0.23,4.2133,4.2491],
[0.73,'Scalef. (A)',13.90,0,0,0,0],
[0.73,'SPL',14.24,0,0,0,0],
[0.73,'Scalef.',14.21,0,0,0,0]])

p032cupidII=np.array([[0.73,'none',6.12,23.07,0.52,4.5786,4.5928],
[0.73,'Psy. Mod.',15.65,22.88,0.54,4.5163,4.5367],
[0.73,'SPL (A)',19.37,22.87,0.55,4.3193,4.3416],
[0.73,'Scalef. (A)',19.15,0,0,0,0],
[0.73,'SPL',19.34,0,0,0,0],
[0.73,'Scalef.',19.18,0,0,0,0]])

#%% Layer comparisons test

#SPLA, LayerI/II, 32,64,96,128,192,256

"""
variable names: song name-Layer
columns: bps,bitrate,SNR,infoRate
"""

ldinerI=np.array([[0.73,32,10.01,0.22],
[1.45,64,17.42,0.84],
[2.18,96,22.96,1.47],
[2.90,128,27.38,2.13],
[4.35,192,35.34,3.46],
[5.80,256,44.30,4.86]])
ldinerII=np.array([[0.73,32,14.38,0.54],
[1.45,64,20.16,1.20],
[2.18,96,25.06,1.88],
[2.90,128,29.24,2.56],
[4.35,192,37.33,3.96],
[5.80,256,46.59,5.40]])

 
lcupidI=np.array([[0.73,32,14.05,0.23],
[1.45,64,22.42,0.85],
[2.18,96,27.41,1.48],
[2.90,128,33.48,2.16],
[4.35,192,42.37,3.49],
[5.80,256,50.32,4.86]])
lcupidII=np.array([[0.73,32,19.37,0.55],
[1.45,64,24.55,1.21],
[2.18,96,30.60,1.90],
[2.90,128,35.97,2.60],
[4.35,192,44.23,3.99],
[5.80,256,52.49,5.42]])


lnaraI=np.array([[0.73,32,8.38,0.22],
[1.45,64,17.06,0.85],
[2.18,96,20.88,1.48],
[2.90,128,23.84,2.14],
[4.35,192,30.98,3.45],
[5.80,256,40.84,4.85]])
lnaraII=np.array([[0.73,32,13.63,0.55],
[1.45,64,19.05,1.23],
[2.18,96,22.28,1.90],
[2.90,128,25.24,2.58],
[4.35,192,33.72,3.98],
[5.80,256,43.34,5.42]])

lwmmI=np.array([[0.73,32,14.36,0.23],
[1.45,64,21.48,0.85],
[2.18,96,25.25,1.48],
[2.90,128,30.96,2.15],
[4.35,192,40.00,3.48],
[5.80,256,47.51,4.86]])
lwmmII=np.array([[0.73,32,19.29,0.55],
[1.45,64,23.08,1.21],
[2.18,96,28.24,1.90],
[2.90,128,33.31,2.59],
[4.35,192,41.73,3.98],
[5.80,256,49.58,5.42]])


lsoulI=np.array([[0.73,32,7.30,0.23],
[1.45,64,13.84,0.84],
[2.18,96,19.06,1.46],
[2.90,128,23.29,2.11],
[4.35,192,31.22,3.43],
[5.80,256,39.93,4.88]])
lsoulII=np.array([[0.73,32,10.56,0.54],
[1.45,64,16.32,1.21],
[2.18,96,21.32,1.89],
[2.90,128,25.51,2.58],
[4.35,192,33.80,3.98],
[5.80,256,39.28,5.43]])
#%%
meanSNRI = np.mean([ldinerI[:,2],lcupidI[:,2],lnaraI[:,2],lwmmI[:,2],lsoulI[:,2]],axis=0)
meanSNRII = np.mean([ldinerII[:,2],lcupidII[:,2],lnaraII[:,2],lwmmII[:,2],lsoulII[:,2]],axis=0)

#%% Perceptual Plots

plt.figure()
# plt.plot(p064dinerI[:,1],p064dinerI[:,2].astype(np.float),label='Toms Diner Layer I')
plt.scatter(p032cupidI[:,1],p032cupidI[:,2].astype(np.float),label='Cupid 32 kbit/s Layer I')
plt.scatter(p032cupidII[:,1],p032cupidII[:,2].astype(np.float),label='Cupid 32 kbit/s Layer II')

plt.scatter(p064dinerI[:,1],p064dinerI[:,2].astype(np.float),label='Toms Diner 64 kbit/s Layer I')
plt.scatter(p064dinerII[:,1],p064dinerII[:,2].astype(np.float),label='Toms Diner 64 kbit/s Layer II')

plt.scatter(p128wmmI[:,1],p128wmmI[:,2].astype(np.float),label='Watermelon Man 128 kbit/s Layer I')
plt.scatter(p128wmmII[:,1],p128wmmII[:,2].astype(np.float),label='Watermelon Man 128 kbit/s Layer II')


plt.xlabel('SMR Model')
plt.ylabel('SNR (dB)')
plt.grid(which='both',axis='y')
# plt.legend()

#%% Layer Plots

plt.figure()
plt.plot(ldinerI[:,0],meanSNRI,label='Mean SNR Layer I')
plt.scatter(ldinerI[:,0],meanSNRI)
plt.plot(ldinerII[:,0],meanSNRII,label='Mean SNR Layer II')
plt.scatter(ldinerII[:,0],meanSNRII)
plt.xlabel('Rate (bits per sample)')
plt.ylabel('SNR (dB)')
plt.grid()
plt.legend()

plt.figure()
plt.plot(ldinerI[:,0],meanSNRII-meanSNRI,label='Mean SNR Difference')
plt.scatter(ldinerI[:,0],meanSNRII-meanSNRI)
plt.xlabel('Rate (bits per sample)')
plt.ylabel('SNR (dB)')
plt.grid()


plt.figure()
plt.plot(ldinerI[:,0],ldinerI[:,2],label='Toms Diner Layer I')
plt.scatter(ldinerI[:,0],ldinerI[:,2])
plt.plot(ldinerII[:,0],ldinerII[:,2],label='Toms Diner Layer II')
plt.scatter(ldinerII[:,0],ldinerII[:,2])
plt.xlabel('Rate (bits per sample)')
plt.ylabel('SNR (dB)')
plt.grid()
plt.legend()

plt.figure()
plt.plot(lcupidI[:,0],lcupidI[:,2],label='Cupid Layer I')
plt.scatter(lcupidI[:,0],lcupidI[:,2])
plt.plot(lcupidII[:,0],lcupidII[:,2],label='Cupid Layer II')
plt.scatter(lcupidII[:,0],lcupidII[:,2])
plt.xlabel('Rate (bits per sample)')
plt.ylabel('SNR (dB)')
plt.grid()
plt.legend()

plt.figure()
plt.plot(lnaraI[:,0],lnaraI[:,2],label='Nara Layer I')
plt.scatter(lnaraI[:,0],lnaraI[:,2])
plt.plot(lnaraII[:,0],lnaraII[:,2],label='Nara Layer II')
plt.scatter(lnaraII[:,0],lnaraII[:,2])
plt.xlabel('Rate (bits per sample)')
plt.ylabel('SNR (dB)')
plt.grid()
plt.legend()

plt.figure()
plt.plot(lwmmI[:,0],lwmmI[:,2],label='Watermelon Man Layer I')
plt.scatter(lwmmI[:,0],lwmmI[:,2])
plt.plot(lwmmII[:,0],lwmmII[:,2],label='Watermelon Man Layer II')
plt.scatter(lwmmII[:,0],lwmmII[:,2])
plt.xlabel('Rate (bits per sample)')
plt.ylabel('SNR (dB)')
plt.grid()
plt.legend()

plt.figure()
plt.plot(lsoulI[:,0],lsoulI[:,2],label='Soul Finger Layer I')
plt.scatter(lsoulI[:,0],lsoulI[:,2])
plt.plot(lsoulII[:,0],lsoulII[:,2],label='Soul Finger Layer II')
plt.scatter(lsoulII[:,0],lsoulII[:,2])
plt.xlabel('Rate (bits per sample)')
plt.ylabel('SNR (dB)')
plt.grid()
plt.legend()

#%%
import scipy.io.wavfile as wav
import scipy.signal as scisig

#%%
sampleRate, cupid_source=wav.read('data/layers/cupid_source.wav')
sampleRate, cupid_recons_I_032=wav.read('data/layers/cupid_recons_I_032.wav')
sampleRate, cupid_recons_II_032=wav.read('data/layers/cupid_recons_II_032.wav')
sampleRate, cupid_recons_I_064=wav.read('data/layers/cupid_recons_I_064.wav')
sampleRate, cupid_recons_II_064=wav.read('data/layers/cupid_recons_II_064.wav')
sampleRate, cupid_recons_I_096=wav.read('data/layers/cupid_recons_I_096.wav')
sampleRate, cupid_recons_II_096=wav.read('data/layers/cupid_recons_II_096.wav')
sampleRate, cupid_recons_I_128=wav.read('data/layers/cupid_recons_I_128.wav')
sampleRate, cupid_recons_II_128=wav.read('data/layers/cupid_recons_II_128.wav')
sampleRate, cupid_recons_I_192=wav.read('data/layers/cupid_recons_I_192.wav')
sampleRate, cupid_recons_II_192=wav.read('data/layers/cupid_recons_I_192.wav')
sampleRate, cupid_recons_I_256=wav.read('data/layers/cupid_recons_I_256.wav')
sampleRate, cupid_recons_II_256=wav.read('data/layers/cupid_recons_II_256.wav')

f, Py_c_I_032 = scisig.welch(cupid_recons_I_032, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_II_032 = scisig.welch(cupid_recons_II_032, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_I_064 = scisig.welch(cupid_recons_I_064, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_II_064 = scisig.welch(cupid_recons_II_064, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_I_096 = scisig.welch(cupid_recons_I_096, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_II_096 = scisig.welch(cupid_recons_II_096, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_I_128 = scisig.welch(cupid_recons_I_128, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_II_128 = scisig.welch(cupid_recons_II_128, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_I_192 = scisig.welch(cupid_recons_I_192, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_II_192 = scisig.welch(cupid_recons_II_192, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_I_256 = scisig.welch(cupid_recons_I_256, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_c_II_256 = scisig.welch(cupid_recons_II_256, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)


f, Px  = scisig.welch(cupid_source      , fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
# spectral comparison
plt.figure()
plt.plot(f,10*np.log10(Py_c_I_032/Px),label='c_I_032')
plt.plot(f,10*np.log10(Py_c_II_032/Px),label='c_II_032')
# plt.plot(f,10*np.log10(Py_c_I_064/Px),label='c_I_064')
# plt.plot(f,10*np.log10(Py_c_II_064/Px),label='c_II_064')
# plt.plot(f,10*np.log10(Py_c_I_096/Px),label='c_I_096')
# plt.plot(f,10*np.log10(Py_c_II_096/Px),label='c_II_096')
plt.plot(f,10*np.log10(Py_c_I_128/Px),label='c_I_128')
plt.plot(f,10*np.log10(Py_c_II_128/Px),label='c_II_128')
# plt.plot(f,10*np.log10(Py_c_I_192/Px),label='c_I_192')
# plt.plot(f,10*np.log10(Py_c_II_192/Px),label='c_II_192')
plt.plot(f,10*np.log10(Py_c_I_256/Px),label='c_I_256')
plt.plot(f,10*np.log10(Py_c_II_256/Px),label='c_II_256')
plt.xscale('log')
plt.grid()
plt.xlim((10, 20000))
#plt.ylim((-4,4))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral deviation (dB)')
plt.legend()

#%%
sampleRate, diner_source=wav.read('data/layers/diner_source.wav')
sampleRate, diner_recons_I_032=wav.read('data/layers/diner_recons_I_032.wav')
sampleRate, diner_recons_II_032=wav.read('data/layers/diner_recons_II_032.wav')
sampleRate, diner_recons_I_064=wav.read('data/layers/diner_recons_I_064.wav')
sampleRate, diner_recons_II_064=wav.read('data/layers/diner_recons_II_064.wav')
sampleRate, diner_recons_I_096=wav.read('data/layers/diner_recons_I_096.wav')
sampleRate, diner_recons_II_096=wav.read('data/layers/diner_recons_II_096.wav')
sampleRate, diner_recons_I_128=wav.read('data/layers/diner_recons_I_128.wav')
sampleRate, diner_recons_II_128=wav.read('data/layers/diner_recons_II_128.wav')
sampleRate, diner_recons_I_192=wav.read('data/layers/diner_recons_I_192.wav')
sampleRate, diner_recons_II_192=wav.read('data/layers/diner_recons_I_192.wav')
sampleRate, diner_recons_I_256=wav.read('data/layers/diner_recons_I_256.wav')
sampleRate, diner_recons_II_256=wav.read('data/layers/diner_recons_II_256.wav')

f, Py_d_I_032 = scisig.welch(diner_recons_I_032, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_II_032 = scisig.welch(diner_recons_II_032, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_I_064 = scisig.welch(diner_recons_I_064, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_II_064 = scisig.welch(diner_recons_II_064, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_I_096 = scisig.welch(diner_recons_I_096, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_II_096 = scisig.welch(diner_recons_II_096, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_I_128 = scisig.welch(diner_recons_I_128, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_II_128 = scisig.welch(diner_recons_II_128, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_I_192 = scisig.welch(diner_recons_I_192, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_II_192 = scisig.welch(diner_recons_II_192, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_I_256 = scisig.welch(diner_recons_I_256, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_d_II_256 = scisig.welch(diner_recons_II_256, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)


f, Px  = scisig.welch(diner_source      , fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
# spectral comparison
plt.figure()
plt.plot(f,10*np.log10(Py_d_I_032/Px),label='d_I_032')
plt.plot(f,10*np.log10(Py_d_II_032/Px),label='d_II_032')
# plt.plot(f,10*np.log10(Py_d_I_064/Px),label='d_I_064')
# plt.plot(f,10*np.log10(Py_d_II_064/Px),label='d_II_064')
# plt.plot(f,10*np.log10(Py_d_I_096/Px),label='d_I_096')
# plt.plot(f,10*np.log10(Py_d_II_096/Px),label='d_II_096')
plt.plot(f,10*np.log10(Py_d_I_128/Px),label='d_I_128')
plt.plot(f,10*np.log10(Py_d_II_128/Px),label='d_II_128')
# plt.plot(f,10*np.log10(Py_d_I_192/Px),label='d_I_192')
# plt.plot(f,10*np.log10(Py_d_II_192/Px),label='d_II_192')
plt.plot(f,10*np.log10(Py_d_I_256/Px),label='d_I_256')
plt.plot(f,10*np.log10(Py_d_II_256/Px),label='d_II_256')
plt.xscale('log')
plt.grid()
plt.xlim((10, 20000))
#plt.ylim((-4,4))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral deviation (dB)')
plt.legend()

#%%
sampleRate, soul_source=wav.read('data/layers/soul_source.wav')
#sampleRate, soul_recons_I_032=wav.read('data/layers/soul_recons_I_032.wav')
sampleRate, soul_recons_II_032=wav.read('data/layers/soul_recons_II_032.wav')
sampleRate, soul_recons_I_064=wav.read('data/layers/soul_recons_I_064.wav')
sampleRate, soul_recons_II_064=wav.read('data/layers/soul_recons_II_064.wav')
sampleRate, soul_recons_I_096=wav.read('data/layers/soul_recons_I_096.wav')
sampleRate, soul_recons_II_096=wav.read('data/layers/soul_recons_II_096.wav')
sampleRate, soul_recons_I_128=wav.read('data/layers/soul_recons_I_128.wav')
sampleRate, soul_recons_II_128=wav.read('data/layers/soul_recons_II_128.wav')
sampleRate, soul_recons_I_192=wav.read('data/layers/soul_recons_I_192.wav')
sampleRate, soul_recons_II_192=wav.read('data/layers/soul_recons_I_192.wav')
sampleRate, soul_recons_I_256=wav.read('data/layers/soul_recons_I_256.wav')
sampleRate, soul_recons_II_256=wav.read('data/layers/soul_recons_II_256.wav')

#f, Py_s_I_032 = scisig.welch(soul_recons_I_032, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_II_032 = scisig.welch(soul_recons_II_032, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_I_064 = scisig.welch(soul_recons_I_064, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_II_064 = scisig.welch(soul_recons_II_064, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_I_096 = scisig.welch(soul_recons_I_096, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_II_096 = scisig.welch(soul_recons_II_096, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_I_128 = scisig.welch(soul_recons_I_128, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_II_128 = scisig.welch(soul_recons_II_128, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_I_192 = scisig.welch(soul_recons_I_192, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_II_192 = scisig.welch(soul_recons_II_192, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_I_256 = scisig.welch(soul_recons_I_256, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_s_II_256 = scisig.welch(soul_recons_II_256, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)


f, Px  = scisig.welch(soul_source      , fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
# spectral comparison
plt.figure()

# plt.plot(f,10*np.log10(Py_s_II_032/Px),label='s_II_032')
plt.plot(f,10*np.log10(Py_s_I_064/Px),label='s_I_064')
plt.plot(f,10*np.log10(Py_s_II_064/Px),label='s_II_064')
# plt.plot(f,10*np.log10(Py_s_I_096/Px),label='s_I_096')
# plt.plot(f,10*np.log10(Py_s_II_096/Px),label='s_II_096')
plt.plot(f,10*np.log10(Py_s_I_128/Px),label='s_I_128')
plt.plot(f,10*np.log10(Py_s_II_128/Px),label='s_II_128')
# plt.plot(f,10*np.log10(Py_s_I_192/Px),label='s_I_192')
# plt.plot(f,10*np.log10(Py_s_II_192/Px),label='s_II_192')
plt.plot(f,10*np.log10(Py_s_I_256/Px),label='s_I_256')
plt.plot(f,10*np.log10(Py_s_II_256/Px),label='s_II_256')
plt.xscale('log')
plt.grid()
plt.xlim((10, 20000))
#plt.ylim((-4,4))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral deviation (dB)')
plt.legend()

#%%

sampleRate, nara_source=wav.read('data/layers/nara_source.wav')
sampleRate, nara_recons_I_032=wav.read('data/layers/nara_recons_I_032.wav')
sampleRate, nara_recons_II_032=wav.read('data/layers/nara_recons_II_032.wav')
sampleRate, nara_recons_I_064=wav.read('data/layers/nara_recons_I_064.wav')
sampleRate, nara_recons_II_064=wav.read('data/layers/nara_recons_II_064.wav')
sampleRate, nara_recons_I_096=wav.read('data/layers/nara_recons_I_096.wav')
sampleRate, nara_recons_II_096=wav.read('data/layers/nara_recons_II_096.wav')
sampleRate, nara_recons_I_128=wav.read('data/layers/nara_recons_I_128.wav')
sampleRate, nara_recons_II_128=wav.read('data/layers/nara_recons_II_128.wav')
sampleRate, nara_recons_I_192=wav.read('data/layers/nara_recons_I_192.wav')
sampleRate, nara_recons_II_192=wav.read('data/layers/nara_recons_I_192.wav')
sampleRate, nara_recons_I_256=wav.read('data/layers/nara_recons_I_256.wav')
sampleRate, nara_recons_II_256=wav.read('data/layers/nara_recons_II_256.wav')

f, Py_n_I_032 = scisig.welch(nara_recons_I_032, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_II_032 = scisig.welch(nara_recons_II_032, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_I_064 = scisig.welch(nara_recons_I_064, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_II_064 = scisig.welch(nara_recons_II_064, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_I_096 = scisig.welch(nara_recons_I_096, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_II_096 = scisig.welch(nara_recons_II_096, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_I_128 = scisig.welch(nara_recons_I_128, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_II_128 = scisig.welch(nara_recons_II_128, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_I_192 = scisig.welch(nara_recons_I_192, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_II_192 = scisig.welch(nara_recons_II_192, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_I_256 = scisig.welch(nara_recons_I_256, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
f, Py_n_II_256 = scisig.welch(nara_recons_II_256, fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)


f, Px  = scisig.welch(nara_source      , fs=sampleRate, window='hamming', nperseg=4096, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)
# spectral comparison
plt.figure()
plt.plot(f,10*np.log10(Py_n_I_032/Px),label='n_II_032')
plt.plot(f,10*np.log10(Py_n_II_032/Px),label='n_II_032')
# plt.plot(f,10*np.log10(Py_n_I_064/Px),label='n_I_064')
# plt.plot(f,10*np.log10(Py_n_II_064/Px),label='n_II_064')
# plt.plot(f,10*np.log10(Py_n_I_096/Px),label='n_I_096')
# plt.plot(f,10*np.log10(Py_n_II_096/Px),label='n_II_096')
# plt.plot(f,10*np.log10(Py_n_I_128/Px),label='n_I_128')
# plt.plot(f,10*np.log10(Py_n_II_128/Px),label='n_II_128')
# plt.plot(f,10*np.log10(Py_n_I_192/Px),label='n_I_192')
# plt.plot(f,10*np.log10(Py_n_II_192/Px),label='n_II_192')
# plt.plot(f,10*np.log10(Py_n_I_256/Px),label='n_I_256')
# plt.plot(f,10*np.log10(Py_n_II_256/Px),label='n_II_256')
plt.xscale('log')
plt.grid()
plt.xlim((10, 20000))
#plt.ylim((-4,4))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral deviation (dB)')
plt.legend()