#%% Encoder

import numpy as np

import mpegAudioPsyAcMod as pam

"""
The encoder stage combines the scale factor calculation, bit allocation and 
quantization of the previously calculated subband samples by continously 
pushing new subband samples into a subbandFrame object and performing the 
operations. It outputs a list of transmitFrame objects which can then be read
by the decoder.
"""

def encoderLayerI(subSamples,nTotalBits,x,sampleRate,smrModel,Aweighting):
    # Input:
    # subSamples - array or list of subband samples calculated from the
    #              analysis polyphase filterbank
    # nTotalBits - the bitrate defined in terms of bits available per frame
    # Output:
    # transmitFrames - list of transmitFrames objects, the simulated bitstream
    
    
    # initialize and push subband samples into a subbandFrame object
    subFrame = subbandFrame(layer=1)
    
    transmitFrames=[]
    
    nFrames = int(subSamples.shape[1]/subFrame.nSamples)
    iSubSample = 0
    iAudioSample = 0
    
    
    # preallocate for bit allocation
    nHeaderBits   =   32 # bits needed for header
    nCrcBits      =    0 # CRC checkword, 16 if used
    nBitAllocBits =  128 # bit allocation -> codes 4bit int values 0...15 for each of 32 subbands
    nAncBits      =    0 # bits needed for ancillary data
    
    nMiscBits = nHeaderBits + nCrcBits + nBitAllocBits +  nAncBits
    
    
    
    
    
    if Aweighting:
        pqmfFreq = sampleRate/64 *(np.arange(0,32)+0.5)        
        Afreq=[10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 
                   250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 
                   3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
        Aweight=[-70.4,0 -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, 
                  -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, 
                  -0.8, 0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5, 
                  -4.3, -6.6, -9.3]     
        pqmfAweight = np.interp(pqmfFreq,Afreq,Aweight)
        pqmfAweight[0] = 0
    else:
        pqmfAweight = np.zeros(32)
    
    
    for iFrame in range(nFrames):

        # push next 12/36 subband samples into the subbandFrame object
        
        subFrame.pushFrame(subSamples[:,iSubSample:iSubSample+12])
        iSubSample += 12

        
        # calculate scalefactors for current frame
        
        #start = time.time()
        scaleFactorVal, scaleFactorInd = calcScaleFactors(subFrame)
        #end = time.time()
        #print("scalefactor calculation in")
        #print(end - start)
          
        # SMR calculation
        if smrModel == 'psy':
            # grab next 512/1024 audio samples for psychoacoustic model
            iPsyModSample = iAudioSample-256-64
            psyModFrame = x[iPsyModSample:iPsyModSample+512]
            iAudioSample += 32*subFrame.nSamples
            if len(psyModFrame)==512:
                SMR = pam.PsyMod(psyModFrame,scaleFactorVal,subFrame.layer,sampleRate,bitrate)
            else:
                SMR = np.zeros(32)
        elif smrModel == 'scf':
            SMR = equivSMR(scaleFactorVal) + pqmfAweight
        elif smrModel == 'spl':
            SMR = 76 + 10 * np.log10( np.sum( subFrame.frame**2 ,axis=1)) + pqmfAweight
        
        # bit allocation for current frame
        
        #start = time.time()        
        nBitsSubband = assignBits(scaleFactorVal,nTotalBits,nMiscBits,subFrame.nSamples,SMR)
        #end = time.time()
        #print("bit allocation in")
        #print(end - start)
    
    
        # quantize subband samples of current frame and store in transmitFrame object
        
        #start = time.time()
        transmit = quantizeSubbandFrame(subFrame,scaleFactorInd,nBitsSubband)
        transmitFrames.append(transmit)
        #end = time.time()
        #print("quantization in")
        #print(end - start)
        
    return transmitFrames
