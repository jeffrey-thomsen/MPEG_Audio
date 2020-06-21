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

def scalefactorClass(dscf):
    # determine proximity of two adjacent scalefactors in a Layer II subband
    # frame by dividing the index distancs dscf into classes
    
    if (dscf<=-3):
        scfClass = 1
    elif (-3<dscf<0):
        scfClass = 2
    elif (dscf==0):
        scfClass = 3
    elif (0<dscf<3):
        scfClass = 4
    else:
        scfClass = 5    
        
    return scfClass

def transmissionPatternsLayerII(scfClass1,scfClass2):
    # using Table C.4 of the MPEG standard, determine how the three
    # scalefactors of a subband in a Layer II subband frame schould be coded
    # input:
    # scfClass1/2 : proximity indicators of the 1st and 2nd, and 2nd and 3rd
    # scalefactor respectively
    
    line = ((C4[:,0]==scfClass1)&(C4[:,1]==scfClass2))
    
    scfUsed = np.array(C4[line,2][0])
    transmissionPattern = np.array(list(C4[line,3][0]))
    selectionInfo = C4[line,4][0]
    
    return scfUsed, transmissionPattern, selectionInfo

def codeScfLayerII(scfInd):
    # compare the three scalefactors per band of a Layer II subband frame and
    # apply smarter coding to save bits
    # input:
    # scfInd - array of size 32x3 containing the scalefactor
    # indices of one Layer II subband frame of 36 subband samples
    
    scfUsed = np.zeros(32)
    trPat = np.zeros(32)
    selInfo = np.zeros(32)
    
    for iBand in range(32):
    
        dscf1 = scfInd[iBand,0]-scfInd[iBand,1]
        dscf2 = scfInd[iBand,1]-scfInd[iBand,2]
    
        scfClass1 = scalefactorClass(dscf1)
        scfClass2 = scalefactorClass(dscf2)
    
        scfUsed[iBand], trPat[iBand], selInfo[iBand] = transmissionPatternsLayerII(scfClass1,scfClass2)

    return scfUsed, trPat, selInfo

def mapScfLayerII(scfUsed,scfInd):
    
    for iBand in range(32):
        if scfUsed[0] != 4:
            scfInd[iBand,:] = scfInd[iBand,scfUsed[iBand]-1]
        else:
            maxInd = np.argmin(scfInd[iBand,:]) # find maximum scalefactor
            scfInd[iBand,:] = scfInd[iBand,maxInd]
            
        scfVal = scaleFactorTable[scfInd]
    
    return scfInd, scfVal

def encoderLayerII(subSamples,nTotalBits,x,sampleRate,smrModel,Aweighting):
    # Input:
    # subSamples - array or list of subband samples calculated from the
    #              analysis polyphase filterbank
    # nTotalBits - the bitrate defined in terms of bits available per frame
    # Output:
    # transmitFrames - list of transmitFrames objects, the simulated bitstream
    
    
    # initialize and push subband samples into a subbandFrame object
    subFrame = subbandFrame(layer=2)
    
    transmitFrames=[]
    
    nFrames = int(subSamples.shape[1]/subFrame.nSamples)
    iSubSample = 0
    iAudioSample = 0
    
    
    # preallocate for bit allocation
    nHeaderBits   =   32 # bits needed for header
    nCrcBits      =    0 # CRC checkword, 16 if used
    nBitAllocBits =    sumNBal # bit allocation is more complex in Layer II and depends on the bit allocation tables defined in Tables B.2
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
        # push next 36 subband samples into the subbandFrame object
        subFrame.pushFrame(subSamples[:,iSubSample:iSubSample+36])
        iSubSample+=36
        
        # calculate scalefactors for current frame
        scaleFactorVal = np.zeros((32,3))
        scaleFactorInd = np.zeros((32,3))
        for iScf in range(3):
            scaleFactorVal[:,iScf], scaleFactorInd[:,iScf] = calcScaleFactors(subFrame.frame[:,iScf*12:iScf*12+12])
            
        scfUsed, scfTransmissionPattern, scfSelectionInfo = codeScfLayerII(scaleFactorInd)
        scaleFactorVal, scaleFactorInd = mapScfLayerII(scfUsed,scalefactorInd)


        # SMR calculation
        if smrModel == 'psy':
            # grab next 512/1024 audio samples for psychoacoustic model
            iPsyModSample = iAudioSample-256+64
            psyModFrame = x[iPsyModSample:iPsyModSample+1024]
            iAudioSample += 32*subFrame.nSamples
            if len(psyModFrame)==1024:
                SMR = pam.PsyMod(psyModFrame,scaleFactorVal,subFrame.layer,sampleRate,bitrate)
            else:
                SMR = np.zeros(32)
        elif smrModel == 'scf':
            SMR = equivSMR(np.max(scaleFactorVal,axis=1)) + pqmfAweight
        elif smrModel == 'spl':
            SMR = 76 + 10 * np.log10( np.sum( subFrame.frame**2 ,axis=1)) + pqmfAweight
        

        
        # bit allocation for current frame
        #start = time.time()        
        nBitsSubband = assignBits(scaleFactorVal,nTotalBits,nMiscBits,subFrame.nSamples,SMR,scfTransmissionPattern)
        #end = time.time()
        #print("bit allocation in")
        #print(end - start)
    
    
        # quantize subband samples of current frame and store in transmitFrame object
        #start = time.time()
        transmit = quantizeSubbandFrame(subFrame,scaleFactorInd,nBitsSubband)
        
        """
        Add scfTransmissionPattern and scfSelectionInfo to TransmitFrame
        """
        
        transmitFrames.append(transmit)
        #end = time.time()
        #print("quantization in")
        #print(end - start)
        
    return transmitFrames



