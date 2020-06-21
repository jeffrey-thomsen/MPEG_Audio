#%% Bit allocation
"""
The bit allocation actually takes into account the output of the psychoacoustic
model, but instead I have implemented an equivalent with equivSMR.
It takes a subbandFrame object, and through an iterative process it 
assigns a number of coding bits to each subband
"""        
        
# main function for bit allocation that calculates the number of bits to be
# assigned to each subband of a frame of subband samples
def assignBits(scaleFactorVal,nTotalBits,nMiscBits,frameSize,SMR,scfTrPat):
    # Input:
    # scaleFactorVal - an array containing the scale factors for each subband
    # nTotalBits - number of bits to be allocated to this frame
      
    assert (len(scaleFactorVal)==32),"scaleFactorVal array wrong length!"
    
    nAvailableBits = nTotalBits - nMiscBits # bits available for coding samples and scalefactors
    nSplBits = 0
    nScfBits = 0
    
    nBitsSubband = np.zeros(32)
    nLevelsSubband = np.zeros(32,dtype=int)
    
    iterFlag = False
    # iterative bit allocation
    while (nAvailableBits > 12):
        
        minSubBand = determineMinimalMNR(nBitsSubband,nAvailableBits,SMR)
        
        #nBitsSubband = increaseNBits(nBitsSubband,minSubBand)
        #nScfBits, nSplBits = updateBitAllocation(nBitsSubband)
        
        nBitsSubbandCheck, nLevelsSubbandCheck, nSplBitsCheck, nScfBitsCheck = spendBit(nBitsSubband[minSubBand], nSplBits, nScfBits, frameSize, minSubBand,nLevelsSubband[minSubBand],scfTrPat)
        
        # if the bit allocation was violated, choose the band with next smallest MNR for bit spending instead
        while (nTotalBits - (nSplBitsCheck + nScfBitsCheck + nMiscBits))<0:
            SMR[minSubBand] = -np.inf
            minSubBand = determineMinimalMNR(nBitsSubband,nAvailableBits,SMR)
            nBitsSubbandCheck, nLevelsSubbandCheck, nSplBitsCheck, nScfBitsCheck = spendBit(nBitsSubband[minSubBand], nSplBits, nScfBits, frameSize, minSubBand,nLevelsSubband[minSubBand],scfTrPat)
            if (max(SMR)==-np.inf):
                nSplBitsCheck = nSplBits
                nScfBitsCheck = nScfBits
                nBitsSubbandCheck = nBitsSubband[minSubBand]
                nLevelsSubbandCheck = nLevelsSubband[minSubBand]
                iterFlag = True
       
        # after successful iteration, assign the permanent new values
        nSplBits = nSplBitsCheck
        nScfBits = nScfBitsCheck
        nBitsSubband[minSubBand] = nBitsSubbandCheck
        nLevelsSubband[minSubBand] = nLevelsSubbandCheck
        nAvailableBits = nTotalBits - (nSplBits + nScfBits + nMiscBits)
        assert (nAvailableBits>=0),"Allocated too many bits!"
        
        if iterFlag:
            break
    
    
    return nBitsSubband  
     
# compare MNRs of each subband and return first subband with lowest MNR
def determineMinimalMNR(nBitsSubband,nAvailableBits,SMR):
    assert (len(nBitsSubband)==32),"Wrong length of input list!"
    
    MNR = updateMNR(nBitsSubband,SMR)
    
    minMNRIndex = np.argmin(MNR)
    
    # exclude bands with already max number of bits allocated to them
    while minMNRIndex in np.where(nBitsSubband == maxBitsSubband)[0]:
        MNR[minMNRIndex] = np.inf
        minMNRIndex = np.argmin(MNR)

    return minMNRIndex

# calculate MNR of all subbands
def updateMNR(nBitsSubband,SMR):
    
    SNR = np.zeros(32)
    MNR = np.zeros(32)
    nBands = 32
    
    for iBand in range(nBands):
        assert (nBitsSubband[iBand]<=16),"Too many bits assigned!"
        snrIndex = np.where(snrTable[:,0] == nBitsSubband[iBand])[0][0]
        SNR[iBand] = snrTable[snrIndex,2]
        
    MNR = SNR - SMR
        
    return MNR

# increase number of allocated bits to chosen subband to next step and
# return new number of bits allocated to code subband samples and scalefactors
def spendBit(nBits,nSplBits,nScfBits,frameSize,nBand,nLevel,scfTrPat):    
    
    assert (nBits<=16),"Too many bits assigned!"
    
    nLevel += 1
    nextLevel = quantLevels[nBand,nLevel]
    assert (nextLevel!=-np.inf), "Bit spending error!"
    
    dBits = snrTable[snrTable[:,1]==nextLevel,0] - nBits
    
    if (nBits==0)&(dBits>0):
        nScfBits += 2 + 6*len(scfTrPat[nBand])

    nBits += dBits 
    
    nSplBits += dBits*frameSize
    
    return nBits, nLevel, nSplBits, nScfBits

# calculate SMR equivalent from scalefactor
def equivSMR(scaleFactorVal):
    
    equivSMR = 20 * np.log10(scaleFactorVal * 32768) - 10
    
    return equivSMR