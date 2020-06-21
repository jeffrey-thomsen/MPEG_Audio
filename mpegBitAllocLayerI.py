#%% Bit allocation
"""
The bit allocation actually takes into account the output of the psychoacoustic
model, but instead I have implemented an equivalent with equivSMR.
It takes a subbandFrame object, and through an iterative process it 
assigns a number of coding bits to each subband
"""

# main function for bit allocation that calculates the number of bits to be
# assigned to each subband of a frame of subband samples
def assignBits(scaleFactorVal,nTotalBits,nMiscBits,frameSize,SMR):
    # Input:
    # scaleFactorVal - an array containing the scale factors for each subband
    # nTotalBits - number of bits to be allocated to this frame
      
    assert (len(scaleFactorVal)==32),"scaleFactorVal array wrong length!"
    
    nAvailableBits = nTotalBits - nMiscBits # bits available for coding samples and scalefactors
    nSplBits = 0
    nScfBits = 0
    
    nBitsSubband = np.zeros(32,dtype=int)
    
    
    # iterative bit allocation
    while (nAvailableBits >= possibleIncreaseInBits(nBitsSubband)):
        
        minSubBand = determineMinimalMNR(nBitsSubband,nAvailableBits,SMR)
        
        #nBitsSubband = increaseNBits(nBitsSubband,minSubBand)
        #nScfBits, nSplBits = updateBitAllocation(nBitsSubband)
        
        nBitsSubband[minSubBand], nSplBits, nScfBits = spendBit(nBitsSubband[minSubBand], nSplBits, nScfBits, frameSize)
        
        nAvailableBits = nTotalBits - (nSplBits + nScfBits + nMiscBits)
        assert (nAvailableBits>=0),"Allocated too many bits!"
    
    
    return nBitsSubband

# given the current state, return the smallest increase in bits allocated
# in a potential next iteration step
def possibleIncreaseInBits(nBitsSubband):
    assert (len(nBitsSubband)==32),"Wrong length of input list!"
    assert (np.max(nBitsSubband)<16),"Too many bits allocated!"
    
    if (np.min(nBitsSubband)==0):
        minIncrease = 30
    elif (0<np.min(nBitsSubband)<15):
        minIncrease = 12
    else:
        assert(np.min(nBitsSubband)==15),"possibleIncreaseInBits loop error!"
        assert(np.max(nBitsSubband)==15),"possibleIncreaseInBits loop error!"
        minIncrease = np.inf # to break the while loop when no more bits can be added

    return minIncrease

# compare MNRs of each subband and return first subband with lowest MNR
def determineMinimalMNR(nBitsSubband,nAvailableBits,SMR):
    assert (len(nBitsSubband)==32),"Wrong length of input list!"
    
    MNR = updateMNR(nBitsSubband,SMR)
    
    minMNRIndex = np.argmin(MNR)
    
    # exclude bands with already 15 bits allocated to them
    while minMNRIndex in np.where(nBitsSubband == 15)[0]:
        MNR[minMNRIndex] = np.inf
        minMNRIndex = np.argmin(MNR)
    
    # exclude bands with only 0 bits allocated to them, because they would need
    # 30 bits instead of just 12 to increase the allocation
    if nAvailableBits<30:
        while minMNRIndex in np.where(nBitsSubband == 0)[0]:
            MNR[minMNRIndex] = np.inf
            minMNRIndex = np.argmin(MNR)
    

    return minMNRIndex

# calculate MNR of all subbands
def updateMNR(nBitsSubband,SMR):
    
    SNR = np.zeros(32)
    MNR = np.zeros(32)
    nBands = 32
    
    for iBand in range(nBands):
        assert (nBitsSubband[iBand]<16),"Too many bits assigned!"
        snrIndex = np.where(snrTable[:,0] == nBitsSubband[iBand])[0][0]
        SNR[iBand] = snrTable[snrIndex,2]
        
    MNR = SNR - SMR
        
    return MNR

# calculate SMR equivalent from scalefactor
def equivSMR(scaleFactorVal):
    
    equivSMR = 20 * np.log10(scaleFactorVal * 32768) - 10
    
    return equivSMR

# increase number of allocated bits to chosen subband to next step and
# return new number of bits allocated to code subband samples and scalefactors
def spendBit(nBits,nSplBits,nScfBits,frameSize):    
    
    assert (nBits<16),"Too many bits assigned!"
    
    if (0<nBits<15):
        nBits += 1
        nSplBits += frameSize
    elif (nBits==0):
        nBits += 2
        nScfBits += 6
        nSplBits += 2*frameSize
    else:
        nBits += 0
    
    return nBits, nSplBits, nScfBits