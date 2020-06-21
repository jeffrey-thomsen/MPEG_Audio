# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:11:24 2020

@author: Jeffrey
"""
import numpy as np

import mpegAudioPsyAcMod as pam



# to time stuff:
#import time
# start = time.time()
# #expression
# end = time.time()
# print("Expression calculated in")
# print(end - start)
        
#%% Load windows and look-up-tables defined in MPEG standard

global layer
global bitrate
global sampleRate
file = open("layer", "rb")
layer = np.load(file)
file = open("bitrate", "rb")
bitrate = np.load(file)
file = open("sampleRate", "rb")
sampleRate = np.load(file)


global C
global M
global scaleFactorTable
global snrTable
global A
global B
global nQuantSteps
global D
global N

C = np.load('data/mpeg_analysis_window.npy') # analysis window defined by MPEG
M = np.load('data/mpeg_polyphase_analysis_matrix_coeff.npy') # analysis filterbank matrix pre-calculated
scaleFactorTable = np.load('data/mpeg_scale_factors.npy') # scale factors defined by MPEG
D = np.load('data/mpeg_synthesis_window.npy') # synthesis window defined by MPEG
N = np.load('data/mpeg_polyphase_synthesis_matrix_coeff.npy') # synthesis filterbank matrix pre-calculated

if layer == 1:
    snrTable = np.load('data/mpeg_snr_layer_i.npy') # SNR levels defined by MPEG
    A = np.load('data/mpeg_qc_layer_i_A.npy') # quantization coefficients defined by MPEG
    B = np.load('data/mpeg_qc_layer_i_B.npy') # quantization coefficients defined by MPEG
    nQuantSteps = np.load('data/mpeg_qc_layer_i_nSteps.npy') # number of quantization steps defined by MPEG
    
elif layer == 2:
    import B2Loader as b2

    snrTable = np.load('data/mpeg_snr_layer_ii.npy') # SNR levels defined by MPEG
    A = np.load('data/mpeg_qc_layer_ii_A.npy') # quantization coefficients defined by MPEG
    B = np.load('data/mpeg_qc_layer_ii_B.npy') # quantization coefficients defined by MPEG
    nQuantSteps = np.load('data/mpeg_qc_layer_ii_nSteps.npy') # number of quantization steps defined by MPEG
    global C4
    C4 = np.load('data/C4.npy',allow_pickle=True) # Layer II scalefactor transmission patterns
    global nbal
    global quantLevels
    global sbLimit
    global sumNbal
    global maxBitsSubband
    nbal, quantLevels, sbLimit, sumNbal, maxBitsSubband = b2.B2Loader(sampleRate,bitrate)
#%% Implementation of the Analysis Polyphase filterbank
"""
The Analysis Filterbank uses a buffer which is fed with blocks of input audio
signal (analysisBuffer object). For each block of 32 audio samples pushed into
 the buffer, the polyphase filterbank calculates one subband sample in 32 
subbands
"""

# object containing 512 samples of audio to be fed to the Polyphase Filterbank
class analysisBuffer:
    def __init__(self,bufferVal=np.zeros(512)):
        assert (len(bufferVal)==512),"Input not length 512!"
        self.bufferVal = bufferVal
    
    # 32 samples are pushed into the buffer at a time
    def pushBlock(self,sampleBlock):
        assert (len(sampleBlock)==32),"Sample block length not 32!"
        self.bufferVal[32:] = self.bufferVal[:-32]
        self.bufferVal[0:32] = sampleBlock[::-1]

# analysis filterbank for MPEG Audio subband coding
def polyphaseFilterbank(x):
    # x - analysisBuffer object
    assert (type(x)==analysisBuffer),"Input not an analysisBuffer object!"
    
    

    Z = C*x.bufferVal
    
    Y = np.zeros(64)
    for k in range(64):
        for m in range(8):
            Y[k] = Y[k] + Z[k+64*m]
    
    subbandSamples = np.zeros(32)
    for n in range(32):
        subbandSamples[n] = np.sum(M[n,:] * Y)

    return subbandSamples

def mFun(n,k):
    M = np.cos((2*n+1)*(k-16)*np.pi/64)
    
    return M

#%% 
"""
Routine that takes blocks of audio and feeds them to the buffer
"""

# takes raw audio signal, runs it into the analysisBuffer and calculates the 
# filterbank outputs subbandSamples
def feedCoder(x):
    # Input:
    # x - mono or stereo PCM audio signal samples as array
    # Output:
    # subbandSamples - array of size 32xnBlocks, each column containing a
    #                  subband sample representing 32 input samples
    
    nSamples,stereo=x.shape
    assert (stereo==0 or stereo==1 or stereo==2),"Input not mono or stereo!"
    
    if stereo==1 or stereo==2:
        x  = x[:,0] # eventually will need to be xLeft/xRight and will need a whole routine to handle both channels
    
    # zero-pad to divisible of 32x36 = 1152 samples, to yield a set of
    # completely filled subband frames, both in Layer I and II
    modulo1152 = nSamples%1152
    if modulo1152!=0:
        nPadding = 1152-modulo1152
    else:
        nPadding = 0
    x = np.concatenate((x,np.zeros(nPadding)))
    nSamples = nSamples+nPadding
    assert (nSamples%32==0),"Zero-padding mistake!"
    nBlocks = int(nSamples/32) 
    
    # initialization
    xBuffer = analysisBuffer()
    iSample = 0
    subbandSamples = np.zeros((32,nBlocks))
    
    for iBlock in range(nBlocks):
        xBuffer.pushBlock(x[iSample:iSample+32])
        subbandSamples[:,iBlock] = polyphaseFilterbank(xBuffer)
        iSample += 32  
    
    
    
    return subbandSamples


#%% subbandFrame object

# object containing a frame of 12 or 36 subband sample outputs of the polyphase filterbank, must be initialized empty first
class subbandFrame:
    def __init__(self,layer=1):
        assert (layer==1 or layer==2 or layer==3),"Encoding layer type not 1, 2 or 3!"
        self.layer=layer
        if self.layer==1:
            self.nSamples = 12
        elif self.layer==2 or self.layer==3:
            self.nSamples = 36
        
        self.frame = np.zeros([32,self.nSamples])
        
    def pushFrame(self,subbandSamples):
        assert (subbandSamples.shape[1]>=self.nSamples),"Not enough entries in subbandSamples array!"
        for i in range(self.nSamples):
            self.frame[:,i]=(subbandSamples[:,i])


#%% Scalefactor Calculation
"""
The scalefactor calculation takes a subbandFrame object, compares the max abs 
values of each subband to the scalefactor table and assigns the fitting scale 
factors to the subbands
"""
# compare max abs values of each subband frame to scalefactor table and deduct
# fitting scale factor
def calcScaleFactors(subbandSamples):
    # Input:
    # subbandFrame - a subbandFrame object containing subband samples of one
    #                frame, i.e. 32 x 12 or 32 x 36 samples
    # Output:
    # scaleFactor - array of lergth 32 containing the numerical scale factor
    #               values for each band
    # scaleFactorIndex - array of length 32 containing the indices pointing to
    #                    the scale factors stored in the table
    
    assert (subbandSamples.shape==(32, 12)),"Wrong subbandFrame array dimensions!"
    subbandMaxVals = np.amax(np.abs(subbandSamples),axis=1)
    assert ((subbandMaxVals<2.0).all()), "Maximum subband value larger than 2!"

    
    
    flippedScaleFactorTable = np.flip(scaleFactorTable) # needed for finding first value larger than sample value
    assert (len(flippedScaleFactorTable)==63),"Table length not 63!"
    
    scaleFactor = np.zeros(32)
    scaleFactorIndex = np.zeros(32,dtype=np.int8)
    for iCompare in range(32):
        scaleFactor[iCompare] = flippedScaleFactorTable[np.argmax(flippedScaleFactorTable>subbandMaxVals[iCompare])]
        scaleFactorIndex[iCompare] = (62-np.argmax(flippedScaleFactorTable>subbandMaxVals[iCompare]))
        assert ((scaleFactorIndex[iCompare])==(np.where(scaleFactorTable==scaleFactor[iCompare])[0][0])),"calcScf error!"
    return scaleFactor, scaleFactorIndex

#%% Bit allocation
"""
The bit allocation actually takes into account the output of the psychoacoustic
model, but instead I have implemented an equivalent with equivSMR.
It takes a subbandFrame object, and through an iterative process it 
assigns a number of coding bits to each subband
"""

# compare MNRs of each subband and return first subband with lowest MNR
def determineMinimalMNR(nBitsSubband,nAvailableBits,SMR):
    assert (len(nBitsSubband)==32),"Wrong length of input list!"
    
    MNR = updateMNR(nBitsSubband,SMR)
    
    minMNRIndex = np.argmin(MNR)
    
    if layer==1:
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
    elif layer==2:
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
        if layer==1:
            assert (nBitsSubband[iBand]<16),"Too many bits assigned!"
        elif layer==2:
            assert (nBitsSubband[iBand]<=16),"Too many bits assigned!"
        snrIndex = np.where(snrTable[:,0] == nBitsSubband[iBand])[0][0]
        SNR[iBand] = snrTable[snrIndex,2]
        
    MNR = SNR - SMR
        
    return MNR

# calculate SMR equivalent from scalefactor
def equivSMR(scaleFactorVal):
    
    equivSMR = 20 * np.log10(scaleFactorVal * 32768) - 10
    
    return equivSMR

if layer==1:
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




elif layer==2:
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
            iterCount = 0
            while (nTotalBits - (nSplBitsCheck + nScfBitsCheck + nMiscBits))<0:
                iterCount +=1
                SMR[minSubBand] = -np.inf
                minSubBand = determineMinimalMNR(nBitsSubband,nAvailableBits,SMR)
                
                nBitsSubbandCheck, nLevelsSubbandCheck, nSplBitsCheck, nScfBitsCheck = spendBit(nBitsSubband[minSubBand], nSplBits, nScfBits, frameSize, minSubBand,nLevelsSubband[minSubBand],scfTrPat)
                
                if (max(SMR)==-np.inf):
                    nSplBitsCheck = nSplBits
                    nScfBitsCheck = nScfBits
                    nBitsSubbandCheck = nBitsSubband[minSubBand]
                    nLevelsSubbandCheck = nLevelsSubband[minSubBand]
                    iterFlag = True
                if iterCount>32:
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
        
        nSplBits += int(dBits*frameSize)
        
        return nBits, nLevel, nSplBits, nScfBits

#%% Quantizer

"""
The quantizer takes a subbandFrame object, its corresponding scalefactors and 
bit assignments, divides each sample in the frame with the corresponding 
scalefactor of the band and applies the quantization according to the number of
 bits allocated to each band. The quantized subband samples are then wrapped 
in a transmitFrame object together with the other input data, quasi 
representing a block of the formatted bitstream.

"""
class transmitFrame:
    def __init__(self,nSubbands=None,nBitsSubband=None,scalefactorInd=None,quantSubbandSamples=None,scfTransmissionPattern=None,scfSelectionInfo=None):
        assert (len(nSubbands)==len(scalefactorInd)==len(quantSubbandSamples)), "Length of input lists not consistent!"
        self.nSubbands = nSubbands
        self.nBitsSubband = nBitsSubband
        self.scalefactorInd = scalefactorInd
        self.quantSubbandSamples = quantSubbandSamples
        self.scfTransmissionPattern = scfTransmissionPattern
        self.scfSelectionInfo = scfSelectionInfo

# Uniform mid-tread quantizer defined by MPEG
def subbandQuantizer(normalizedBand,nBits):
    if layer==1:
        indBits = nBits-2
    elif layer==2:
        indBits = [np.where(snrTable[:,0] == nBits)[0][0]-1]
    
    quantizedBand = A[indBits]*normalizedBand+B[indBits]

    # assign each quantized value to a fixed value dependent on nBits
    assignedVals = np.arange(-1,1+1e-12,2/(nQuantSteps[indBits]-1))
    threshVals   = np.arange(-1,A[indBits]+B[indBits],-2*B[indBits])-2*B[indBits]
    
    for iSample in range(len(quantizedBand)):
        assert(-1<=quantizedBand[iSample]<=A[indBits]+B[indBits]),"Quantized value out of bounds!"
        quantizedBand[iSample]=assignedVals[np.argmax(quantizedBand[iSample]<=threshVals)]
        
    return quantizedBand
            
if layer==1:    #

            
    
    # multiply subband frame by scalefactors and qua0,,ntize them with the number of bits
    def quantizeSubbandFrame(subFrame,scaleFactorInd,nBitsSubband):
        # Input:
        # subFrame - a subbandFrame object containing 12/36 output samples of the
        #                polyphase filterbank
        # scaleFactorInd - array of length 32 containing the indices pointing to
        #                  the respective scale factors for each band
        # nBitsSubband - array of length 32 determining the bit allocation for
        #                each band
        # Output:
        # transmit - transmitFrame object containing the indices of subbands to be
        #            coded, the number of bits allocated to each band, the indices
        #            of scale factors for every coded band and the quantized
        #            samples for every coded band
        
        assert (type(subFrame)==subbandFrame),"Input not a subbandFrame object!"
        
        transmitScalefactorInd = []
        transmitSubband = []
        transmitNSubbands = []
        
        
        for iBand in range(32):
            if nBitsSubband[iBand]>0:
                transmitNSubbands.append(iBand)
                transmitScalefactorInd.append(scaleFactorInd[iBand])
                
                normalizedBand = subFrame.frame[iBand,:]/scaleFactorTable[scaleFactorInd[iBand]]
                assert((normalizedBand<1).all())
                quantizedBand = subbandQuantizer(normalizedBand,nBitsSubband[iBand])
                
                
                transmitSubband.append(quantizedBand)
                
        transmitNSubbands = np.array(transmitNSubbands,dtype=int)
        transmitScalefactorInd = np.array(transmitScalefactorInd,dtype=int)
        transmitSubband = np.array(transmitSubband)
        
        transmit=transmitFrame(transmitNSubbands,nBitsSubband,transmitScalefactorInd,transmitSubband)
                
        return transmit


elif layer==2:              
    # multiply subband frame by scalefactors and qua0,,ntize them with the number of bits
    def quantizeSubbandFrame(subFrame,scaleFactorInd,nBitsSubband,scfTransmissionPattern,scfSelectionInfo):
        # Input:
        # subFrame - a subbandFrame object containing 12/36 output samples of the
        #                polyphase filterbank
        # scaleFactorInd - array of length 32 containing the indices pointing to
        #                  the respective scale factors for each band
        # nBitsSubband - array of length 32 determining the bit allocation for
        #                each band
        # Output:
        # transmit - transmitFrame object containing the indices of subbands to be
        #            coded, the number of bits allocated to each band, the indices
        #            of scale factors for every coded band and the quantized
        #            samples for every coded band
        
        assert (type(subFrame)==subbandFrame),"Input not a subbandFrame object!"
        
        transmitScalefactorInd = []
        transmitSubband = []
        transmitNSubbands = []
        
        
        for iBand in range(32):
            if nBitsSubband[iBand]>0:
                transmitNSubbands.append(iBand)
                transmitScalefactorInd.append(scaleFactorInd[iBand])
                
                quantizedBand = np.zeros(36)
                for iScf in range(3):
                    normalizedBand = subFrame.frame[iBand,iScf*12:iScf*12+12]/scaleFactorTable[scaleFactorInd[iBand,iScf]]
                    assert((normalizedBand<1).all())
                    quantizedBand[iScf*12:iScf*12+12] = subbandQuantizer(normalizedBand,nBitsSubband[iBand])
                
                
                transmitSubband.append(quantizedBand)
                
        transmitNSubbands = np.array(transmitNSubbands,dtype=int)
        transmitScalefactorInd = np.array(transmitScalefactorInd,dtype=int)
        transmitSubband = np.array(transmitSubband)
        
        transmit=transmitFrame(transmitNSubbands,nBitsSubband,transmitScalefactorInd,transmitSubband,scfTransmissionPattern,scfSelectionInfo)
                
        return transmit
    



#%%

"""
The encoder stage combines the scale factor calculation, bit allocation and 
quantization of the previously calculated subband samples by continously 
pushing new subband samples into a subbandFrame object and performing the 
operations. It outputs a list of transmitFrame objects which can then be read
by the decoder.
"""
if layer==1:
    def encoder(subSamples,nTotalBits,x,sampleRate,smrModel,Aweighting):
        # Input:
        # subSamples - array or list of subband samples calculated from the
        #              analysis polyphase filterbank
        # nTotalBits - the bitrate defined in terms of bits available per frame
        # Output:
        # transmitFrames - list of transmitFrames objects, the simulated bitstream
        
        
        # initialize and push subband samples into a subbandFrame object
        subFrame = subbandFrame(layer=layer)
        
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
            scaleFactorVal, scaleFactorInd = calcScaleFactors(subFrame.frame)
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
            elif smrModel == 'zero':
                SMR = np.zeros(32)
            
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

elif layer==2:
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
        
        scfUsed = []
        trPat = []
        selInfo = np.zeros(32,dtype=int)
        
        for iBand in range(32):
        
            dscf1 = scfInd[iBand,0]-scfInd[iBand,1]
            dscf2 = scfInd[iBand,1]-scfInd[iBand,2]
        
            scfClass1 = scalefactorClass(dscf1)
            scfClass2 = scalefactorClass(dscf2)
        
            sU, tP, selInfo[iBand] = transmissionPatternsLayerII(scfClass1,scfClass2)
            scfUsed.append(sU)
            trPat.append(tP)
        
        scfUsed = np.array(scfUsed)
        trPat = np.array(trPat)
        
        return scfUsed, trPat, selInfo
    
    def mapScfLayerII(scfUsed,scfInd):
        
        for iBand in range(32):
            if scfUsed[iBand,0] != 4:
                scfInd[iBand,:] = scfInd[iBand,scfUsed[iBand]-1]
            else:
                maxInd = np.argmin(scfInd[iBand,:]) # find maximum scalefactor
                scfInd[iBand,:] = scfInd[iBand,maxInd] 
            scfVal = scaleFactorTable[scfInd]
        
        return scfInd, scfVal
    
    def encoder(subSamples,nTotalBits,x,sampleRate,smrModel,Aweighting):
        # Input:
        # subSamples - array or list of subband samples calculated from the
        #              analysis polyphase filterbank
        # nTotalBits - the bitrate defined in terms of bits available per frame
        # Output:
        # transmitFrames - list of transmitFrames objects, the simulated bitstream
        
        
        # initialize and push subband samples into a subbandFrame object
        subFrame = subbandFrame(layer=layer)
        
        transmitFrames=[]
        
        nFrames = int(subSamples.shape[1]/subFrame.nSamples)
        iSubSample = 0
        iAudioSample = 0
        
        
        # preallocate for bit allocation
        nHeaderBits   =   32 # bits needed for header
        nCrcBits      =    0 # CRC checkword, 16 if used
        nBitAllocBits =    sumNbal # bit allocation is more complex in Layer II and depends on the bit allocation tables defined in Tables B.2
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
            scaleFactorInd = np.zeros((32,3),dtype=int)
            for iScf in range(3):
                scaleFactorVal[:,iScf], scaleFactorInd[:,iScf] = calcScaleFactors(subFrame.frame[:,iScf*12:iScf*12+12])
                
            scfUsed, scfTransmissionPattern, scfSelectionInfo = codeScfLayerII(scaleFactorInd)
            
            scaleFactorInd, scaleFactorVal = mapScfLayerII(scfUsed,scaleFactorInd)
    
    
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
            elif smrModel == 'zero':
                SMR = np.zeros(32)
            
            # bit allocation for current frame
            #start = time.time()        
            nBitsSubband = assignBits(scaleFactorVal,nTotalBits,nMiscBits,subFrame.nSamples,SMR,scfTransmissionPattern)
            #end = time.time()
            #print("bit allocation in")
            #print(end - start)
            # quantize subband samples of current frame and store in transmitFrame object
            #start = time.time()
            transmit = quantizeSubbandFrame(subFrame,scaleFactorInd,nBitsSubband,scfTransmissionPattern,scfSelectionInfo)
        
            
            transmitFrames.append(transmit)
            #end = time.time()
            #print("quantization in")
            #print(end - start)
            
        return transmitFrames

#%% Decoder

"""
The decoder takes each transmitFrame object and multiplies the quantized
subband values with the corresponding scale factors and then runs the resulting
stream of subband-separated samples into the synthesis filterbank to
reconstruct a single audio time signal

Refer to flowcharts A.1 and A.2 in the MPEG-1 Part 3 standard
"""

# object containing 1024 samples to be fed to the Synthesis Filterbank
class synthesisBuffer:
    def __init__(self,bufferVal=np.zeros(1024)):
        assert (len(bufferVal)==1024),"Input not length 1024!"
        self.bufferVal = bufferVal
    
    # 1 subband sample with 32 bands is pushed into the buffer at a time
    def pushBlock(self,decodedBand):
        self.bufferVal[64:] = self.bufferVal[:-64]
        self.bufferVal[0:64] = synthesisMatrixing(decodedBand)


# part of the synthesis filter process
def synthesisMatrixing(sampleBlock):
    assert (len(sampleBlock)==32),"Input not length 32!"
    V = np.zeros(64)
    for n in range(64):
        V[n] = np.sum(N[n,:]*sampleBlock)
            
    return V


def nFun(n,k):
    N = np.cos((16+n)*(2*k+1)*np.pi/64)
    
    return N
    
# analysis filterbank for MPEG Audio subband coding
def synthesisFilterbank(x):
    # x - synthesisBuffer object
    
    assert (type(x)==synthesisBuffer),"Input not a synthesisBuffer object!"
    

    U = np.zeros(512)
    for n in range(8):
        for m in range(32):
                U[n*64+m]    = x.bufferVal[n*128+m]
                U[n*64+32+m] = x.bufferVal[n*128+96+m]
    
    
    
    W = U*D
    
    S = np.zeros(32)
    for m in range(32):
        S[m] = np.sum(W[m+32*np.arange(0,16)])
        
    return S
    

def decoder(transmitFrames):
    # Input:
    # transmitFrames - list of transmitFrames objects, the simulated bitstream
    # Output:
    # decodedSignal - reconstructed audio time signal stream
    
    
    nFrames = len(transmitFrames)
    
    if layer==1:
        nSubSamples = nFrames*12
    elif layer==2:
        nSubSamples = nFrames*36
    decodedBands = np.zeros((nSubSamples,32))
    
    
    for iFrame in range(nFrames):
        iBand=0
        for indBand in transmitFrames[iFrame].nSubbands:
                if layer==1:
                    decodedBands[iFrame*12:(iFrame+1)*12,indBand] = transmitFrames[iFrame].quantSubbandSamples[iBand]*scaleFactorTable[transmitFrames[iFrame].scalefactorInd[iBand]]
                elif layer==2:
                    for iScf in range(3):
                        decodedBands[(iFrame*36+iScf*12):(iFrame*36+iScf*12+12),indBand] = transmitFrames[iFrame].quantSubbandSamples[iBand,(iScf*12):(iScf*12+12)]*scaleFactorTable[transmitFrames[iFrame].scalefactorInd[iBand,iScf]]

                iBand +=1
    
    synBuff = synthesisBuffer()
    decodedSignal = np.zeros(nSubSamples*32)
    iSample = 0
    
    for iSubSample in range(nSubSamples):
        synBuff.pushBlock(decodedBands[iSubSample,:])
        decodedSignal[iSample:iSample+32] = synthesisFilterbank(synBuff)
        iSample += 32
    
    return decodedSignal


#%% Miscellaneous

# Convert right-left stereo array to mid-side coded array
def convRLtoMS(RLsig):
    
    nSamples,stereo=RLsig.shape
    assert stereo==2,"Not a stereo file!"
    MSsig = np.zeros([nSamples,2])
    
    MSsig[:,0] = (RLsig[:,0] + RLsig[:,1])/np.sqrt(2)
    MSsig[:,1] = (RLsig[:,0] - RLsig[:,1])/np.sqrt(2)
    
    return MSsig
