# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:11:24 2020

@author: Jeffrey
"""
import numpy as np

# # to time stuff:
# import time
# start = time.time()
# #expression
# end = time.time()
# print("Expression calculated in")
# print(end - start)
        


#%% Implementation of the Polyphase filterbank

# object containing 512 samples of audio to be fed to the Polyphase Filterbank
class analysisBuffer:
    def __init__(self,bufferVal=np.zeros(512)):
        assert (len(bufferVal)==512),"Input not length 512!"
        self.bufferVal = bufferVal
    
    # by definition, 32 samples are pushed into the buffer at a time
    def pushBlock(self,sampleBlock):
        assert (len(sampleBlock)==32),"Sample block length not 32!"
        self.bufferVal[32:] = self.bufferVal[:-32]
        self.bufferVal[0:32] = sampleBlock

# analysis filterbank for MPEG Audio subband coding
def polyphaseFilterbank(x):
    # x - analysisBuffer object
    assert (type(x)==analysisBuffer),"Input not an analysisBuffer object!"
    
    C = np.load('data/mpeg_analysis_window.npy') # analysis window defined by MPEG

    assert (len(C)==512),"Window length not 512!"

    M = np.load('data/mpeg_polyphase_analysis_matrix_coeff.npy')

    subbandSamples = np.zeros(32)
    for n in range(32): # this nested for-loop structure takes all the time! How to speed up?
        for k in range(63):
            for m in range(7):
                subbandSamples[n] = subbandSamples[n] + (M[n,k] * (C[k+64*m]*x.bufferVal[k+64*m]))
        
    return subbandSamples


# part of the Polyphase Filterbank
# this function was originally used to calculate the matrix every time,
# but now it is simply loaded from an .npy array

# def calcAnalysisMatrixCoeff():
#     M = np.zeros([32,64])
#     for n in range(32):
#         for k in range(64):
#             M[n,k] = np.cos((2*n+1)*(k-16)*np.pi/64)
            
#     return M
     
#%% Routine that takes blocks of audio and feeds it to the buffer, 
#   then groups subband samples into frames

# takes raw audio signal, runs it into the analysisBuffer and calculates the 
# filterbank outputs subbandSamples
def feedCoder(x):
    # Input:
    # x - mono or stereo PCM audio signal samples as array
    # Output:
    # subbandSamples - list of numpy arrays of length 32, one array for each 
    #                  subband sample corresponding to 32 input samples
    
    nSamples,stereo=x.shape
    assert (stereo==1 or stereo==2),"Input not mono or stereo!"
    
    if stereo==2:
        x  = x[:,0] # eventually will need to be xLeft/xRight and will need a whole routine to handle both channels
        #xRight = x[:,1]
    
    # zero-pad to divisible of 32 samples
    modulo32 = nSamples%32
    nPadding = 32-modulo32
    xHold = x
    x = np.zeros([nSamples])
    x[0:nSamples] = xHold
    nSamples = nSamples+nPadding
    #nBlocks = int(nSamples/32)
    assert (nSamples%32==0),"Zero-padding mistake!"
        
    xBuffer = analysisBuffer()
    iBlock = 0
    iSample = 0
    subbandSamples = []
    
    while iSample+32<=4096:#32768: # eventually needs to be nSamples, but right now it's too slow
        xBuffer.pushBlock(x[iSample:iSample+32])
        subbandSamples.append((polyphaseFilterbank(xBuffer)))
        iBlock  += 1
        iSample += 32
    
    return subbandSamples


#%% 

# object containing a frame of 12 or 36 subband sample outputs of the polyphase filterbank, must be initialized empty first
class subbandFrame:
    def __init__(self,layer=1):
        assert (layer==1 or layer==2 or layer==3),"Encoding layer type not 1, 2 or 3!"
        self.layer=layer
        if self.layer==1:
            self.nSamples = 12
        elif self.layer ==2 or self.layer==3:
            self.nSamples = 36
        
        self.frame = np.zeros([32,self.nSamples])
        
    def pushFrame(self,subbandSamples):
        assert (len(subbandSamples)>=self.nSamples),"Not enough entries in subbandSamples list!"
        for i in range(self.nSamples):
            self.frame[:,i]=(subbandSamples[i])
        subbandSamples = subbandSamples[self.nSamples:]   
        return subbandSamples


#%% Scalefactor Calculation

# compare max abs values of each subband frame to scalefactor table and deduct
# fitting scale factor
def calcScaleFactors(subbandFrame):
    if subbandFrame.layer == 1:
        assert (subbandFrame.frame.shape==(32, 12)),"Wrong subbandFrame array dimensions!"
        subbandMaxVals = np.amax(np.abs(subbandFrame.frame),axis=1)
    else:
        print("Error! Layer II and III coding not implemented yet")
        raise SystemExit(0)
    
    scaleFactorTable = np.load('data/mpeg_scale_factors.npy') # scale factors defined by MPEG
    scaleFactorTable = np.flip(scaleFactorTable)
    assert (len(scaleFactorTable)==64),"Table length not 64!"
    
    scaleFactor = np.zeros(32)
    scaleFactorIndex = []
    for iCompare in range(32):
        scaleFactor[iCompare]=scaleFactorTable[np.argmax(scaleFactorTable>subbandMaxVals[iCompare])]
        scaleFactorIndex.append(63-np.argmax(scaleFactorTable>subbandMaxVals[iCompare]))
    
    return scaleFactor, scaleFactorIndex

# convert scale factor indices into binary representation for coding
# NOTE: right now no conversion to binary yet
def codeScaleFactors(scaleFactorIndex):
    assert (type(scaleFactorIndex[0]==int)),"Input not integer value!"
    
    codedScaleFactor = []
    for iBand in range(len(scaleFactorIndex)):
        codedScaleFactor.append(bin(scaleFactorIndex[iBand]))
    
    return codedScaleFactor




#%% Bit allocation

# main function for bit allocation that calculates the number of bits to be
# assigned to each subband of a frame of subband samples
def assignBits(subbandFrame):
    # Input:
    # subbandFrame - a subbandFrame object containing 12 output samples of the
    #                polyphase filterbank
    
    nTotalBits    = 3072 # bits available per frame representing 384 samples @8bps
    nHeaderBits   =   32 # bits needed for header
    nCrcBits      =    0 # CRC checkword, 16 if used
    nBitAllocBits =  128 # bit allocation -> codes 4bit int values 0...15 for each of 32 subbands
    nAncBits      =    0 # bits needed for ancillary data
    
    nAvailableBits = nTotalBits - (nHeaderBits + nCrcBits + nBitAllocBits + nAncBits) # bits available for coding samples and scalefactors

    nBitsSubband=[]
    for iC in range(32):
        nBitsSubband.append(0)
    
    while nAvailableBits > possibleIncreaseInBits(nBitsSubband):
        
        minSubBand = determineMinimalMNR(nBitsSubband)
        nBitsSubband = increaseNBits(nBitsSubband,minSubBand)
        
        nScfBits, nSplBits = updateBitAllocation(nBitsSubband)
        nAvailableBits = nTotalBits - (nHeaderBits + nCrcBits + nBitAllocBits + nSplBits + nScfBits + nAncBits)
    
    # for now it's just a fixed 8 bits per subband sample,  allocation routine will be included later
    # nBitsSubband = []
    # for iC in range(32):
    #     nBitsSubband.append(8)
    
    return nBitsSubband, nSplBits, nScfBits, nAvailableBits

# compare MNRs of each subband and return first subband with lowest MNR
def determineMinimalMNR(nBitsSubband):
    assert (len(nBitsSubband)==32),"Wrong length of input list!"
    
    MNR = updateMNR(nBitsSubband)
    
    minMNRIndex = np.argmin(MNR)
    
    return minMNRIndex

# increase number of allocated bits to chosen subband to next step
def increaseNBits(nBitsSubband,minSubBand):    
    
    currentNBits = nBitsSubband[minSubBand]
    if (currentNBits>0 & currentNBits<15):
        nBitsSubband[minSubBand] += 1
    elif (currentNBits==0):
        nBitsSubband[minSubBand] += 2
    else:
        nBitsSubband[minSubBand] += 0
    
    return nBitsSubband

# calculate MNR of all subbands
def updateMNR(nBitsSubband):
    
    snrTable = np.load('data/mpeg_snr_layer_i.npy') # SNR levels defined by MPEG
    SNR = np.zeros(32)
    MNR = np.zeros(32)
    nBands = 32
    
    for iBand in range(nBands):
        snrIndex = np.where(snrTable[:,0] == nBitsSubband[iBand])[0][0]
        SNR[iBand] = snrTable[snrIndex,2]
        SMR = 0 # eventually determine this from psychoacoustic model
        MNR[iBand] = SNR[iBand] - SMR
        
    return MNR

# returns new number of bits allocated to code subband samples and scalefactors
# according to the update of nBits per subband
def updateBitAllocation(nBitsSubband):
    
    nScfBits = 0 # bits allocated to code scalefactors
    nSplBits = 0 # bits allocated to code samples
    nBands = 32
    
    for iBand in range(nBands):
        if nBitsSubband[iBand]>0:
            nScfBits += 6
            nSplBits += nBitsSubband[iBand]*12
        elif nBitsSubband[iBand]==0:
            nScfBits += 0
            nSplBits += 0
    
    return nScfBits, nSplBits

def possibleIncreaseInBits(nBitsSubband):
    assert (len(nBitsSubband)==32),"Wrong length of input list!"
    # given the current state, return the smallest increase in bits allocated
    # in a potential next iteration step
    
    if not not [i for i, e in enumerate(nBitsSubband) if 0<e<15]:
        minIncrease = 12
    elif not not [i for i, e in enumerate(nBitsSubband) if e == 0]:
        minIncrease = 30
    else:
        minIncrease = 0

    return minIncrease








#%% Quantizer


# multiply subband frame by scalefactors and quantize them with the number of bits
def quantizeSubbandFrame(subbandFrame,scaleFactor,nBitsSubband):
    
    transmitScalefactor = []
    transmitSubband = []
    transmitNSubbands = []
    
    for iBand in range(32):
        if nBitsSubband[iBand]>0:
            transmitNSubbands.append(iBand)
            transmitScalefactor.append(scaleFactor[iBand])
            
            normalizedBand = subbandFrame.frame[iBand,:]*scaleFactor[iBand]
            quantizedBand = subbandQuantizer(normalizedBand,nBitsSubband[iBand])
            
            transmitSubband.append(codeSubband(quantizedBand,nBitsSubband[iBand]))
            
            return transmitSubband


# Quantizer defined by MPEG
def subbandQuantizer(normalizedBand,nBits):
    
    nSteps = int(2**nBits-1)
    indBits = int(nBits-2)
    print(type(indBits))
    nSteps = np.load('data/mpeg_qc_layer_i_nSteps.npy')
    A = np.load('data/mpeg_qc_layer_i_A.npy')
    B = np.load('data/mpeg_qc_layer_i_B.npy')
    
    quantizedBand = A[indBits]*normalizedBand+B[indBits]

    return quantizedBand

# convert scale factor indices into binary representation for coding
def codeSubband(quantizedBand,nBits):
    assert (type(scaleFactorIndex[0]==int)),"Input not integer value!"
    
    codedSubband = []
    for iSample in range(len(quantizedBand)):
        print(quantizedBand[iSample])
        codedVal    = bin(quantizedBand[iSample]) # this should be an integer number but isn't???
        codedVal    = codedVal[0:nBits]
        codedVal[0] = ~codedVal[0]
        codedSubband.append()
    
    return codedScaleFactor

#%% Miscellaneous

# Convert right-left stereo array to mid-side coded array



def convRLtoMS(RLsig):
    
    nSamples,stereo=RLsig.shape
    assert stereo==2,"Not a stereo file!"
    MSsig = np.zeros([nSamples,2])
    
    MSsig[:,0] = (RLsig[:,0] + RLsig[:,1])/np.sqrt(2)
    MSsig[:,1] = (RLsig[:,0] - RLsig[:,1])/np.sqrt(2)
    
    return MSsig
