# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:11:24 2020

@author: Jeffrey
"""
import numpy as np

# to time stuff:
#import time
# start = time.time()
# #expression
# end = time.time()
# print("Expression calculated in")
# print(end - start)
        

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
    
    # by definition, 32 samples are pushed into the buffer at a time
    def pushBlock(self,sampleBlock):
        assert (len(sampleBlock)==32),"Sample block length not 32!"
        self.bufferVal[32:] = self.bufferVal[:-32]
        self.bufferVal[0:32] = sampleBlock[::-1]

# analysis filterbank for MPEG Audio subband coding
def polyphaseFilterbank(x):
    # x - analysisBuffer object
    assert (type(x)==analysisBuffer),"Input not an analysisBuffer object!"
    
    C = np.load('data/mpeg_analysis_window.npy') # analysis window defined by MPEG

    Z = C*x.bufferVal
    
    Y = np.zeros(64)
    for k in range(64):
        for m in range(8):
            Y[k] = Y[k] + Z[k+64*m]
    
    subbandSamples = np.zeros(32)
    for n in range(32):
        subbandSamples[n] = np.sum(mFun(n,np.arange(0,64)) * Y)

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
    # subbandSamples - list of numpy arrays of length 32, one array for each 
    #                  subband sample corresponding to 32 input samples
    
    nSamples,stereo=x.shape
    assert (stereo==0 or stereo==1 or stereo==2),"Input not mono or stereo!"
    
    if stereo==1 or stereo==2:
        x  = x[:,0] # eventually will need to be xLeft/xRight and will need a whole routine to handle both channels
    
    # zero-pad to divisible of 32 samples
    modulo32 = nSamples%32
    nPadding = 32-modulo32
    xHold = x
    x = np.zeros([nSamples+nPadding])
    x[0:nSamples] = xHold
    nSamples = nSamples+nPadding
    assert (nSamples%32==0),"Zero-padding mistake!"
        
    xBuffer = analysisBuffer()
    iBlock = 0
    iSample = 0
    subbandSamples = []
    
    while (iSample+32<=nSamples):
        xBuffer.pushBlock(x[iSample:iSample+32])
        subbandSamples.append((polyphaseFilterbank(xBuffer)))
        iBlock  += 1
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
"""
The scalefactor calculation takes a subbandFrame object, compares the max abs 
values of each subband to the scalefactor table and assigns the fitting scale 
factors to the subbands
"""
# compare max abs values of each subband frame to scalefactor table and deduct
# fitting scale factor
def calcScaleFactors(subbandFrame):
    if subbandFrame.layer == 1:
        assert (subbandFrame.frame.shape==(32, 12)),"Wrong subbandFrame array dimensions!"
        subbandMaxVals = np.amax(np.abs(subbandFrame.frame),axis=1)
        assert ((subbandMaxVals<2.0).all()), "Maximum subband value larger than 2!"
    else:
        print("Error! Layer II and III coding not implemented yet")
        raise SystemExit(0)
    
    scaleFactorTable = np.load('data/mpeg_scale_factors.npy') # scale factors defined by MPEG
    scaleFactorTable = np.flip(scaleFactorTable) # needed for finding first value larger than sample value
    assert (len(scaleFactorTable)==63),"Table length not 64!"
    
    scaleFactor = np.zeros(32)
    scaleFactorIndex = []
    for iCompare in range(32):
        scaleFactor[iCompare]=scaleFactorTable[np.argmax(scaleFactorTable>subbandMaxVals[iCompare])]
        scaleFactorIndex.append(62-np.argmax(scaleFactorTable>subbandMaxVals[iCompare]))

    
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
"""
The bit allocation actually takes into account the output of the psychoacoustic
model, but instead I have implemented an equivalent with equivSMR.
It takes a subbandFrame object, and through an iterative process it 
assigns a number of coding bits to each subband
"""

# main function for bit allocation that calculates the number of bits to be
# assigned to each subband of a frame of subband samples
def assignBits(subbandFrame,scaleFactorVal,nTotalBits):
    # Input:
    # subbandFrame - a subbandFrame object containing 12 output samples of the
    #                polyphase filterbank
    # scaleFactorVal - an array containing the scale factors for each subband
    # nTotalBits - number of bits to be allocated to this frame
    
    #nTotalBits    =  768 # bits available per frame representing 384 samples @8bps
    nHeaderBits   =   32 # bits needed for header
    nCrcBits      =    0 # CRC checkword, 16 if used
    nBitAllocBits =  128 # bit allocation -> codes 4bit int values 0...15 for each of 32 subbands
    nAncBits      =    0 # bits needed for ancillary data
    
    nAvailableBits = nTotalBits - (nHeaderBits + nCrcBits + nBitAllocBits + nAncBits) # bits available for coding samples and scalefactors

    nBitsSubband=np.zeros(32,dtype=int)
    
    while (nAvailableBits >= possibleIncreaseInBits(nBitsSubband)) & (np.where(nBitsSubband < 15)[0].size > 0):
        
        minSubBand = determineMinimalMNR(nBitsSubband,scaleFactorVal,nAvailableBits)
        nBitsSubband = increaseNBits(nBitsSubband,minSubBand)
        
        nScfBits, nSplBits = updateBitAllocation(nBitsSubband)
        nAvailableBits = nTotalBits - (nHeaderBits + nCrcBits + nBitAllocBits + nSplBits + nScfBits + nAncBits)
        assert (nAvailableBits>=0),"Allocated too many bits!"
    
    return nBitsSubband, nSplBits, nScfBits, nAvailableBits

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

# compare MNRs of each subband and return first subband with lowest MNR
def determineMinimalMNR(nBitsSubband,scaleFactorVal,nAvailableBits):
    assert (len(nBitsSubband)==32),"Wrong length of input list!"
    
    MNR = updateMNR(nBitsSubband,scaleFactorVal)
    
    minMNRIndex = np.argmin(MNR)
    
    while minMNRIndex in np.where(nBitsSubband == 15)[0]:
        MNR[minMNRIndex] = np.inf
        minMNRIndex = np.argmin(MNR)
    
    if nAvailableBits<30:
        while minMNRIndex in np.where(nBitsSubband == 0)[0]:
            MNR[minMNRIndex] = np.inf
            minMNRIndex = np.argmin(MNR)
    
    return minMNRIndex

# calculate MNR of all subbands
def updateMNR(nBitsSubband,scaleFactorVal):
    
    snrTable = np.load('data/mpeg_snr_layer_i.npy') # SNR levels defined by MPEG
    SNR = np.zeros(32)
    MNR = np.zeros(32)
    nBands = 32
    
    for iBand in range(nBands):
        assert (nBitsSubband[iBand]<16),"Too many bits assigned!"
        snrIndex = np.where(snrTable[:,0] == nBitsSubband[iBand])[0][0]
        SNR = snrTable[snrIndex,2]
        
        SMR = equivSMR(scaleFactorVal[iBand])
        
        MNR[iBand] = SNR - SMR
        
    return MNR

# calculate SMR equivalent from scalefactor
def equivSMR(scaleFactorVal):
    
    equivSMR = 20 * np.log10(scaleFactorVal * 32768) - 10
    
    return equivSMR

# increase number of allocated bits to chosen subband to next step
def increaseNBits(nBitsSubband,minSubBand):    
    
    currentNBits = nBitsSubband[minSubBand]
    assert (currentNBits<16),"Too many bits assigned!"
    if (currentNBits>0) & (currentNBits<15):
        nBitsSubband[minSubBand] += 1
    elif (currentNBits==0):
        nBitsSubband[minSubBand] += 2
    else:
        nBitsSubband[minSubBand] += 0
    
    return nBitsSubband

# returns new number of bits allocated to code subband samples and scalefactors
# according to the update of nBits per subband
def updateBitAllocation(nBitsSubband):
    assert (len(nBitsSubband)==32),"Wrong length of input list!"
    
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



#%% Quantizer

"""
The quantizer takes a subbandFrame object, its corresponding scalefactors and 
bit assignments, divides each sample in the frame with the corresponding 
scalefactor of the band and applies the quantization according to the number of
 bits allocated to each band. The quantized subband samples are then wrapped 
in a transmitFrame object together with the other input data, quasi 
representing a block of the formatted bitstream.

"""

#
class transmitFrame:
    def __init__(self,nSubbands=None,nBitsSubband=None,scalefactorInd=None,quantSubbandSamples=None):
        assert (len(nSubbands)==len(scalefactorInd)==len(quantSubbandSamples)), "Length of input lists not consistent!"
        self.nSubbands = nSubbands
        self.nBitsSubband = nBitsSubband
        self.scalefactorInd = scalefactorInd
        self.quantSubbandSamples = quantSubbandSamples
        

# multiply subband frame by scalefactors and qua0,,ntize them with the number of bits
def quantizeSubbandFrame(subbandFrame,scaleFactorInd,nBitsSubband):

    transmitScalefactorInd = []
    transmitSubband = []
    transmitNSubbands = []
    
    scaleFactorTable = np.load('data/mpeg_scale_factors.npy') # scale factors defined by MPEG
    
    for iBand in range(32):
        if nBitsSubband[iBand]>0:
            transmitNSubbands.append(iBand)
            transmitScalefactorInd.append(scaleFactorInd[iBand])
            
            normalizedBand = subbandFrame.frame[iBand,:]/scaleFactorTable[scaleFactorInd[iBand]]
            quantizedBand = subbandQuantizer(normalizedBand,nBitsSubband[iBand])
            
            
            transmitSubband.append(quantizedBand)
            
    
    transmit=transmitFrame(transmitNSubbands,nBitsSubband,transmitScalefactorInd,transmitSubband)
            
    return transmit


# Quantizer defined by MPEG
def subbandQuantizer(normalizedBand,nBits):
    
    indBits = int(nBits-2)
    
    A = np.load('data/mpeg_qc_layer_i_A.npy')
    B = np.load('data/mpeg_qc_layer_i_B.npy')
    nSteps = np.load('data/mpeg_qc_layer_i_nSteps.npy')
    
    quantizedBand = A[indBits]*normalizedBand+B[indBits]

    # assign each quantized value to a fixed value dependent on nBits
    assignedVals = np.arange(-1,1+1e-12,2/(nSteps[indBits]-1))
    threshVals   = np.arange(-1,A[indBits]+B[indBits],-2*B[indBits])-2*B[indBits]
    
    for iSample in range(len(quantizedBand)):
        assert(-1<=quantizedBand[iSample]<=A[indBits]+B[indBits]),"Quantized value out of bounds!"
        assigned=False
        iStep=0
        while not assigned:
            if quantizedBand[iSample]<=threshVals[iStep]:
                quantizedBand[iSample]=assignedVals[iStep]
                assigned=True
            iStep += 1
            assert (iStep<33000), "Quantization while-loop error!"

    return quantizedBand


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
    
    # by definition, 1 subband sample with 32 bands is pushed into the buffer at a time
    def pushBlock(self,decodedBands):
        
        self.bufferVal[64:] = self.bufferVal[:-64]
        self.bufferVal[0:64] = synthesisMatrixing(decodedBands[0,:])
        decodedBands = decodedBands[1:,:]   
        return decodedBands



# part of the synthesis filter process
def synthesisMatrixing(sampleBlock):
    V = np.zeros(64)
    for n in range(64):
        V[n] = np.sum(nFun(n,np.arange(0,32))*sampleBlock)
            
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
    
    D = np.load('data/mpeg_synthesis_window.npy') # synthesis window defined by MPEG
    
    W = U*D
    
    S = np.zeros(32)
    for m in range(32):
        S[m] = np.sum(W[m+32*np.arange(0,16)])
        
    return S
    

def decoder(transmitFrames):
    
    nFrames = len(transmitFrames)
    
    decodedBands = np.zeros((nFrames*12,32))
    
    scaleFactorTable = np.load('data/mpeg_scale_factors.npy') # scale factors defined by MPEG
    
    for iFrame in range(nFrames):
        iBand=0
        for indBand in transmitFrames[iFrame].nSubbands:
                decodedBands[iFrame*12:(iFrame+1)*12,indBand] = transmitFrames[iFrame].quantSubbandSamples[iBand]*scaleFactorTable[transmitFrames[iFrame].scalefactorInd[iBand]]
                iBand +=1
    
    synBuff = synthesisBuffer()
    decodedSignal=[]
    
    while decodedBands[:,0].size>0:
        decodedBands = synBuff.pushBlock(decodedBands)
        decodedSignal+= list(synthesisFilterbank(synBuff))
    decodedSignal = np.array(decodedSignal)
    
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
