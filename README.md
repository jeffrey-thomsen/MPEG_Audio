# MPEG_Audio

Project for the DTU course Data Compression and Image Communication. A Python implementation of an audio coder based on the MPEG-1 standard.

# Introduction

The scope of this project is to implement an audio encoder and decoder based on the principles of Part 3 of the MPEG-1 standard (ISO/IEC 11172-3) \cite{mpeg} in Python. In particular, Level I of the three coding levels described is implemented, but without the use of a complex psychoacoustic model.

The Layer I audio coder is based on a subband coding approach, using a so-called polyphase filterbank, and scaling and quantizing blocks of subband samples individually. To determine the optimal distribution of coding bits available for each frame amongst the subbands, mask-to-noise ratios (MNR) are calculated for each subband and coding bits are spent in an iterative manner to the subband that will benefit most from an increase in quantization steps.

The evaluation of the coder is performed on a chosen test music signal by measuring the code length, determining the distortion and calculating an entropy estimate to give indications of the potential usefulness of adding an entropy coder to the system.

# How to use

To look at the results of some tests that were performed, evaluationScript.py offers the reproduction of the results presented in the report, including rate-distoriton, spectral comparison and entropy estimate.

The heart of the coder is implemented in mpegAudioFunctions.py

These can be tested with the mpegFullTest.py script. Within the script the bit rate can be altered and the test data can be chosen. In the current setting 2 seconds of the song "Traffic in the Sky" will be coded at 1 bit per sample.

The test script can be run with an IDE like Spyder or with a command line operation, provided Python is installed on your system. In the current directory, run the command

python mpegFullTest.py

You will recieve a command line output stating the running time of the different coding steps as well as SNR calculation and entropy estimate of the scale factors. Two WAV-files are generated for perceptual evaluation, test_source.wav and test_recons.wav. This way you can compare the source signal with the resonstructed signal.



Additional scripts include mpegFilterBankTest.py and quantizeCompare.py which are not relevant for testing the implementation in its current state.