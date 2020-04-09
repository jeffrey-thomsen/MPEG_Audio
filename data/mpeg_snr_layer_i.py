# this script is used to generate a numpy array containing the signal to noise ratios for Layer I audio coding as defined in Table C.2 of the ISO/IEC 11172-3 (aka MPEG-1 Audio) standard

import numpy
snr=numpy.array([[    0,  0.00],
                 [    3,  7.00],
                 [    7, 16.00],
                 [   15, 25.28],
                 [   31, 31.59],
                 [   63, 37.75],
                 [  127, 43.84],
                 [  255, 49.89],
                 [  511, 55.93],
                 [ 1023, 61.96],
                 [ 2047, 67.98],
                 [ 4095, 74.01],
                 [ 8191, 80.03],
                 [16383, 86.05],
                 [32767, 92.01]])