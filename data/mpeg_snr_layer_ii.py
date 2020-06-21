# this script is used to generate a numpy array containing the signal to noise ratios for Layer II audio coding as defined in Table C.5 of the ISO/IEC 11172-3 (aka MPEG-1 Audio) standard

import numpy
snr=numpy.array([[0, 0,  0.00],
                 [5/3, 3,  7.00],
                 [7/3, 5, 11.00],
                 [3, 7, 16.00],
                 [10/3, 9, 20.84],
                 [4, 15, 25.28],
                 [5, 31, 31.59],
                 [6, 63, 37.75],
                [7, 127, 43.84],
                [8, 255, 49.89],
                [9, 511, 55.93],
               [10, 1023, 61.96],
               [11, 2047, 67.98],
               [12, 4095, 74.01],
               [13, 8191, 80.03],
              [14, 16383, 86.05],
              [15, 32767, 92.01],
              [16, 65535, 98.01]])

