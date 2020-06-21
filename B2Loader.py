# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:20:48 2020

@author: Jeffrey
"""
import numpy as np

def B2Loader(sampleRate,bitrate):
    import numpy
    
    if (sampleRate == 44100):
        if (bitrate in (56, 64, 80)):
            table = 'a'
        elif (bitrate in (32, 48)):
            table = 'c'
        else:
            table = 'b'
    elif (sampleRate == 32000):
        if (bitrate in (56, 64, 80)):
            table = 'a'
        elif (bitrate in (32, 48)):
            table = 'c'
        else:
            table = 'b'
    elif (sampleRate == 48000):
        if (bitrate in (32, 48)):
            table = 'd'
        else:
            table = 'a'
        
        
    if table == 'a':
        nbal = numpy.array([4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,0,0,0,0,0])
        quantLevels = numpy.array([[0,3,7,15,31,63,127,   255,       511,   1023,   2047,   4095,    8191,    16383,    32767,    65535],
        [0,      3,   7,   15,      31,   63,   127,   255,       511,   1023,   2047,   4095,    8191,    16383,    32767,   65535],
        [0,      3,   7,   15,      31,   63,   127,   255,       511,   1023,   2047,   4095,    8191,    16383,    32767,    65535],
        [0,      3,   5,   7,      9,   15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
        [0,      3,   5,   7,      9,   15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
        [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
        [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
        [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
        [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
        [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
        [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,      3,   5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])

               
        sbLimit = 27
        sumNbal = 88
        
        maxBitsSubband = np.array([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,0,0,0,0,0])
        
    elif table == 'b':
        nbal = numpy.array([4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,0,0])
        quantLevels = numpy.array([[0,3,7,15,31,63,127,   255,      511,    1023,    2047,   4095,    8191,    16383,       32767,   65535],
        [0,   3,    7,  15,      31,      63,  127,    255,      511,    1023,    2047,   4095,    8191,    16383,       32767,   65535],
        [0,   3,    7,  15,      31,      63,  127,    255,      511,    1023,    2047,   4095,    8191,    16383,       32767,   65535],
        [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
        [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
        [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
        [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
        [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
        [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
        [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
        [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])
             
        sbLimit = 30
        sumNbal = 94
        
        maxBitsSubband = np.array([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,0,0])
        
    elif table == 'c':
        nbal = numpy.array([4,4,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        quantLevels = numpy.array([[0,3,5,9,15,31,63,127,255,      511,  1023, 2047, 4095,   8191,    16383,    32767],
        [0,     3,    5,   9, 15,         31,    63,     127,255,511, 1023, 2047, 4095,   8191,    16383,    32767],
        [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])
             
        sbLimit = 8
        sumNbal = 26
        
        maxBitsSubband = np.array([15,15,7,7,7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        
    elif table == 'd':
        nbal = numpy.array([4,4,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        quantLevels = numpy.array([[0,   3,       5,   9,   15,    31,      63,     127,      255,   511,   1023, 2047, 4095,   8191,   16383,   32767],
        [0,   3,       5,   9,   15,    31,      63,     127,      255,   511,   1023, 2047, 4095,   8191,   16383,   32767],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
        [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])
             
        sbLimit = 12
        sumNbal = 38
        
        maxBitsSubband = np.array([15,15,7,7,7,7,7,7,7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    return nbal, quantLevels, sbLimit, sumNbal, maxBitsSubband