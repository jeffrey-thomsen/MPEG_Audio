# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:20:48 2020

@author: Jeffrey
"""
import numpy as np

def B2Loader(sampleRate,bitrate):
    import numpy
    
    """
    Hopefully improved allocation table to save some bits
    """
    nbal = numpy.array([4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,0,0])
    quantLevels = numpy.array([
      [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
      [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
      [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
      [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
      [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
      [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
      [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
      [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15, 31, 63, 127, 255,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0, 3, 7, 15,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
      [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])
    
    sbLimit = 30
    sumNbal = 96
    
    maxBitsSubband = np.array([16,16,16,16,16,16,16,16,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,0,0])
    
    """
    egalitarian 2 to 15 bit solution, identical to Layer I
    """
    
    # nbal = numpy.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4])
    # quantLevels = numpy.array([
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767],
    #   [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767]])
    
     
    # sbLimit = 32
    # sumNbal = 128
    
    # maxBitsSubband = np.array([15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15])
    
    
    """
    First attempt to reduce allocation of many bits to high bands,
    without abolishing 5- and 9-step yet
    """
    # nbal = numpy.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4])
    # quantLevels = numpy.array([
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383],
    #  [0, 3, 5,  7,  9, 15,  31,  63, 127,  255,  511, 1023, 2047,  4095,  8191, 16383]])
    
     
    # sbLimit = 32
    # sumNbal = 128
    
    # maxBitsSubband = np.array([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14])
    
    
    """
    Original MPEG defined tables. Abolished these because apparently the
    5- and 9-step quantizations introduce aliasing
    """
    
    
    # if (sampleRate == 44100):
    #     if (bitrate in (56, 64, 80)):
    #         table = 'a'
    #     elif (bitrate in (32, 48)):
    #         table = 'c'
    #     else:
    #         table = 'b'
    # elif (sampleRate == 32000):
    #     if (bitrate in (56, 64, 80)):
    #         table = 'a'
    #     elif (bitrate in (32, 48)):
    #         table = 'c'
    #     else:
    #         table = 'b'
    # elif (sampleRate == 48000):
    #     if (bitrate in (32, 48)):
    #         table = 'd'
    #     else:
    #         table = 'a'
        
        
    # if table == 'a':
    #     nbal = numpy.array([4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,0,0,0,0,0])
    #     quantLevels = numpy.array([[0,3,7,15,31,63,127,   255,       511,   1023,   2047,   4095,    8191,    16383,    32767,    65535],
    #     [0,      3,   7,   15,      31,   63,   127,   255,       511,   1023,   2047,   4095,    8191,    16383,    32767,   65535],
    #     [0,      3,   7,   15,      31,   63,   127,   255,       511,   1023,   2047,   4095,    8191,    16383,    32767,    65535],
    #     [0,      3,   5,   7,      9,   15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
    #     [0,      3,   5,   7,      9,   15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
    #     [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
    #     [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
    #     [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
    #     [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
    #     [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
    #     [0,      3,   5,   7,      9,    15,   31,   63,       127,   255,   511,   1023,    2047,    4095,    8191,    65535],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   7,      9,    15,   31,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,      3,   5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])

               
    #     sbLimit = 27
    #     sumNbal = 88
        
    #     maxBitsSubband = np.array([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,0,0,0,0,0])
        
    # elif table == 'b':
    #     nbal = numpy.array([4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,0,0])
    #     quantLevels = numpy.array([[0,3,7,15,31,63,127,   255,      511,    1023,    2047,   4095,    8191,    16383,       32767,   65535],
    #     [0,   3,    7,  15,      31,      63,  127,    255,      511,    1023,    2047,   4095,    8191,    16383,       32767,   65535],
    #     [0,   3,    7,  15,      31,      63,  127,    255,      511,    1023,    2047,   4095,    8191,    16383,       32767,   65535],
    #     [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
    #     [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
    #     [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
    #     [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
    #     [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
    #     [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
    #     [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
    #     [0,   3,    5,   7,       9,      15,   31,     63,      127,    255,     511,    1023,    2047,    4095 ,       8191 ,   65535],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   7,       9,      15,   31,     65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,    5,   65535,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])
             
    #     sbLimit = 30
    #     sumNbal = 94
        
    #     maxBitsSubband = np.array([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,0,0])
        
    # elif table == 'c':
    #     nbal = numpy.array([4,4,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #     quantLevels = numpy.array([[0,3,5,9,15,31,63,127,255,      511,  1023, 2047, 4095,   8191,    16383,    32767],
    #     [0,     3,    5,   9, 15,         31,    63,     127,255,511, 1023, 2047, 4095,   8191,    16383,    32767],
    #     [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,     3,    5,   9, 15,         31,    63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])
             
    #     sbLimit = 8
    #     sumNbal = 26
        
    #     maxBitsSubband = np.array([15,15,7,7,7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        
    # elif table == 'd':
    #     nbal = numpy.array([4,4,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #     quantLevels = numpy.array([[0,   3,       5,   9,   15,    31,      63,     127,      255,   511,   1023, 2047, 4095,   8191,   16383,   32767],
    #     [0,   3,       5,   9,   15,    31,      63,     127,      255,   511,   1023, 2047, 4095,   8191,   16383,   32767],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,   3,       5,   9,   15,    31,      63,     127,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])
             
    #     sbLimit = 12
    #     sumNbal = 38
        
    #     maxBitsSubband = np.array([15,15,7,7,7,7,7,7,7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    return nbal, quantLevels, sbLimit, sumNbal, maxBitsSubband