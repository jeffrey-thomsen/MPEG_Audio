import numpy
#%% Table B.2d. -- Possible quantization per subband

# Fs = 48kHz           ------- not relevant -------
# Fs = 44,1kHz         ------- not relevant -------
# Fs = 32kHz           Bitrates per channel = 32, 48 kbits/s.

nbal = numpy.array([4,4,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
quantLevels = numpy.array([[0,   3,       5,   9,   15,    31,      63,     127,      255,   511,   1023, 2047, 4095,   8191,   16383,   32767],
[0,   3,       5,   9,   15,    31,      63,     127,      255,   511,   1023, 2047, 4095,   8191,   16383,   32767],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,   3,       5,   9,   15,    31,      63,     127,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
     
sbLimit = 12
sumNbal = 38