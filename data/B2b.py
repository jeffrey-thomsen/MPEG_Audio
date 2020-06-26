import numpy
#%% Table B.2b -- Possible quantization per subband

# Fs = 48 kHz            -------------- not relevant -------------
# Fs = 44,1 kHz          Bitrates per channel = 96, 112, 128, 160, 192 kbits/s and free format
# Fs = 32 kHz            Bitrates per channel = 96, 112, 128, 160, 192 kbits/s and free format

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
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   7,       9,      15,   31,     65535,0,0,0,0,0,0,0,0],
[0,   3,    5,   65535,0,0,0,0,0,0,0,0,0,0,0,0],
[0,   3,    5,   65535,0,0,0,0,0,0,0,0,0,0,0,0],
[0,   3,    5,   65535,0,0,0,0,0,0,0,0,0,0,0,0],
[0,   3,    5,   65535,0,0,0,0,0,0,0,0,0,0,0,0],
[0,   3,    5,   65535,0,0,0,0,0,0,0,0,0,0,0,0],
[0,   3,    5,   65535,0,0,0,0,0,0,0,0,0,0,0,0],
[0,   3,    5,   65535,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
     
sbLimit = 30
sumNbal = 94