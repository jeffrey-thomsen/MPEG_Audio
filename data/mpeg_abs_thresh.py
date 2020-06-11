import numpy

"""
Tables D.1a to D.1f from the MPEG-1 Audio standard

D1a - LayerI@32kHz
D1b - LayerI@44.1kHz
D1c - LayerI@48kHz
D1d - LayerI@32kHz
D1e - LayerI@44.1kHz
D1f - LayerI@48kHz
"""
# Table D.1a. -- Frequencies, critical band rates and absolute threshold
# Table is valid for Layer I at a sampling rate of 32 kHz.

# LayerI@32kHz
# Index Number - Frequency (Hz) - Crit. Band Rate (z) - Absolute Thresh. (dB)

D1a = numpy.array([[1,   62.50,  0.617,33.44],
   [2,  125.00,  1.232,19.20],
   [3,  187.50,  1.842,13.87],
   [4,  250.00,  2.445,11.01],
   [5,  312.50,  3.037, 9.20],
   [6,  375.00,  3.618, 7.94],
   [7,  437.50,  4.185, 7.00],
   [8,  500.00,  4.736, 6.28],
   [9,  562.50,  5.272, 5.70],
  [10,  625.00,  5.789, 5.21],
  [11,  687.50,  6.289, 4.80],
  [12,  750.00,  6.770, 4.45],
  [13,  812.50,  7.233, 4.14],
  [14,  875.00,  7.677, 3.86],
  [15,  937.50,  8.103, 3.61],
  [16, 1000.00,  8.511, 3.37],
  [17, 1062.50,  8.901, 3.15],
  [18, 1125.00,  9.275, 2.93],
  [19, 1187.50,  9.632, 2.73],
  [20, 1250.00,  9.974, 2.53],
  [21, 1312.50, 10.301, 2.32],
  [22, 1375.00, 10.614, 2.12],
  [23, 1437.50, 10.913, 1.92],
  [24, 1500.00, 11.199, 1.71],
  [25, 1562.50, 11.474, 1.49],
  [26, 1625.00, 11.736, 1.27],
  [27, 1687.50, 11.988, 1.04],
  [28, 1750.00, 12.230, 0.80],
  [29, 1812.50, 12.461, 0.55],
  [30, 1875.00, 12.684, 0.29],
  [31, 1937.50, 12.898, 0.02],
 [32, 2000.00, 13.104, -0.25],
 [33, 2062.50, 13.302, -0.54],
 [34, 2125.00, 13.493, -0.83],
 [35, 2187.50, 13.678, -1.12],
 [36, 2250.00, 13.855, -1.43],
 [37, 2312.50, 14.027, -1.73],
 [38, 2375.00, 14.193, -2.04],
 [39, 2437.50, 14.354, -2.34],
 [40, 2500.00, 14.509, -2.64],
 [41, 2562.50, 14.660, -2.93],
 [42, 2625.00, 14.807, -3.22],
 [43, 2687.50, 14.949, -3.49],
 [44, 2750.00, 15.087, -3.74],
 [45, 2812.50, 15.221, -3.98],
 [46, 2875.00, 15.351, -4.20],
 [47, 2937.50, 15.478, -4.40],
 [48, 3000.00, 15.602, -4.57],
 [49, 3125.00, 15.841, -4.82],
 [50, 3250.00, 16.069, -4.96],
 [51, 3375.00, 16.287, -4.97],
 [52, 3500.00, 16.496, -4.86],
 [53, 3625.00, 16.697, -4.63],
 [54, 3750.00, 16.891, -4.29],
 [55, 3875.00, 17.078, -3.87],
 [56, 4000.00, 17.259, -3.39],
 [57, 4125.00, 17.434, -2.86],
 [58, 4250.00, 17.605, -2.31],
 [59, 4375.00, 17.770, -1.77],
 [60, 4500.00, 17.932, -1.24],
 [61, 4625.00, 18.089, -0.74],
 [62, 4750.00, 18.242, -0.29],
  [63, 4875.00, 18.392, 0.12],
  [64, 5000.00, 18.539, 0.48],
  [65, 5125.00, 18.682, 0.79],
  [66, 5250.00, 18.823, 1.06],
  [67, 5375.00, 18.960, 1.29],
  [68, 5500.00, 19.095, 1.49],
  [69, 5625.00, 19.226, 1.66],
  [70, 5750.00, 19.356, 1.81],
  [71, 5875.00, 19.482, 1.95],
  [72, 6000.00, 19.606, 2.08],
  [73, 6250.00, 19.847, 2.33],
  [74, 6500.00, 20.079, 2.59],
  [75, 6750.00, 20.300, 2.86],
  [76, 7000.00, 20.513, 3.17],
  [77, 7250.00, 20.717, 3.51],
  [78, 7500.00, 20.912, 3.89],
  [79, 7750.00, 21.098, 4.31],
  [80, 8000.00, 21.275, 4.79],
  [81, 8250.00, 21.445, 5.31],
  [82, 8500.00, 21.606, 5.88],
  [83, 8750.00, 21.760, 6.50],
  [84, 9000.00, 21.906, 7.19],
  [85, 9250.00, 22.046, 7.93],
  [86, 9500.00, 22.178, 8.75],
  [87, 9750.00, 22.304, 9.63],
[88, 10000.00, 22.424, 10.58],
[89, 10250.00, 22.538, 11.60],
[90, 10500.00, 22.646, 12.71],
[91, 10750.00, 22.749, 13.90],
[92, 11000.00, 22.847, 15.18],
[93, 11250.00, 22.941, 16.54],
[94, 11500.00, 23.030, 18.01],
[95, 11750.00, 23.114, 19.57],
[96, 12000.00, 23.195, 21.23],
[97, 12250.00, 23.272, 23.01],
[98, 12500.00, 23.345, 24.90],
[99, 12750.00, 23.415, 26.90],
[100, 13000.0, 23.482, 29.03],
[101, 13250.0, 23.546, 31.28],
[102, 13500.0, 23.607, 33.67],
[103, 13750.0, 23.666, 36.19],
[104, 14000.0, 23.722, 38.86],
[105, 14250.0, 23.775, 41.67],
[106, 14500.0, 23.827, 44.63],
[107, 14750.0, 23.876, 47.76],
[108, 15000.0, 23.923, 51.04]])

# Table D.1b. -- Frequencies, critical band rates and absolute thresbold
# Table is valid for Layer I at a sampling rate of 44.1 kHz.

# LayerI@44.1kHz
# Index Number - Frequency (Hz) - Crit. Band Rate (z) - Absolute Thresh. (dB)

D1b = numpy.array([[1, 86.13, 0.850, 25.87],
[2, 172.27, 1.694, 14.85],
[3, 258.40, 2.525, 10.72],
[4, 344.53, 3.337, 8.50],
[5, 430.66, 4.124, 7.10],
[6, 516.80, 4.882, 6.11],
[7, 602.93, 5.608, 5.37],
[8, 689.06, 6.301, 4.79],
[9, 775.20, 6.959, 4.32],
[10, 861.33, 7.581, 3.92],
[11, 947.46, 8.169, 3.57],
[12, 1033.59, 8.723, 3.25],
[13, 1119.73, 9.244, 2.95],
[14, 1205.86, 9.734, 2.67],
[15, 1291.99, 10.195, 2.39],
[16, 1378.13, 10.629, 2.11],
[17, 1464.26, 11.037, 1.83],
[18, 1550.39, 11.421, 1.53],
[19, 1636.52, 11.783, 1.23],
[20, 1722.66, 12.125, 0.90],
[21, 1808.79, 12.448, 0.56],
[22, 1894.92, 12.753, 0.21],
[23, 1981.05, 13.042, -0.17],
[24, 2067.19, 13.317, -0.56],
[25, 2153.32, 13.578, -0.96],
[26, 2239.45, 13.826, -1.38],
[27, 2325.59, 14.062, -1.79],
[28, 2411.72, 14.288, -2.21],
[29, 2497.85, 14.504, -2.63],
[30, 2583.98, 14.711, -3.03],
[31, 2670.12, 14.909, -3.41],
[32, 2756.25, 15.100, -3.77],
[33, 2842.38, 15.284, -4.09],
[34, 2928.52, 15.460, -4.37],
[35, 3014.65, 15.631, -4.60],
[36, 3100.78, 15.796, -4.78],
[37, 3186.91, 15.955, -4.91],
[38, 3273.05, 16.110, -4.97],
[39, 3359.18, 16.260, -4.98],
[40, 3445.31, 16.406, -4.92],
[41, 3531.45, 16.547, -4.81],
[42, 3617.58, 16.685, -4.65],
[43, 3703.71, 16.820, -4.43],
[44, 3789.84, 16.951, -4.17],
[45, 3875.98, 17.079, -3.87],
[46, 3962.11, 17.205, -3.54],
[47, 4048.24, 17.327, -3.19],
[48, 4134.38, 17.447, -2.82],
[49, 4306.64, 17.680, -2.06],
[50, 4478.91, 17.905, -1.32],
[51, 4651.17, 18.121, -0.64],
[52, 4823.44, 18.331, -0.04],
[53, 4995.70, 18.534, 0.47],
[54, 5167.97, 18.731, 0.89],
[55, 5340.23, 18.922, 1.23],
[56, 5512.50, 19.108, 1.51],
[57, 5684.77, 19.289, 1.74],
[58, 5857.03, 19.464, 1.93],
[59, 6029.30, 19.635, 2.11],
[60, 6201.56, 19.801, 2.28],
[61, 6373.83, 19.963, 2.46],
[62, 6546.09, 20.120, 2.63],
[63, 6718.36, 20.273, 2.82],
[64, 6890.63, 20.421, 3.03],
[65, 7062.89, 20.565, 3.25],
[66, 7235.16, 20.705, 3.49],
[67, 7407.42, 20.840, 3.74],
[68, 7579.69, 20.972, 4.02],
[69, 7751.95, 21.099, 4.32],
[70, 7924.22, 21.222, 4.64],
[71, 8096.48, 21.342, 4.98],
[72, 8268.75, 21.457, 5.35],
[73, 8613.28, 21.677, 6.15],
[74, 8957.81, 21.882, 7.07],
[75, 9302.34, 22.074, 8.10],
[76, 9646.88, 22.253, 9.25],
[77, 9991.41, 22.420, 10.54],
[78, 10335.94, 22.576, 11.97],
[79, 10680.47, 22.721, 13.56],
[80, 11025.00, 22.857, 15.31],
[81, 11369.53, 22.984, 17.23],
[82, 11714.06, 23.102, 19.34],
[83, 12058.59, 23.213, 21.64],
[84, 12403.13, 23.317, 24.15],
[85, 12747.66, 23.415, 26.88],
[86, 13092.19, 23.506, 29.84],
[87, 13436.72, 23.592, 33.05],
[88, 13781.25, 23.673, 36.52],
[89, 14125.78, 23.749, 40.25],
[90, 14470.31, 23.821, 44.27],
[91, 14814.84, 23.888, 48.59],
[92, 15159.38, 23.952, 53.22],
[93, 15503.91, 24.013, 58.18],
[94, 15848.44, 24.070, 63.49],
[95, 16192.97, 24.125, 68.00],
[96, 16537.50, 24.176, 68.00],
[97, 16882.03, 24.225, 68.00],
[98, 17226.56, 24.271, 68.00],
[99, 17571.09, 24.316, 68.00],
[100, 17915.63, 24.358, 68.00],
[101, 18260.16, 24.398, 68.00],
[102, 18604.69, 24.436, 68.00],
[103, 18949.22, 24.473, 68.00],
[104, 19293.75, 24.508, 68.00],
[105, 19638.28, 24.542, 68.00],
[106, 19982.81, 24.574, 68.00]])

# Table D.1c. Frequencies, critical band rates and absolute threshold
# Table is valid for Layer I at a sampling rate of 48 kHz.

# LayerI@48kHz
# Index Number - Frequency (Hz) - Crit. Band Rate (z) - Absolute Thresh. (dB)

D1c = numpy.array([[1, 93.75, 0.925, 24.17],
[2, 187.50, 1.842, 13.87],
[3, 281.25, 2.742, 10.01],
[4, 375.00, 3.618, 7.94],
[5, 468.75, 4.463, 6.62],
[6, 562.50, 5.272, 5.70],
[7, 656.25, 6.041, 5.00],
[8, 750.00, 6.770, 4.45],
[9, 843.75, 7.457, 4.00],
[10, 937.50, 8.103, 3.61],
[11, 1031.25, 8.708, 3.26],
[12, 1125.00, 9.275, 2.93],
[13, 1218.75, 9.805, 2.63],
[14, 1312.50, 10.301, 2.32],
[15, 1406.25, 10.765, 2.02],
[16, 1500.00, 11.199, 1.71],
[17, 1593.75, 11.606, 1.38],
[18, 1687.50, 11.988, 1.04],
[19, 1781.25, 12.347, 0.67],
[20, 1875.00, 12.684, 0.29],
[21, 1968.75, 13.002, -0.11],
[22, 2062.50, 13.302, -0.54],
[23, 2156.25, 13.586, -0.97],
[24, 2250.00, 13.855, -1.43],
[25, 2343.75, 14.111, -1.88],
[26, 2437.50, 14.354, -2.34],
[27, 2531.25, 14.585, -2.79],
[28, 2625.00, 14.807, -3.22],
[29, 2718.75, 15.018, -3.62],
[30, 2812.50, 15.221, -3.98],
[31, 2906.25, 15.415, -4.30],
[32, 3000.00, 15.602, -4.57],
[33, 3093.75, 15.783, -4.77],
[34, 3187.50, 15.956, -4.91],
[35, 3281.25, 16.124, -4.98],
[36, 3375.00, 16.287, -4.97],
[37, 3468.75, 16.445, -4.90],
[38, 3562.50, 16.598, -4.76],
[39, 3656.25, 16.746, -4.55],
[40, 3750.00, 16.891, -4.29],
[41, 3843.75, 17.032, -3.99],
[42, 3937.50, 17.169, -3.64],
[43, 4031.25, 17.303, -3.26],
[44, 4125.00, 17.434, -2.86],
[45, 4218.75, 17.563, -2.45],
[46, 4312.50, 17.688, -2.04],
[47, 4406.25, 17.811, -1.63],
[48, 4500.00, 17.932, -1.24],
[49, 4687.50, 18.166, -0.51],
[50, 4875.00, 18.392, 0.12],
[51, 5062.50, 18.611, 0.64],
[52, 5250.00, 18.823, 1.06],
[53, 5437.50, 19.028, 1.39],
[54, 5625.00, 19.226, 1.66],
[55, 5812.50, 19.419, 1.88],
[56, 6000.00, 19.606, 2.08],
[57, 6187.50, 19.788, 2.27],
[58, 6375.00, 19.964, 2.46],
[59, 6562.50, 20.135, 2.65],
[60, 6750.00, 20.300, 2.86],
[61, 6937.50, 20.461, 3.09],
[62, 7125.00, 20.616, 3.33],
[63, 7312.50, 20.766, 3.60],
[64, 7500.00, 20.912, 3.89],
[65, 7687.50, 21.052, 4.20],
[66, 7875.00, 21.188, 4.54],
[67, 8062.50, 21.318, 4.91],
[68, 8250.00, 21.445, 5.31],
[69, 8437.50, 21.567, 5.73],
[70, 8625.00, 21.684, 6.18],
[71, 8812.50, 21.797, 6.67],
[72, 9000.00, 21.906, 7.19],
[73, 9375.00, 22.113, 8.33],
[74, 9750.00, 22.304, 9.63],
[75, 10125.00, 22.482, 11.08],
[76, 10500.00, 22.646, 12.71],
[77, 10875.00, 22.799, 14.53],
[78, 11250.00, 22.941, 16.54],
[79, 11625.00, 23.072, 18.77],
[80, 12000.00, 23.195, 21.23],
[81, 12375.00, 23.309, 23.94],
[82, 12750.00, 23.415, 26.90],
[83, 13125.00, 23.515, 30.14],
[84, 13500.00, 23.607, 33.67],
[85, 13875.00, 23.694, 37.51],
[86, 14250.00, 23.775, 41.67],
[87, 14625.00, 23.852, 46.17],
[88, 15000.00, 23.923, 51.04],
[89, 15375.00, 23.991, 56.29],
[90, 15750.00, 24.054, 61.94],
[91, 16125.00, 24.114, 68.00],
[92, 16500.00, 24.171, 68.00],
[93, 16875.00, 24.224, 68.00],
[94, 17250.00, 24.275, 68.00],
[95, 17625.00, 24.322, 68.00],
[96, 18000.00, 24.368, 68.00],
[97, 18375.00, 24.411, 68.00],
[98, 18750.00, 24.452, 68.00],
[99, 19125.00, 24.491, 68.00],
[100, 19500.00, 24.528, 68.00],
[101, 19875.00, 24.564, 68.00],
[102, 20250.00, 24.597, 68.00]])

# Table D.1d. -- Frequencies, critical band rates and absolute threshold
# Table is valid for Layer II at a sampling rate of 32 kHz.

# LayerII@32kHz
# Index Number - Frequency (Hz) - Crit. Band Rate (z) - Absolute Thresh. (dB)

D1d = numpy.array([[1, 31.25, 0.309, 58.23],
[2, 62.50, 0.617, 33.44],
[3, 93.75, 0.925, 24.17],
[4, 125.00, 1.232, 19.20],
[5, 156.25, 1.538, 16.05],
[6, 187.50, 1.842, 13.87],
[7, 218.75, 2.145, 12.26],
[8, 250.00, 2.445, 11.01],
[9, 281.25, 2.742, 10.01],
[10, 312.50, 3.037, 9.20],
[11, 343.75, 3.329, 8.52],
[12, 375.00, 3.618, 7.94],
[13, 406.25, 3.903, 7.44],
[14, 437.50, 4.185, 7.00],
[15, 468.75, 4.463, 6.62],
[16, 500.00, 4.736, 6.28],
[17, 531.25, 5.006, 5.97],
[18, 562.50, 5.272, 5.70],
[19, 593.75, 5.533, 5.44],
[20, 625.00, 5.789, 5.21],
[21, 656.25, 6.041, 5.00],
[22, 687.50, 6.289, 4.80],
[23, 718.75, 6.532, 4.62],
[24, 750.00, 6.770, 4.45],
[25, 781.25, 7.004, 4.29],
[26, 812.50, 7.233, 4.14],
[27, 843.75, 7.457, 4.00],
[28, 875.00, 7.677, 3.86],
[29, 906.25, 7.892, 3.73],
[30, 937.50, 8.103, 3.61],
[31, 968.75, 8.309, 3.49],
[32, 1000.00, 8.511, 3.37],
[33, 1031.25, 8.708, 3.26],
[34, 1062.50, 8.901, 3.15],
[35, 1093.75, 9.090, 3.04],
[36, 1125.00, 9.275, 2.93],
[37, 1156.25, 9.456, 2.83],
[38, 1187.50, 9.632, 2.73],
[39, 1218.75, 9.805, 2.63],
[40, 1250.00, 9.974, 2.53],
[41, 1281.25, 10.139, 2.42],
[42, 1312.50, 10.301, 2.32],
[43, 1343.75, 10.459, 2.22],
[44, 1375.00, 10.614, 2.12],
[45, 1406.25, 10.765, 2.02],
[46, 1437.50, 10.913, 1.92],
[47, 1468.75, 11.058, 1.81],
[48, 1500.00, 11.199, 1.71],
[49, 1562.50, 11.474, 1.49],
[50, 1625.00, 11.736, 1.27],
[51, 1687.50, 11.988, 1.04],
[52, 1750.00, 12.230, 0.80],
[53, 1812.50, 12.461, 0.55],
[54, 1875.00, 12.684, 0.29],
[55, 1937.50, 12.898, 0.02],
[56, 2000.00, 13.104, -0.25],
[57, 2062.50, 13.302, -0.54],
[58, 2125.00, 13.493, -0.83],
[59, 2187.50, 13.678, -1.12],
[60, 2250.00, 13.855, -1.43],
[61, 2312.50, 14.027, -1.73],
[62, 2375.00, 14.193, -2.04],
[63, 2437.50, 14.354, -2.34],
[64, 2500.00, 14.509, -2.64],
[65, 2562.50, 14.660, -2.93],
[66, 2625.00, 14.807, -3.22],
[67, 2687.50, 14.949, -3.49],
[68, 2750.00, 15.087, -3.74],
[69, 2812.50, 15.221, -3.98],
[70, 2875.00, 15.351, -4.20],
[71, 2937.50, 15.478, -4.40],
[72, 3000.00, 15.602, -4.57],
[73, 3125.00, 15.841, -4.82],
[74, 3250.00, 16.069, -4.96],
[75, 3375.00, 16.287, -4.97],
[76, 3500.00, 16.496, -4.86],
[77, 3625.00, 16.697, -4.63],
[78, 3750.00, 16.891, -4.29],
[79, 3875.00, 17.078, -3.87],
[80, 4000.00, 17.259, -3.39],
[81, 4125.00, 17.434, -2.86],
[82, 4250.00, 17.605, -2.31],
[83, 4375.00, 17.770, -1.77],
[84, 4500.00, 17.932, -1.24],
[85, 4625.00, 18.089, -0.74],
[86, 4750.00, 18.242, -0.29],
[87, 4875.00, 18.392, 0.12],
[88, 5000.00, 18.539, 0.48],
[89, 5125.00, 18.682, 0.79],
[90, 5250.00, 18.823, 1.06],
[91, 5375.00, 18.960, 1.29],
[92, 5500.00, 19.095, 1.49],
[93, 5625.00, 19.226, 1.66],
[94, 5750.00, 19.356, 1.81],
[95, 5875.00, 19.482, 1.95],
[96, 6000.00, 19.606, 2.08],
[97, 6250.00, 19.847, 2.33],
[98, 6500.00, 20.079, 2.59],
[99, 6750.00, 20.300, 2.86],
[100, 7000.00, 20.513, 3.17],
[101, 7250.00, 20.717, 3.51],
[102, 7500.00, 20.912, 3.89],
[103, 7750.00, 21.098, 4.31],
[104, 8000.00, 21.275, 4.79],
[105, 8250.00, 21.445, 5.31],
[106, 8500.00, 21.606, 5.88],
[107, 8750.00, 21.760, 6.50],
[108, 9000.00, 21.906, 7.19],
[109, 9250.00, 22.046, 7.93],
[110, 9500.00, 22.178, 8.75],
[111, 9750.00, 22.304, 9.63],
[112, 10000.00, 22.424, 10.58],
[113, 10250.00, 22.538, 11.60],
[114, 10500.00, 22.646, 12.71],
[115, 10750.00, 22.749, 13.90],
[116, 11000.00, 22.847, 15.18],
[117, 11250.00, 22.941, 16.54],
[118, 11500.00, 23.030, 18.01],
[119, 11750.00, 23.114, 19.57],
[120, 12000.00, 23.195, 21.23],
[121, 12250.00, 23.272, 23.01],
[122, 12500.00, 23.345, 24.90],
[123, 12750.00, 23.415, 26.90],
[124, 13000.00, 23.482, 29.03],
[125, 13250.00, 23.546, 31.28],
[126, 13500.00, 23.607, 33.67],
[127, 13750.00, 23.666, 36.19],
[128, 14000.00, 23.722, 38.86],
[129, 14250.00, 23.775, 41.67],
[130, 14500.00, 23.827, 44.63],
[131, 14750.00, 23.876, 47.76],
[132, 15000.00, 23.923, 51.04]])

# Table D.1e. -- Frequencies, Critical Band Rates and Absolute Tbresbold
# Table is valid for Layer II at a sampling rate of 44,1 kHz.

# LayerII@44.1kHz
# Index Number - Frequency (Hz) - Crit. Band Rate (z) - Absolute Thresh. (dB)

D1e = numpy.array([[1, 43.07, 0.425, 45.05],
[2, 86.13, 0.850, 25.87],
[3, 129.20, 1.273, 18.70],
[4, 172.27, 1.694, 14.85],
[5, 215.33, 2.112, 12.41],
[6, 258.40, 2.525, 10.72],
[7, 301.46, 2.934, 9.47],
[8, 344.53, 3.337, 8.50],
[9, 387.60, 3.733, 7.73],
[10, 430.66, 4.124, 7.10],
[11, 473.73, 4.507, 6.56],
[12, 516.80, 4.882, 6.11],
[13, 559.86, 5.249, 5.72],
[14, 602.93, 5.608, 5.37],
[15, 646.00, 5.959, 5.07],
[16, 689.06, 6.301, 4.79],
[17, 732.13, 6.634, 4.55],
[18, 775.20, 6.959, 4.32],
[19, 818.26, 7.274, 4.11],
[20, 861.33, 7.581, 3.92],
[21, 904.39, 7.879, 3.74],
[22, 947.46, 8.169, 3.57],
[23, 990.53, 8.450, 3.40],
[24, 1033.59, 8.723, 3.25],
[25, 1076.66, 8.987, 3.10],
[26, 1119.73, 9.244, 2.95],
[27, 1162.79, 9.493, 2.81],
[28, 1205.86, 9.734, 2.67],
[29, 1248.93, 9.968, 2.53],
[30, 1291.99, 10.195, 2.39],
[31, 1335.06, 10.416, 2.25],
[32, 1378.13, 10.629, 2.11],
[33, 1421.19, 10.836, 1.97],
[34, 1464.26, 11.037, 1.83],
[35, 1507.32, 11.232, 1.68],
[36, 1550.39, 11.421, 1.53],
[37, 1593.46, 11.605, 1.38],
[38, 1636.52, 11.783, 1.23],
[39, 1679.59, 11.957, 1.07],
[40, 1722.66, 12.125, 0.90],
[41, 1765.72, 12.289, 0.74],
[42, 1808.79, 12.448, 0.56],
[43, 1851.86, 12.603, 0.39],
[44, 1894.92, 12.753, 0.21],
[45, 1937.99, 12.900, 0.02],
[46, 1981.05, 13.042, -0.17],
[47, 2024.12, 13.181, -0.36],
[48, 2067.19, 13.317, -0.56],
[49, 2153.32, 13.578, -0.96],
[50, 2239.45, 13.826, -1.38],
[51, 2325.59, 14.062, -1.79],
[52, 2411.72, 14.288, -2.21],
[53, 2497.85, 14.504, -2.63],
[54, 2583.98, 14.711, -3.03],
[55, 2670.12, 14.909, -3.41],
[56, 2756.25, 15.100, -3.77],
[57, 2842.38, 15.284, -4.09],
[58, 2928.52, 15.460, -4.37],
[59, 3014.65, 15.631, -4.60],
[60, 3100.78, 15.796, -4.78],
[61, 3186.91, 15.955, -4.91],
[62, 3273.05, 16.110, -4.97],
[63, 3359.18, 16.260, -4.98],
[64, 3445.31, 16.406, -4.92],
[65, 3531.45, 16.547, -4.81],
[66, 3617.58, 16.685, -4.65],
[67, 3703.71, 16.820, -4.43],
[68, 3789.84, 16.951, -4.17],
[69, 3875.98, 17.079, -3.87],
[70, 3962.11, 17.205, -3.54],
[71, 4048.24, 17.327, -3.19],
[72, 4134.38, 17.447, -2.82],
[73, 4306.64, 17.680, -2.06],
[74, 4478.91, 17.905, -1.32],
[75, 4651.17, 18.121, -0.64],
[76, 4823.44, 18.331, -0.04],
[77, 4995.70, 18.534, 0.47],
[78, 5167.97, 18.731, 0.89],
[79, 5340.23, 18.922, 1.23],
[80, 5512.50, 19.108, 1.51],
[81, 5684.77, 19.289, 1.74],
[82, 5857.03, 19.464, 1.93],
[83, 6029.30, 19.635, 2.11],
[84, 6201.56, 19.801, 2.28],
[85, 6373.83, 19.963, 2.46],
[86, 6546.09, 20.120, 2.63],
[87, 6718.36, 20.273, 2.82],
[88, 6890.63, 20.421, 3.03],
[89, 7062.89, 20.565, 3.25],
[90, 7235.16, 20.705, 3.49],
[91, 7407.42, 20.840, 3.74],
[92, 7579.69, 20.972, 4.02],
[93, 7751.95, 21.099, 4.32],
[94, 7924.22, 21.222, 4.64],
[95, 8096.48, 21.342, 4.98],
[96, 8268.75, 21.457, 5.35],
[97, 8613.28, 21.677, 6.15],
[98, 8957.81, 21.882, 7.07],
[99, 9302.34, 22.074, 8.10],
[100, 9646.88, 22.253, 9.25],
[101, 9991.41, 22.420, 10.54],
[102, 10335.94, 22.576, 11.97],
[103, 10680.47, 22.721, 13.56],
[104, 11025.00, 22.857, 15.31],
[105, 11369.53, 22.984, 17.23],
[106, 11714.06, 23.102, 19.34],
[107, 12058.59, 23.213, 21.64],
[108, 12403.13, 23.317, 24.15],
[109, 12747.66, 23.415, 26.88],
[110, 13092.19, 23.506, 29.84],
[111, 13436.72, 23.592, 33.05],
[112, 13781.25, 23.673, 36.52],
[113, 14125.78, 23.749, 40.25],
[114, 14470.31, 23.821, 44.27],
[115, 14814.84, 23.888, 48.59],
[116, 15159.38, 23.952, 53.22],
[117, 15503.91, 24.013, 58.18],
[118, 15848.44, 24.070, 63.49],
[119, 16192.97, 24.125, 68.00],
[120, 16537.50, 24.176, 68.00],
[121, 16882.03, 24.225, 68.00],
[122, 17226.56, 24.271, 68.00],
[123, 17571.09, 24.316, 68.00],
[124, 17915.63, 24.358, 68.00],
[125, 18260.16, 24.398, 68.00],
[126, 18604.69, 24.436, 68.00],
[127, 18949.22, 24.473, 68.00],
[128, 19293.75, 24.508, 68.00],
[129, 19638.28, 24.542, 68.00],
[130, 19982.81, 24.574, 68.00]])

# Table D.1f. -- Frequencies, critical band rates and absolute threshold
# Table is valid for Layer II at a sampling rate of 48 kHz.

# LayerII@48kHz
# Index Number - Frequency (Hz) - Crit. Band Rate (z) - Absolute Thresh. (dB)

D1f = numpy.array([[1, 46.88, 0.463, 42.10],
[2, 93.75, 0.925, 24.17],
[3, 140.63, 1.385, 17.47],
[4, 187.50, 1.842, 13.87],
[5, 234.38, 2.295, 11.60],
[6, 281.25, 2.742, 10.01],
[7, 328.13, 3.184, 8.84],
[8, 375.00, 3.618, 7.94],
[9, 421.88, 4.045, 7.22],
[10, 468.75, 4.463, 6.62],
[11, 515.63, 4.872, 6.12],
[12, 562.50, 5.272, 5.70],
[13, 609.38, 5.661, 5.33],
[14, 656.25, 6.041, 5.00],
[15, 703.13, 6.411, 4.71],
[16, 750.00, 6.770, 4.45],
[17, 796.88, 7.119, 4.21],
[18, 843.75, 7.457, 4.00],
[19, 890.63, 7.785, 3.79],
[20, 937.50, 8.103, 3.61],
[21, 984.38, 8.410, 3.43],
[22, 1031.25, 8.708, 3.26],
[23, 1078.13, 8.996, 3.09],
[24, 1125.00, 9.275, 2.93],
[25, 1171.88, 9.544, 2.78],
[26, 1218.75, 9.805, 2.63],
[27, 1265.63, 10.057, 2.47],
[28, 1312.50, 10.301, 2.32],
[29, 1359.38, 10.537, 2.17],
[30, 1406.25, 10.765, 2.02],
[31, 1453.13, 10.986, 1.86],
[32, 1500.00, 11.199, 1.71],
[33, 1546.88, 11.406, 1.55],
[34, 1593.75, 11.606, 1.38],
[35, 1640.63, 11.800, 1.21],
[36, 1687.50, 11.988, 1.04],
[37, 1734.38, 12.170, 0.86],
[38, 1781.25, 12.347, 0.67],
[39, 1828.13, 12.518, 0.49],
[40, 1875.00, 12.684, 0.29],
[41, 1921.88, 12.845, 0.09],
[42, 1968.75, 13.002, -0.11],
[43, 2015.63, 13.154, -0.32],
[44, 2062.50, 13.302, -0.54],
[45, 2109.38, 13.446, -0.75],
[46, 2156.25, 13.586, -0.97],
[47, 2203.13, 13.723, -1.20],
[48, 2250.00, 13.855, -1.43],
[49, 2343.75, 14.111, -1.88],
[50, 2437.50, 14.354, -2.34],
[51, 2531.25, 14.585, -2.79],
[52, 2625.00, 14.807, -3.22],
[53, 2718.75, 15.018, -3.62],
[54, 2812.50, 15.221, -3.98],
[55, 2906.25, 15.415, -4.30],
[56, 3000.00, 15.602, -4.57],
[57, 3093.75, 15.783, -4.77],
[58, 3187.50, 15.956, -4.91],
[59, 3281.25, 16.124, -4.98],
[60, 3375.00, 16.287, -4.97],
[61, 3468.75, 16.445, -4.90],
[62, 3562.50, 16.598, -4.76],
[63, 3656.25, 16.746, -4.55],
[64, 3750.00, 16.891, -4.29],
[65, 3843.75, 17.032, -3.99],
[66, 3937.50, 17.169, -3.64],
[67, 4031.25, 17.303, -3.26],
[68, 4125.00, 17.434, -2.86],
[69, 4218.75, 17.563, -2.45],
[70, 4312.50, 17.688, -2.04],
[71, 4406.25, 17.811, -1.63],
[72, 4500.00, 17.932, -1.24],
[73, 4687.50, 18.166, -0.51],
[74, 4875.00, 18.392, 0.12],
[75, 5062.50, 18.611, 0.64],
[76, 5250.00, 18.823, 1.06],
[77, 5437.50, 19.028, 1.39],
[78, 5625.00, 19.226, 1.66],
[79, 5812.50, 19.419, 1.88],
[80, 6000.00, 19.606, 2.08],
[81, 6187.50, 19.788, 2.27],
[82, 6375.00, 19.964, 2.46],
[83, 6562.50, 20.135, 2.65],
[84, 6750.00, 20.300, 2.86],
[85, 6937.50, 20.461, 3.09],
[86, 7125.00, 20.616, 3.33],
[87, 7312.50, 20.766, 3.60],
[88, 7500.00, 20.912, 3.89],
[89, 7687.50, 21.052, 4.20],
[90, 7875.00, 21.188, 4.54],
[91, 8062.50, 21.318, 4.91],
[92, 8250.00, 21.445, 5.31],
[93, 8437.50, 21.567, 5.73],
[94, 8625.00, 21.684, 6.18],
[95, 8812.50, 21.797, 6.67],
[96, 9000.00, 21.906, 7.19],
[97, 9375.00, 22.113, 8.33],
[98, 9750.00, 22.304, 9.63],
[99, 10125.00, 22.482, 11.08],
[100, 10500.00, 22.646, 12.71],
[101, 10875.00, 22.799, 14.53],
[102, 11250.00, 22.941, 16.54],
[103, 11625.00, 23.072, 18.77],
[104, 12000.00, 23.195, 21.23],
[105, 12375.00, 23.309, 23.94],
[106, 12750.00, 23.415, 26.90],
[107, 13125.00, 23.515, 30.14],
[108, 13500.00, 23.607, 33.67],
[109, 13875.00, 23.694, 37.51],
[110, 14250.00, 23.775, 41.67],
[111, 14625.00, 23.852, 46.17],
[112, 15000.00, 23.923, 51.04],
[113, 15375.00, 23.991, 56.29],
[114, 15750.00, 24.054, 61.94],
[115, 16125.00, 24.114, 68.00],
[116, 16500.00, 24.171, 68.00],
[117, 16875.00, 24.224, 68.00],
[118, 17250.00, 24.275, 68.00],
[119, 17625.00, 24.322, 68.00],
[120, 18000.00, 24.368, 68.00],
[121, 18375.00, 24.411, 68.00],
[122, 18750.00, 24.452, 68.00],
[123, 19125.00, 24.491, 68.00],
[124, 19500.00, 24.528, 68.00],
[125, 19875.00, 24.564, 68.00],
[126, 20250.00, 24.597, 68.00]])