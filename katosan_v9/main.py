import numpy as np
import os
import autoreservoir

# Setting random seed
seed = 42
# Setting parameters
# Input data
teacher = 'n00000018.wav'
testing_data = 'n00000011.wav'
testing_data_abnormal = 'a00000000.wav'
# Setting FFT
NFFT = 256
noverlap = 255
# Setting input scaling and bias
bias = 0.001
inputScaling = 3
# Setting teacher scaling and bias
teacherScaling = 1.12
teacherShift = -0.7
# Setting esn parameters
InputDimension = 130
OutputDimension = 129
ReservoirSize = 800
Connectivity = 0.05
SpectralRadius = 1.35
noise = 0.001
# LeakingRate = 0.01
# delay = 5


Sweep_LeakingRate = [0.01]
Sweep_delay = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
               42, 43, 44, 45, 46, 47, 48, 49, 50]
# Sweep_SpectralRadius = [1.35]
# Sweep_ReservoirSize = [800]
# Sweep_SpectralRadius = [0.10, 0.50, 1.00, 1.35, 2.00, 2.50]
# Sweep_ReservoirSize = [10, 50, 100, 400, 800, 1200]
p = 0
q = 0
f_measure_90 = np.zeros((len(Sweep_delay), len(Sweep_LeakingRate)))
f_measure_95 = np.zeros((len(Sweep_delay), len(Sweep_LeakingRate)))
# f_measure_90 = np.zeros((len(Sweep_SpectralRadius), len(Sweep_ReservoirSize)))
# f_measure_95 = np.zeros((len(Sweep_SpectralRadius), len(Sweep_ReservoirSize)))
dir = "np.random.seed(" + str(seed) + ")"
os.mkdir(dir)
# for ReservoirSize in Sweep_ReservoirSize:
#     for SpectralRadius in Sweep_SpectralRadius:
#         autoreservoir.autoreservoir(seed, teacher, testing_data, testing_data_abnormal, NFFT,
#                                     noverlap, bias, inputScaling, teacherScaling, teacherShift,
#                                     InputDimension, OutputDimension, ReservoirSize, Connectivity,
#                                     SpectralRadius, noise, LeakingRate, delay, dir)


for LeakingRate in Sweep_LeakingRate:
    p = 0
    for delay in Sweep_delay:
        f = autoreservoir.autoreservoir(seed, teacher, testing_data, testing_data_abnormal, NFFT,
                                    noverlap, bias, inputScaling, teacherScaling, teacherShift,
                                    InputDimension, OutputDimension, ReservoirSize, Connectivity,
                                    SpectralRadius, noise, LeakingRate, delay, dir)
        f_measure_90[p][q] = f[0]
        f_measure_95[p][q] = f[1]
        p = p + 1
    q = q + 1

np.savetxt(dir + "/f_measure_90.csv", f_measure_90, delimiter=",")
np.savetxt(dir + "/f_measure_95.csv", f_measure_95, delimiter=",")
