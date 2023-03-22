import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg 
import random
import os
import time
import gc
import datetime

len = 4320
res_size = 1000
dim = 9
for_size = 100

# Out = np.random.rand(len,dim)
# Y = np.random.rand(len,dim)
# W = np.random.rand(res_size,res_size)
# Win = np.random.rand(dim,res_size)
# Wout = np.random.rand(dim,res_size)
# X = np.random.rand(res_size,len)
X = np.random.rand(res_size*3,len)

# np.savez_compressed(Tdata_file,Win=Win,W=W,X=X,Wout=Wout,x=x,Data=Data)
# np.savez_compressed(test_file, Y=Y, UU=UU, XX=XX, Out=Out, trainO=trainO)


start_time = time.perf_counter() #Start
print("Start:"+ str(start_time))

for i in range(for_size):
    # np.savez_compressed("hogehoge",Out=Out,X=X,Y=Y,Win=Win,W=W,Wout=Wout)
    np.savez_compressed("hogehoge",X=X)

end_time = time.perf_counter() #End
print("Save Train Data:"+ str(end_time - start_time) + "s")
