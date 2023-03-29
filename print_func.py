# -*- coding: utf-8 -*-
# Ver0324

# import
import pandas as pd
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy import linalg

import os
import time

from train_func import *
from test_func import *

#INITIALIZE
#Fig Params
plt.rcParams["font.size"] =15
Figsize = (40,20)
plt.rcParams['image.cmap'] = 'bwr'

def load_geo_test_data(main_path, res_params, Distance, mesh_code):
    test_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        test_path += "-" + str(res_params[prm_i])
    test_path += "/"
    test_path += str(Distance)+"step-test/"

    # read mesh_code
    tested_file = test_path + str(mesh_code)
    if os.path.isfile(tested_file+".npz"):
        test_data = np.load(tested_file+".npz")
        (Y,UU,XX,Out,trainO) = (test_data["Y"],test_data["UU"],test_data["XX"],test_data["Out"],test_data["trainO"])
        return (Y,UU,XX,Out,trainO)
    else:
        print(test_path)
        print("ERROR at " + str(mesh_code))
  
def load_nco_test_data(main_path, res_params, Distance, mesh_code):
    test_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        test_path += "-" + str(res_params[prm_i])
    test_path += "/"
    test_path += str(Distance)+"step-test-nco/"

    # read mesh_code
    tested_file = test_path + str(mesh_code)
    if os.path.isfile(tested_file+".npz"):
        test_data = np.load(tested_file+".npz")
        (Y,UU,XX,Out,trainO) = (test_data["Y"],test_data["UU"],test_data["XX"],test_data["Out"],test_data["trainO"])
        return (Y,UU,XX,Out,trainO)
    else:
        print(test_path)
        print("ERROR at " + str(mesh_code))