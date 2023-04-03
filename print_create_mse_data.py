# -*- coding: utf-8 -*-
# Ver0324

# import
import pandas as pd
import os
import time
from print_func import *

if __name__ == '__main__':
     # Variable

    df_path = "./df/"
    main_path = './'
    saved_test_path = './all_prgrm_output/'

    if os.path.isfile(df_path+"df.csv"):
        df = pd.read_csv(df_path+"df.csv")
    else:
        print("NO DF FILE")

    # (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
    #  initLen, trainLen, testLen,
    #  reg, seed_num, conectivity) = res_params
    res_params = (-9.5, 1, 1000, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, 0.001)
    distance = 30
    Smesh_list = [45]

    start_time = time.perf_counter()  # Start
    print("Start Print")
    #create_mse_maps(main_path, saved_test_path, res_params, distance, df, Smesh_list)
    for cone in [0.001, 0.01, 0.1, 1.0]:
        for sc in [-1, -3, -5, -7]:
            res_params = (sc, 1, 1000, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, cone)
            create_mse_maps(main_path, saved_test_path, res_params, distance, df, Smesh_list)
    end_time = time.perf_counter()  # End
    print("Saved Print:" + str(end_time - start_time) + "s")