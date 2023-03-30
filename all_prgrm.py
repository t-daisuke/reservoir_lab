# -*- coding: utf-8 -*-
# Ver0324

# import
import pandas as pd
import os
import time

from train_func import *
from test_func import *

if __name__ == '__main__':
    # Variable

    df_path = "./KDDI/df/"
    main_path = './all_prgrm_output/'

    if os.path.isfile(df_path+"df.csv"):
        df = pd.read_csv(df_path+"df.csv")
    else:
        print("NO DF FILE")

    # (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
    #  initLen, trainLen, testLen,
    #  reg, seed_num, conectivity) = res_params
    res_params = (-9.5, 1, 1000, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, 0.1)
    distance = 30
    is_up = False
    
    all_program_start_time = time.perf_counter()
    start_time = time.perf_counter() #Start
    print("Start Train")

    create_trained_data(main_path,res_params,df,is_update = is_up)

    print("Save Train Data:"+ str(time.perf_counter() - start_time) + "s")

    start_time = time.perf_counter()  # Start
    print("Start GR")

    create_gr_test_data(main_path, res_params, distance, df)

    print("Save GR Test Data:" + str(time.perf_counter() - start_time) + "s")

    start_time = time.perf_counter()  # Start
    print("Start NCO")

    create_nco_test_data(main_path, res_params, distance, df)

    print("Save NCO Test Data:" + str(time.perf_counter() - start_time) + "s")
