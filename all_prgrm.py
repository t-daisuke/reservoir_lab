# -*- coding: utf-8 -*-
# Ver0324

# import
import pandas as pd
import os
import time

from train_func import *
from test_func import *
from print_func import *

if __name__ == '__main__':
    # Variable

    df_path = "./df/"
    main_path = './all_prgrm_output/'

    if os.path.isfile(df_path+"df.csv"):
        df = pd.read_csv(df_path+"df.csv")
    else:
        print("NO DF FILE")

    # (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
    #  initLen, trainLen, testLen,
    #  reg, seed_num, conectivity) = res_params
    res_params = (-13, 1, 100, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, 10**(-3))
    distance = 32
    Smesh_list = [45]
    is_up = False
    
    for_time = time.perf_counter()
    
    # start_time = time.perf_counter() #Start
    # print("Start Train")
    # create_local_area_trained_data_with_repeated_real_data(main_path,res_params,df,Smesh_list,repeat_num=60,is_update = is_up)
    # print("Save Train Data:")
    # display_time(time.perf_counter() - start_time)

    start_time = time.perf_counter()  # Start
    print("Start GR")
    print(str(res_params) + "d:" + str(distance)) 
    print()
    create_local_gr_test_data(main_path, res_params, distance, df, Smesh_list)
    print("Save GR Test Data:" )
    display_time(time.perf_counter() - start_time)

    # start_time = time.perf_counter()  # Start
    # print("Start NCO")
    # create_local_nco_test_data(main_path, res_params, distance, df, Smesh_list)
    # print("Save NCO Test Data:" )
    # display_time(time.perf_counter() - start_time)
    
    print("For Time:")
    display_time(time.perf_counter() - for_time)
    print(str(res_params) + "d:" + str(distance)) 
    print()