# -*- coding: utf-8 -*-
# Ver0324

# import
import pandas as pd
import os
import time

from train_func import *
from test_func import *
from print_func import *
from katosan_func import *

if __name__ == '__main__':
    # Variable

    df_path = "./df/"
    main_path = './all_prgrm_output/'
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
                  1e-8, 2, 1.0)
    distance = 30
    is_up = False
    Smesh_list = [45]
    center_mesh_mat = (24,35) #ikebukuro 533945774 9*9
    
    all_program_start_time = time.perf_counter()
    
    for cone in [0.001, 0.01, 0.1, 1.0]:
        for sc in [-1, -3, -5, -7, -9]:
            # [-1, -3, -5, -7, -9]
            res_params = (sc, 1, 1000, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, cone)
            main_path = './all_prgrm_output/'
            ###################Reservoir
            for_time = time.perf_counter()
            
            start_time = time.perf_counter() #Start
            print("Start Train")

            create_local_area_trained_data(main_path,res_params,df,Smesh_list,is_update = is_up)

            print("Save Train Data:"+ str(time.perf_counter() - start_time) + "s")

            start_time = time.perf_counter()  # Start
            print("Start GR")

            create_local_gr_test_data(main_path, res_params, distance, df, Smesh_list)

            print("Save GR Test Data:" + str(time.perf_counter() - start_time) + "s")

            start_time = time.perf_counter()  # Start
            print("Start NCO")

            create_local_nco_test_data(main_path, res_params, distance, df, Smesh_list)

            print("Save NCO Test Data:" + str(time.perf_counter() - start_time) + "s")
            
            print(str(res_params) + "for Time:" + str(time.perf_counter() - for_time) + "s")
            
            #####################Print
            main_path = './'
            
            start_time = time.perf_counter()  # Start
            print("Start mse create")
            
            create_mse_maps(main_path, saved_test_path, res_params, distance, df, Smesh_list)
            
            print("Save mse create Test Data:" + str(time.perf_counter() - start_time) + "s")
            
            start_time = time.perf_counter()  # Start
            print("Start copy gr data")
            
            copy_gr_data(main_path, saved_test_path, res_params, distance, Smesh_list, center_mesh_mat)
            
            print("Save copy gr data Test Data:" + str(time.perf_counter() - start_time) + "s")
            
            print(str(res_params) + "for Time:" + str(time.perf_counter() - for_time) + "s")
    
    print("All Time:" + str(time.perf_counter() - all_program_start_time) + "s")