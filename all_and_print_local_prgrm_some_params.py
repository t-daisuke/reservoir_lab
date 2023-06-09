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
    res_params = (-9.5, 1, 100, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, 1.0)
    distance = 30
    Smesh_list = [45]
    center_mesh_mat = (24,35) #ikebukuro 533945774 9*9
    
    
    # d_list=[2,4,8,16,32,64]
    d_list=[16,32,64]
    is_up = False
    cnct_list = [0.001, 0.01, 0.1, 1.0]
    sc_list = [-1, -3, -5, -7, -9, -11, -13]
    
    all_program_start_time = time.perf_counter()
    
    for d in d_list:       
        for cone in cnct_list:
            for sc in sc_list:
                res_params = (sc, 1, 1000, 0.75, 9, 9,
                    24*60, 3*24*60, 2*24*60-60+1,
                    1e-8, 2, cone)
                distance = d
                
                main_path = './all_prgrm_output/'
                ###################Reservoir
                for_time = time.perf_counter()
                
                # start_time = time.perf_counter() #Start
                # print("Start Train")

                # # create_local_area_trained_data(main_path,res_params,df,Smesh_list,is_update = is_up)
                # #TODO repeat_numはファイル名やres_paramsに含めていません。
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

                start_time = time.perf_counter()  # Start
                print("Start NCO")

                create_local_nco_test_data(main_path, res_params, distance, df, Smesh_list)

                print("Save NCO Test Data:" )
                display_time(time.perf_counter() - start_time)
                
                
                #####################Print
                main_path = './'
                
                start_time = time.perf_counter()  # Start
                print("Start mse create")
                
                create_mse_maps(main_path, saved_test_path, res_params, distance, df, Smesh_list)
                
                print("Save mse create Test Data:" )
                display_time(time.perf_counter() - start_time)
                
                start_time = time.perf_counter()  # Start
                print("Start copy gr data")
                
                copy_gr_data(main_path, saved_test_path, res_params, distance, Smesh_list, center_mesh_mat)
                
                print("Save copy gr data Test Data:" )
                display_time(time.perf_counter() - start_time)
                
                print("For Time:")
                display_time(time.perf_counter() - for_time)
                print(str(res_params) + "d:" + str(distance)) 
                print()
    
    print("All Time:")
    display_time(time.perf_counter() - all_program_start_time)