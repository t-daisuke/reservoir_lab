# -*- coding: utf-8 -*-
# Ver0324

# import
import time
from print_func import *

if __name__ == '__main__':
     # Variable

    main_path = './'
    saved_test_path = './all_prgrm_output/'

    # (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
    #  initLen, trainLen, testLen,
    #  reg, seed_num, conectivity) = res_params
    res_params = (-9.5, 1, 1000, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, 0.001)
    distance = 30
    Smesh_list = [45]
    center_mesh_mat = (24,35) #ikebukuro 533945774 9*9

    start_time = time.perf_counter()  # Start
    print("Start Print")
    for cone in [0.001, 0.01, 0.1, 1.0]:
        for sc in [-9]:
            # [-1, -3, -5, -7]
            res_params = (sc, 1, 1000, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, cone)
            copy_gr_data(main_path, saved_test_path, res_params, distance, Smesh_list, center_mesh_mat)
    print("Copy:" + str(time.perf_counter() - start_time) + "s")