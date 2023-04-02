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

    df_path = "./df/"
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
                  1e-8, 2, 0.001)
    distance = 30
    is_up = False
    Smesh_list = [45]
    
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(res_params[4])
    print("Local area trained at " + str(Smesh_list))
    all_dma = get_raw_mesh_array(df)
    dma = cut_mlist(all_dma,Smesh_list)
    Rl = get_R_list(dma, gmom, gnl)
    grld = get_Rlist_dict(Rl,gmom,gnl)
    
    print(len(dma))
    print(dma[0:10])
    print("Data mesh:" + str(len(dma)))
    print("Reservoir mesh:" + str(len(Rl)))
    print(len(grld.keys()))
    
    
