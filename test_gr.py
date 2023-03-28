# -*- coding: utf-8 -*-
# Ver0324

# import
import pandas as pd
import os
import time

from test_func import *

if __name__ == '__main__':
    # Variable

    df_path = "./KDDI/df/"
    main_path = './'

    if os.path.isfile(df_path+"df.csv"):
        df = pd.read_csv(df_path+"df.csv")
    else:
        print("NO DF FILE")

    # (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
    #  initLen, trainLen, testLen,
    #  reg, seed_num, conectivity) = res_params
    res_params = (-9.5, 1, 1000, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, 0.01)
    is_up = False

    start_time = time.perf_counter()  # Start
    print("Start:" + str(start_time))

    # create_trained_data(main_path, res_params, df, is_update=is_up)
    # path = './KDDI/'
    # res_params = (1, 1000, 0.75, 9, 9, 24*60, 3*24*60, 2*24*60-60+1, 1e-8,1)
    # expIndex = -9.5
    # dis = 30
    # print("Res Params")
    # print(res_params)
    # print("Exp Index")
    # print(expIndex)
    # print("Dis")
    # print(dis)
    # print(path)
    # gmom = get_matrix_of_mesh()
    # gnl = get_n_list(res_params[3])
    # dma = get_raw_mesh_array(df) #すべてのメッシュ
    # # dma = cut_mlist(dma,[45]) #Smesh == 45に固定
    # print("Data mesh:" + str(len(dma)))
    # Rl = get_R_list(dma, gmom, gnl)
    # print("Reservoir mesh:" + str(len(Rl)))
    # grld = get_Rlist_dict(Rl,gmom,gnl)

    test_GR(path, res_params, expIndex, dis, grld)

    end_time = time.perf_counter()  # End
    print("Save Train Data:" + str(end_time - start_time) + "s")
