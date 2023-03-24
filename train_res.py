# -*- coding: utf-8 -*-
# Ver0324

# import
import pandas as pd
import os
import time

from train_func import *

if __name__ == '__main__':
    # Variable

    df_path = "./KDDI/df/"
    path = './KDDI/'

    if os.path.isfile(df_path+"df.csv"):
        df = pd.read_csv(df_path+"df.csv")
    else:
        print("NO DF FILE")

    #(leakingRate, resSize, spectralRadius, inSize, outSize,
    # initLen, trainLen, testLen, reg, seed_num) = res_params
    res_params = (1, 1000, 0.75, 9, 9, 24*60, 3*24*60, 2*24*60-60+1, 1e-8, 2)
    expIndex = -9.5
    is_up = False


    start_time = time.perf_counter() #Start
    print("Start:"+ str(start_time))

    create_Tdata(path,res_params,df,expIndex,is_update = is_up)

    end_time = time.perf_counter() #End
    print("Save Train Data:"+ str(end_time - start_time) + "s")