# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

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
    
    print("Test 46 meshcode matrix")
    print(len(cut_mlist(get_raw_mesh_array(df),[46])))
    print()
    
    #1.2 mesh code is 533954911
    print("Test the correspondence between a matrix and a mesh code.")
    tmp = (1,2)
    print(tmp)
    A = get_matrix_of_mesh()
    a_tmp = A["mat"][tmp[0],tmp[1]]
    print(a_tmp)
    a2_tmp = A["dic"][a_tmp]
    print(a2_tmp)
    print()
    
    print("Test get mesh near O mesh")
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(9)
    print(get_mesh_list(533945002,gmom,gnl))
    print()

    print("Test count mesh and resrvoir-ble mesh")
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(9)
    dma = get_raw_mesh_array(df) #おおすぎた
    
    print(len(dma))
    Rl = get_R_list(dma, gmom, gnl)
    print(len(Rl))
    
    print("Smesh == 45に固定")
    dma = cut_mlist(dma,[45])
    print(len(dma))
    Rl = get_R_list(dma, gmom, gnl)
    print(len(Rl))
    print()
    
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(9)
    dma = get_raw_mesh_array(df) #おおすぎた
    # dma = cut_mlist(dma,[45]) #Smesh == 45に固定
    print(len(dma))
    Rl = get_R_list(dma, gmom, gnl)
    print(len(Rl))
    grld = get_Rlist_dict(Rl,gmom,gnl)
    print(len(grld.keys()))
    tmp=533934493
    print(tmp in Rl)
    print(grld[tmp])
    # tmp=533934482
    # print(tmp in Rl)
    # print(grld[tmp]) #KeyError Reservoirじゃないから