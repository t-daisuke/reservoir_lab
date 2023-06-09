# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

import os
import time
import pdb
from train_func import *

def draw_coordinates(coordinate_list, mesh_code_list, size):
    # Initialize the map
    map_ = [[' ']*size[1] for _ in range(size[0])]
    
    # Draw the coordinates and their corresponding mesh code
    for coordinate, mesh_code in zip(coordinate_list, mesh_code_list):
        i, j = coordinate
        #TODO Smesh45の時のみの仕様
        i = i-20
        j = j-20
        map_[i][j] = str(coordinate) + '(' + str(mesh_code) + ')'

    # Print the map
    for row in map_:
        print(' '.join(row))

if __name__ == '__main__':
    # Variable

    df_path = "./df/"
    path = './KDDI/'

    if os.path.isfile(df_path+"df.csv"):
        df = pd.read_csv(df_path+"df.csv")
    else:
        print("NO DF FILE")
    
    # pdb.set_trace()
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
    print("Shape")
    print(A["mat"].shape)
    print()
    
    print("Test get mesh near O mesh")
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(9)
    print(get_mesh_list(533945002,gmom,gnl))
    print()

    print("Test count mesh and resrvoir-able mesh")
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(9)
    dma = get_raw_mesh_array(df)
    
    print(len(dma))
    Rl = get_R_list(dma, gmom, gnl)
    print(len(Rl))
    
    print("Smesh == 45に固定")
    dma = cut_mlist(dma,[45])
    print(len(dma))
    Rl = get_R_list(dma, gmom, gnl)
    print(len(Rl))
    print("A[mat].shape")
    dma_set = set(dma)
    
    # Initialize empty list to store the indices
    indices_index = []
    indices_mesh = []
    
    # Iterate over the mesh_matrix
    for i in range(A["mat"].shape[0]):
        for j in range(A["mat"].shape[1]):
            # If the mesh code is in dma, store the index
            if A["mat"][i, j] in dma_set:
                indices_index.append((i, j))
                indices_mesh.append(str(A["mat"][i, j])[4:])
                
    # Test
    coordinate_list = indices_index
    mesh_code_list = indices_mesh
    size = (20, 20)
    draw_coordinates(coordinate_list, mesh_code_list, size)
    
    
    
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(9)
    dma = get_raw_mesh_array(df) 
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