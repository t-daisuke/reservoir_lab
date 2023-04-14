# import
import pandas as pd
import os
from train_func import *
from print_func import *

if __name__ == '__main__':
    # Variable

    df_path = "./df/"
    path = './KDDI/'

    if os.path.isfile(df_path+"df.csv"):
        df = pd.read_csv(df_path+"df.csv")
    else:
        print("NO DF FILE")
    
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(9) #たぶんinSize
    dma = get_raw_mesh_array(df) #おおすぎた

    Rlist = get_R_list(dma,gmom,gnl)
    mesh_code = 533945774
    gml = get_mesh_list(mesh_code, gmom, gnl)
    raw_data = create_subset_from_data_and_mesh_list(df,gml)
    show_print_array(raw_data[0:9,0:].T)
    
    