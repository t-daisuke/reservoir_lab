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
    mesh_code_list=[533945054,533945262,533945171,533945181]
    
    for i,mesh_code in enumerate(mesh_code_list):
        gml = get_mesh_list(mesh_code, gmom, gnl)
        
        ###Figure Print
        
        raw_data = create_subset_from_data_and_mesh_list(df,gml)
        real_data = extract_data_every_n(raw_data,60)
        show_print_array(real_data[0:1,0:].T,f"{mesh_code}real data-1",2*i,figsize=(20,5))
        show_print_array(real_data.T,f"{mesh_code}real data",2*i+1,figsize=(20,5))
        
        # repeated_data = repeat_data_columns(real_data,60)
        # show_print_array(repeated_data[0:,0:144*3].T,f"{mesh_code}subset of repeated data",2*i+1)
    
    plt.show()
    
    