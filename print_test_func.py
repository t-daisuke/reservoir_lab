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
    mesh_code = 533945262
    gml = get_mesh_list(mesh_code, gmom, gnl)
    
    ###Figure Print
    
    raw_data = create_subset_from_data_and_mesh_list(df,gml)
    show_print_array(raw_data[0:9,0:].T,"raw_data",1)
    
    # normed_raw_data, mvlist = create_normalized_data_from_mesh_list(df,gml)
    # show_print_array(normed_raw_data[0:9,0:].T,"normalized_data",2)
    
    # show_print_array(raw_data[0:9,0:144].T,":144 data",3)
    
    real_data = extract_data_every_n(raw_data,60)
    show_print_array(real_data.T,"real data",4)
    print(real_data.shape)
    
    repeated_data = repeat_data_columns(real_data,60)
    show_print_array(repeated_data.T,"repeated data",5, figsize=(100,5))
    print(repeated_data.shape)
    
    show_print_array(repeated_data[0:,0:144*3].T,"subset of repeated data",6)
    
    raw_data2=create_hourly_subset_from_data_and_mesh_list(df,gml)
    show_print_array(raw_data2[0:9,0:].T,"raw_data2",7)
    print(raw_data2.shape)
    
    plt.show()
    
    