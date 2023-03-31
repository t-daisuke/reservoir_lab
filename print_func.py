# -*- coding: utf-8 -*-
# Ver0324

# import
import pandas as pd
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy import linalg

import os
import time

from train_func import *
from test_func import *

"""
save file function
"""
def get_current_date(s=""):
    now = datetime.datetime.now()
    return now.strftime(str(s) + "%Y%m%d")

def save_file(title, s):
    filename = f"{title}-v1"  # 初期のファイル名

    # すでに同じ名前のファイルがある場合、新しいファイル名を作成する
    if os.path.isfile(filename):
        version = 1
        while True:
            version += 1
            new_filename = f"{title}-v{version}"
            if not os.path.isfile(new_filename):
                filename = new_filename
                break

    # ファイルを保存する
    with open(filename, "w") as f:
        f.write(s)

"""
LOAD FUNCTION
"""

def load_geo_test_data(main_path, res_params, distance, mesh_code):
    test_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        test_path += "-" + str(res_params[prm_i])
    test_path += "/"
    test_path += str(distance)+"step-test/"

    # read mesh_code
    tested_file = test_path + str(mesh_code)
    if os.path.isfile(tested_file+".npz"):
        test_data = np.load(tested_file+".npz")
        (Y,UU,XX,Out,trainO) = (test_data["Y"],test_data["UU"],test_data["XX"],test_data["Out"],test_data["trainO"])
        return (Y,UU,XX,Out,trainO)
    else:
        print(test_path)
        print("ERROR at " + str(mesh_code))
  
def load_nco_test_data(main_path, res_params, distance, mesh_code):
    test_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        test_path += "-" + str(res_params[prm_i])
    test_path += "/"
    test_path += str(distance)+"step-test-nco/"

    # read mesh_code
    tested_file = test_path + str(mesh_code)
    if os.path.isfile(tested_file+".npz"):
        test_data = np.load(tested_file+".npz")
        (Y,UU,XX,Out,trainO) = (test_data["Y"],test_data["UU"],test_data["XX"],test_data["Out"],test_data["trainO"])
        return (Y,UU,XX,Out,trainO)
    else:
        print(test_path)
        print("ERROR at " + str(mesh_code))

"""
GET MSE MAP FUNCTION
"""

#sqrt 1/Time * sigma(t in Time){(Y - Yt)/ Yt}**2
def get_ave_MSE(teacher, output):
  se = np.zeros(len(output))
  for i in range(len(output)):
    se[i] =  np.square( (output[i] - teacher[i]) / teacher[i])

  mse = np.sqrt(se.sum()/len(se))
  return mse

#基本的に上書きせず、あればそれを読み込む
def save_or_load_mse_map(name, main_path, res_params, distance, mse_map):
  # write into main_path/mse_map
  save_path = main_path + "mse_map/"
  if not os.path.isdir(save_path):
      os.mkdir(save_path)
      print("make " + str(save_path))
  mse_name = save_path + str(res_params[0])
  for prm_i in range(1,len(res_params)):
      mse_name += "-" + str(res_params[prm_i])
  mse_name += "-" + distance + str(name)

  if os.path.isfile(mse_name+".npz"):
    print("Existed!" + str(save_path))
    Out_mse_map = np.load(mse_name+".npz")
    #作成日時を表示
    time = os.path.getmtime(mse_name+".npz")
    d_time = datetime.datetime.fromtimestamp(time)
    print("Born: " + str(d_time))
    return
  elif type(mse_map) == type(np.ones(1)):
    np.savez_compressed(mse_name,mse_map = mse_map)
    print("Saved!" + str(save_path))
    return

  else:
    print("ERROR")
    return
  
#読み込む
def load_mse_map(name, main_path, res_params, distance):
  # write into main_path/mse_map
  save_path = main_path + "mse_map/"
  if not os.path.isdir(save_path):
      os.mkdir(save_path)
      print("make " + str(save_path))
  mse_name = save_path + str(res_params[0])
  for prm_i in range(1,len(res_params)):
      mse_name += "-" + str(res_params[prm_i])
  mse_name += "-" + distance + str(name)

  if os.path.isfile(mse_name+".npz"):
    Out_mse_map = np.load(mse_name+".npz")
    return True, Out_mse_map["mse_map"]
  else:
    print("NOT existed")
    return False, 0

def get_mse_map_GR(grl, gmom, main_path, saved_test_path, res_params, distance):
  is_existed, mse = load_mse_map("geo", main_path, res_params, distance)
  if is_existed == True: return mse
  
  #Out_put_matrix
  gmom_mat = gmom["mat"]
  mse_mat = np.ones(gmom_mat.shape)
  #Reservoir
  set_grl = set(grl)
  
  start_time = time.time()
  for y in range(gmom_mat.shape[0]):
    for x in range(gmom_mat.shape[1]):
      mesh_code = gmom_mat[y,x]
      if mesh_code in set_grl:
        tmp_test_data = load_geo_test_data(saved_test_path, res_params, distance, mesh_code)

        if type(tmp_test_data) != type((1,2)):
          #ERROE
          print(tmp_test_data)
          return tmp_test_data

        else:
          Y = tmp_test_data[0]
          Out = tmp_test_data[3]
          #MSE
          mse_mat[y,x] = get_ave_MSE(Out[0,:]/(10**res_params[0]) ,Y[0,:]/(10**res_params[0]) )
      else:
        mse_mat[y,x] = float("nan")
    
    rate = 100 * y/gmom_mat.shape[0]
    if sprit_printer(y,gmom_mat.shape[0],sprit_num=10):
        print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed. in GR")
  print("Complete!")
  return mse_mat

def get_mse_map_NCO(grl, gmom, main_path, saved_test_path, res_params, distance):
  is_existed, mse = load_mse_map("nco", main_path, res_params, distance)
  if is_existed == True: return mse
  
  #Out_put_matrix
  gmom_mat = gmom["mat"]
  mse_mat = np.ones(gmom_mat.shape)
  #Reservoir
  set_grl = set(grl)
  
  start_time = time.time()
  for y in range(gmom_mat.shape[0]):
    for x in range(gmom_mat.shape[1]):
      mesh_code = gmom_mat[y,x]
      if mesh_code in set_grl:
        tmp_test_data = load_nco_test_data(saved_test_path, res_params, distance, mesh_code)

        if type(tmp_test_data) != type((1,2)):
          #ERROE
          print(tmp_test_data)
          return tmp_test_data

        else:
          Y = tmp_test_data[0]
          Out = tmp_test_data[3]
          #MSE
          mse_mat[y,x] = get_ave_MSE(Out[0,:]/(10**res_params[0]) ,Y[0,:]/(10**res_params[0]) )
      else:
        mse_mat[y,x] = float("nan")
    
    rate = 100 * y/gmom_mat.shape[0]
    if sprit_printer(y,gmom_mat.shape[0],sprit_num=10):
        print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed. in NCO")
  print("Complete!")
  return mse_mat

def get_diff_map(grl, gmom,G_map, N_map, main_path, res_params, distance):
  is_existed, mse = load_mse_map("diff", main_path, res_params, distance)
  if is_existed == True: return mse
  
  gmom_mat = gmom["mat"]
  mse_mat = np.ones(gmom_mat.shape)
  set_grl = set(grl)
  for y in range(gmom_mat.shape[0]):
    for x in range(gmom_mat.shape[1]):
      mse_mat[y,x] = G_map[y,x] - N_map[y,x]

  return mse_mat

def create_mse_maps(main_path, saved_test_path, res_params, distance, df, Smesh_list=[]):
  gmom = get_matrix_of_mesh()
  gnl = get_n_list(res_params[4])
  dma = get_raw_mesh_array(df)
  
  if Smesh_list.length != 0:
    dma = cut_mlist(dma,Smesh_list)
    local_area_path = saved_test_path + "smesh"
    for smesh in Smesh_list:
        local_area_path += "-" + str(smesh)
    local_area_path += "/"
    if not os.path.isdir(local_area_path):
        print("No path " + str(local_area_path))
        return
    saved_test_path = local_area_path
    
    main_path = main_path + "smesh"
    for smesh in Smesh_list:
        main_path += "-" + str(smesh)
    main_path += "/"
    if not os.path.isdir(main_path):
        print("No path " + str(main_path))
        return
    
  Rl = get_R_list(dma, gmom, gnl)

  geo_mse_map = get_mse_map_GR(Rl, gmom, main_path, saved_test_path, res_params, distance)
  save_or_load_mse_map("geo", main_path, res_params, distance, geo_mse_map)
  
  nco_mse_map = get_mse_map_NCO(Rl, gmom, main_path, saved_test_path, res_params, distance)
  save_or_load_mse_map("nco", main_path, res_params, distance, nco_mse_map)
  
  diff_map = get_diff_map(Rl, gmom, geo_mse_map, nco_mse_map, main_path, res_params, distance)
  save_or_load_mse_map("diff", main_path, res_params, distance, diff_map)
  return