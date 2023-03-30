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
def get_current_date():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d")

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

def load_geo_test_data(main_path, res_params, Distance, mesh_code):
    test_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        test_path += "-" + str(res_params[prm_i])
    test_path += "/"
    test_path += str(Distance)+"step-test/"

    # read mesh_code
    tested_file = test_path + str(mesh_code)
    if os.path.isfile(tested_file+".npz"):
        test_data = np.load(tested_file+".npz")
        (Y,UU,XX,Out,trainO) = (test_data["Y"],test_data["UU"],test_data["XX"],test_data["Out"],test_data["trainO"])
        return (Y,UU,XX,Out,trainO)
    else:
        print(test_path)
        print("ERROR at " + str(mesh_code))
  
def load_nco_test_data(main_path, res_params, Distance, mesh_code):
    test_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        test_path += "-" + str(res_params[prm_i])
    test_path += "/"
    test_path += str(Distance)+"step-test-nco/"

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

#1/Time * sigma(t in Time){(Y - Yt)**2 / Yt**2}
def get_ave_MSE(teacher, output):
  se = np.zeros(len(output))
  for i in range(len(output)):
    se[i] =  np.square( (output[i] - teacher[i]) / teacher[i])

  mse = np.sqrt(se.sum()/len(se))
  return mse
    
def get_mse_map_LGR(grl, gmom, expIndex,Distance, path, inSize):
  #Out_put_matrix
  gmom_mat = gmom["mat"]
  mse_mat = np.ones(gmom_mat.shape)
  #Reservoir
  set_grl = set(grl)

  for y in range(gmom_mat.shape[0]):
    for x in range(gmom_mat.shape[1]):
      tmp_mesh = gmom_mat[y,x]
      if tmp_mesh in set_grl:
        tmp_test_data = load_geo_test_data(path, tmp_mesh, expIndex, Distance, inSize)

        if type(tmp_test_data) != type((1,2)):
          #ERROE
          print(tmp_test_data)
          return tmp_test_data

        else:
          Y = tmp_test_data[0]
          Out = tmp_test_data[3]
          #MSE
          mse_mat[y,x] = get_ave_MSE(Out[0,:]/(10**(expIndex)) ,Y[0,:]/(10**(expIndex)) )
      else:
        mse_mat[y,x] = float("nan")
    
    print(f"{y * 1000 / gmom_mat.shape[0] // 1 /10} % Done")
  print("Complete!")
  return mse_mat

#基本的に上書きせず、あればそれを読み込む
def save_or_load_mse_map(path, expIndex, Distance, inSize, mse_map):
  Test_path=path+"MSE_MAP/"
  if not os.path.isdir(Test_path): os.mkdir(Test_path)
  Test_path=Test_path+"e"+str(expIndex)+"/"
  if not os.path.isdir(Test_path): os.mkdir(Test_path)
  Test_path=Test_path+"C"+str(inSize)+"/"
  if not os.path.isdir(Test_path): os.mkdir(Test_path)
  Test_path = Test_path + str(Distance) + "step/"
  if not os.path.isdir(Test_path): os.mkdir(Test_path)

  mse_file = Test_path + "mse_data"

  if os.path.isfile(mse_file+".npz"):
    print("Existed!" + str(Test_path))
    Out_mse_map = np.load(mse_file+".npz")
    #作成日時を表示
    time = os.path.getmtime(mse_file+".npz")
    d_time = datetime.datetime.fromtimestamp(time)
    print("Born: " + str(d_time))
    return Out_mse_map["mse_map"]

  elif type(mse_map) == type(np.ones(1)):
    np.savez_compressed(mse_file,mse_map = mse_map)
    print("Saved!" + str(Test_path))
    return mse_map

  else:
    print("ERROR")
    return Test_path

#リザバーにすることができるメッシュのMSEをMAP
#sub_in: get_matrix_of_mesh, get_R_list,load_geo_test_data, get_ave_mse
#in: Reservoir List, Matrix_of_mesh, expIndex,Distance,path,inSize
#out: Reservoir_list
#note: 上のget_mesh_list()やその子関数の出力が必要
def get_mse_NCo_map_LGR(grl, gmom, expIndex,Distance, path, inSize):
  #Out_put_matrix
  gmom_mat = gmom["mat"]
  mse_mat = np.ones(gmom_mat.shape)
  #Reservoir
  set_grl = set(grl)

  for y in range(gmom_mat.shape[0]):
    for x in range(gmom_mat.shape[1]):
      tmp_mesh = gmom_mat[y,x]
      if tmp_mesh in set_grl:
        tmp_test_data = load_nco_test_data(path, tmp_mesh, expIndex, Distance, inSize)

        if type(tmp_test_data) != type((1,2)):
          #ERROE
          print(tmp_test_data)
          return tmp_test_data

        else:
          Y = tmp_test_data[0]
          Out = tmp_test_data[3]
          #MSE
          mse_mat[y,x] = get_ave_MSE(Out[0,:]/(10**(expIndex)) ,Y[0,:]/(10**(expIndex)) )
      else:
        mse_mat[y,x] = float("nan")
    
    print(f"{y * 1000 / gmom_mat.shape[0] // 1 /10} % Done")
  print("Complete!")
  return mse_mat

def get_diff_map(grl, gmom,G_map, N_map):
  gmom_mat = gmom["mat"]
  mse_mat = np.ones(gmom_mat.shape)
  set_grl = set(grl)
  for y in range(gmom_mat.shape[0]):
    for x in range(gmom_mat.shape[1]):
      mse_mat[y,x] = G_map[y,x] - N_map[y,x]

  return mse_mat