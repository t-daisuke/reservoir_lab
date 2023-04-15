# -*- coding: utf-8 -*-
# Ver0324

# import
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy import linalg

import os
import time

from train_func import *
from test_func import *

def display_time(t):
    t = float(t)
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = round((t % 60), 1)

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
    if i == len(teacher):
      #TODO (-1, 1, 100, 0.75, 9, 9, 1440, 4320, 2821, 1e-08, 2, 0.001), d=1でエラー
      print(str(i)+": (Yt, Yout) = "+str((len(teacher),len(output))))
      break
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
  mse_name += "-" + str(distance) + str(name)

  if os.path.isfile(mse_name+".npz"):
    print("Existed!" + str(save_path))
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
  mse_name += "-" + str(distance) + str(name)

  if os.path.isfile(mse_name+".npz"):
    Out_mse_map = np.load(mse_name+".npz")
    return True, Out_mse_map["mse_map"]
  else:
    print("NOT existed")
    return False, 0

#読み込む
def load_gr_data(mesh_code, main_path, res_params, distance):
  # write into main_path/gr_data
  save_path = main_path + "gr_data/"
  if not os.path.isdir(save_path):
      os.mkdir(save_path)
      print("make " + str(save_path))
  mse_name = save_path + str(res_params[0])
  for prm_i in range(1,len(res_params)):
      mse_name += "-" + str(res_params[prm_i])
  mse_name += "-" + str(distance) + str(mesh_code)

  if os.path.isfile(mse_name+".npz"):
    return True
  else:
    print("NOT existed")
    return False

def get_mse_map_GR(grl, gmom, main_path, saved_test_path, res_params, distance, Smesh_list=[]):
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

def get_mse_map_NCO(grl, gmom, main_path, saved_test_path, res_params, distance, Smesh_list=[]):
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

def get_diff_map(grl, gmom,G_map, N_map, main_path, res_params, distance, Smesh_list=[]):
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
  if len(Smesh_list) != 0:
    gmom = get_matrix_of_mesh(Smesh_l=[Smesh_list])
  gnl = get_n_list(res_params[4])
  dma = get_raw_mesh_array(df)
  
  if len(Smesh_list) != 0:
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
      os.mkdir(main_path)
      print("make" + str(main_path))
    
  Rl = get_R_list(dma, get_matrix_of_mesh(), gnl)

  geo_mse_map = get_mse_map_GR(Rl, gmom, main_path, saved_test_path, res_params, distance, Smesh_list)
  save_or_load_mse_map("geo", main_path, res_params, distance, geo_mse_map)
  
  nco_mse_map = get_mse_map_NCO(Rl, gmom, main_path, saved_test_path, res_params, distance, Smesh_list)
  save_or_load_mse_map("nco", main_path, res_params, distance, nco_mse_map)
  
  diff_map = get_diff_map(Rl, gmom, geo_mse_map, nco_mse_map, main_path, res_params, distance, Smesh_list)
  save_or_load_mse_map("diff", main_path, res_params, distance, diff_map)
  return

def get_copy_gr_data(main_path, saved_test_path, res_params, distance, mesh_list):
  is_existed = True
  for m in mesh_list:
    is_existed = is_existed and load_gr_data(m, main_path, res_params, distance)
  if is_existed:
    print("Existed:" + str(mesh_list))
    print(str(res_params))
    return
  
  save_path = main_path + "gr_data/"
  if not os.path.isdir(save_path):
      os.mkdir(save_path)
      print("make " + str(save_path))
  
  for mesh_code in mesh_list:
    tmp_geo_test_data = load_geo_test_data(saved_test_path, res_params, distance, mesh_code)

    if type(tmp_geo_test_data) != type((1,2)):
      #ERROE
      print(tmp_geo_test_data)
      return tmp_geo_test_data
    
    tmp_nco_test_data = load_nco_test_data(saved_test_path, res_params, distance, mesh_code)

    if type(tmp_nco_test_data) != type((1,2)):
      #ERROE
      print(tmp_nco_test_data)
      return tmp_nco_test_data
    
    tmp_train_data = load_trained_data(saved_test_path, res_params, mesh_code)
    if type(tmp_train_data) != type({"A": 1}):
        print(tmp_train_data)
        return
    else:
      tmp_train_data = tmp_train_data["trained_data"]
    
    (Win, W, X, Wout, x, Data) = tmp_train_data
    
    (Ygeo,UUgeo,XXgeo,Outgeo,trainOgeo) = tmp_geo_test_data
    
    (Ynco,UUnco,XXnco,Outnco,trainOnco) = tmp_nco_test_data
    
    gr_data_name = save_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        gr_data_name += "-" + str(res_params[prm_i])
    gr_data_name += "-" + str(distance) + str(mesh_code)
    # np.savez_compressed(gr_data_name,geo_test_data = tmp_geo_test_data, nco_test_data = tmp_nco_test_data, train_data = tmp_train_data)
    np.savez_compressed(gr_data_name,
                        Win=Win, W=W, X=X,Wout=Wout, x=x, Data=Data,
                        Ygeo=Ygeo, UUgeo=UUgeo, XXgeo=XXgeo,Outgeo=Outgeo, trainOgeo=trainOgeo,
                        Ynco=Ynco, UUnco=UUnco, XXnco=XXnco,Outnco=Outnco, trainOnco=trainOnco)
    print("Saved!" + str(gr_data_name))
    

    
  print("Copy Data Complete!")
  return

#default: ikebukuro
def copy_gr_data(main_path, saved_test_path, res_params, distance, Smesh_list=[], center_mesh_mat = (24,35)):
  gmom = get_matrix_of_mesh()
  
  if len(Smesh_list) != 0:
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
      os.mkdir(main_path)
      print("make" + str(main_path))
  
  mesh_list=[]
  for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
      if i*j != 0: continue
      mesh_list.append(gmom["mat"][center_mesh_mat[0] + i, center_mesh_mat[1] + j])

  get_copy_gr_data(main_path, saved_test_path, res_params, distance, mesh_list)
  return

def show_print_array(X,name="no_name",figure_number=None,figsize=(10,5),fontsize=15):
  plt.rcParams["font.size"] =fontsize
  fig = plt.figure(figsize=figsize, num=figure_number)
  ax = fig.add_subplot(111, title=name)

  ax.plot(X)
  
  ax.grid(True)
  
  if figure_number==None:plt.show()
  else: return
