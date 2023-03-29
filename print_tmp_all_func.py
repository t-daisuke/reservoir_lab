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

#INITIALIZE
#Fig Params
plt.rcParams["font.size"] =15
Figsize = (40,20)
plt.rcParams['image.cmap'] = 'bwr'

#test_dataを吸い出して出力
#For Test ver 1.0
def load_test_data(path, mesh_code, expIndex, Distance, inSize):
  Test_path = path + "train/"
  if not os.path.isdir(Test_path): return("Error" + str(mesh_code))
  Test_path=Test_path+str(mesh_code)+"/"
  if not os.path.isdir(Test_path): return("Error" + str(mesh_code))
  Test_path=Test_path+"e"+str(expIndex)+"/"
  if not os.path.isdir(Test_path): return("Error" + str(mesh_code))
  Test_path=Test_path+"C"+str(inSize)+"/"
  if not os.path.isdir(Test_path): os.mkdir(Test_path)
  Test_path=Test_path+str(Distance) + "step/"
  if not os.path.isdir(Test_path): os.mkdir(Test_path)

  test_file = Test_path + "test_data"

  if os.path.isfile(test_file+".npz"):
    test_data = np.load(test_file+".npz")
    (Y,UU,XX,Out,trainO) = (test_data["Y"],test_data["UU"],test_data["XX"],test_data["Out"],test_data["trainO"])
    return (Y,UU,XX,Out,trainO)
  else: return("Error" + str(mesh_code))
  
  # test_dataを吸い出して出力
# For Test ver 1.1
def load_NCo_data(path, mesh_code, expIndex, Distance, inSize):
  Test_path = path + "NCo/"
  if not os.path.isdir(Test_path): return("Error" + str(mesh_code))
  Test_path=Test_path+str(mesh_code)+"/"
  if not os.path.isdir(Test_path): return("Error" + str(mesh_code))
  Test_path=Test_path+"e"+str(expIndex)+"/"
  if not os.path.isdir(Test_path): return("Error" + str(mesh_code))
  Test_path=Test_path+"C"+str(inSize)+"/"
  if not os.path.isdir(Test_path): os.mkdir(Test_path)
  Test_path=Test_path+str(Distance) + "step/"
  if not os.path.isdir(Test_path): os.mkdir(Test_path)

  test_file = Test_path + "test_data"

  if os.path.isfile(test_file+".npz"):
    test_data = np.load(test_file+".npz")
    (Y,UU,XX,Out,trainO) = (test_data["Y"],test_data["UU"],test_data["XX"],test_data["Out"],test_data["trainO"])
    return (Y,UU,XX,Out,trainO)
  else: return("Error" + str(mesh_code))
  
def print_MSE(teacher, output, inputScaling, name, is_save, is_show, ylim=0,out_path="", now_str=""):
  if len(teacher) != len(output):
    print("print_MSE ERROR not same")
    return
  se = np.zeros(len(output))
  out_mse = output.sum()/len(output)

  for i in range(len(output)):
    se[i] =  np.sqrt(np.square( (output[i] - teacher[i])/out_mse )) #割合表示にした
    # se[i] =  np.sqrt(np.square( (output[i] - teacher[i])/inputScaling ))


  mse = se.sum()/len(se)

  print(name + ':MSE = ' + str( mse ))

  #print

  fig = plt.figure(figsize=Figsize)
  ax = fig.add_subplot(111, title=name+str(mse))
  ax.set_xlim(auto=True) #横軸
  ax.set_xlabel("time")
  if ylim != 0: ax.set_ylim(0,ylim)
  else:  ax.set_ylim(auto=True) #縦軸
  ax.set_ylabel("square error")
  ax.plot(se)
  filename = out_path + "/" + name +"@"+ now_str
  if is_save: fig.savefig(filename)
  if is_show: fig.show()
  else:
    plt.clf()
    plt.close()

  return mse
  
def show_print_array(X,name="no_name",is_show=True,is_save=False,x_lab="time",y_lab="signal",x_lim=True,y_lim=True,grid=True,out_path="", now_str=""):
  #Figsize,out_pathはglobal変数とする
  fig = plt.figure(figsize=Figsize)
  ax = fig.add_subplot(111, title=name)
  if grid is True:ax.grid(grid)

  if x_lim is True:ax.set_xlim(auto=x_lim) #横軸
  else:ax.set_xlim(x_lim)
  ax.set_xlabel(x_lab)

  if y_lim is True:ax.set_ylim(auto=y_lim) #縦軸
  else:ax.set_ylim(y_lim)
  ax.set_ylabel(y_lab)

  ax.plot(X)
  
  filename = out_path + "/" + name +"@"+ now_str
  if is_save: fig.savefig(filename)
  if is_show: fig.show()
  else:
    plt.clf()
    plt.close()
  return

def show_print_array_for_wout(Wout,name="no_name",is_show=True,is_save=False,x_lab="time",y_lab="signal",x_lim=True,y_lim=True,grid=True,out_path="", now_str=""):
  #Wout
  #Figsize,out_pathはglobal変数とする
  fig = plt.figure(figsize=Figsize)
  ax = fig.add_subplot(111, title=name)
  if grid is True:ax.grid(grid)

  if x_lim is True:ax.set_xlim(auto=x_lim) #横軸
  else:ax.set_xlim(x_lim)
  ax.set_xlabel(x_lab)

  if y_lim is True:ax.set_ylim(auto=y_lim) #縦軸
  else:ax.set_ylim(y_lim)
  ax.set_ylabel(y_lab)

  ax.bar(np.arange(300), Wout)

  filename = out_path + "/" + name +"@"+ now_str
  if is_save: fig.savefig(filename)
  if is_show: fig.show()
  else:
    plt.clf()
    plt.close()

  return

def show_print_array_for_test(Yt,Y,name="no_name",is_show=True,is_save=False,x_lab="time",y_lab="signal",x_lim=True,y_lim=True,grid=True,out_path="", now_str=""):
  #Figsize,out_pathはglobal変数とする
  fig = plt.figure(figsize=Figsize)
  ax = fig.add_subplot(111, title=name)
  if grid is True:ax.grid(grid)

  if x_lim is True:ax.set_xlim(auto=x_lim) #横軸
  else:ax.set_xlim(x_lim)
  ax.set_xlabel(x_lab)

  if y_lim is True:ax.set_ylim(auto=y_lim) #縦軸
  else:ax.set_ylim(y_lim)
  ax.set_ylabel(y_lab)

  ax.plot( Yt, 'g' )
  ax.plot( Y, 'b' )

  legends=['Target signal', 'Free-running predicted signal']
  ax.legend(legends)
  
  filename = out_path + "/" + name +"@"+ now_str
  if is_save: fig.savefig(filename)
  if is_show: fig.show()
  else:
    plt.clf()
    plt.close()

  return

def show_print_3array_for_test(Yt,Y,Y_o,name="no_name",is_show=True,is_save=False,x_lab="time",y_lab="signal",x_lim=True,y_lim=True,grid=True,out_path="", now_str=""):
  #Figsize,out_pathはglobal変数とする
  fig = plt.figure(figsize=Figsize)
  ax = fig.add_subplot(111, title=name)
  if grid is True:ax.grid(grid)

  if x_lim is True:ax.set_xlim(auto=x_lim) #横軸
  else:ax.set_xlim(x_lim)
  ax.set_xlabel(x_lab)

  if y_lim is True:ax.set_ylim(auto=y_lim) #縦軸
  else:ax.set_ylim(y_lim)
  ax.set_ylabel(y_lab)

  ax.plot( Yt, 'g' )
  ax.plot( Y, 'b' )
  ax.plot(Y_o, 'c')

  legends=['Yt', 'Y_Geo', 'Y_NCo']
  ax.legend(legends)
  
  filename = out_path + "/" + name +"@"+ now_str
  if is_save: fig.savefig(filename)
  if is_show: fig.show()
  else:
    plt.clf()
    plt.close()

  return

"""#### all_print"""

def print_all_GR(path, mesh_code, expIndex, Distance, inSize, project_name_main, is_save=False, is_show=True,out_path="", now_str=""):
  tmp = load_trained_data(path,expIndex,mesh_code,inSize)
  if type(tmp) != type({"A":1}):
    print(tmp)
    return
  (Win,W,X,Wout,x,Data) = tmp["trained_data"]

  tmp = load_test_data(path, mesh_code, expIndex, Distance, inSize)
  if type(tmp) != type((1,2)):
    print(tmp)
    return
  (Y,UU,XX,Out,trainO) = tmp

  tmp = load_NCo_data(path, mesh_code, expIndex, Distance, inSize)
  if type(tmp) != type((1,2)):
    print(tmp)
    return
  (Y_nco,UU_nco,XX_nco,Out_nco,trainO_nco) = tmp

  project_name = project_name_main +"Geo"+ "-C"+str(inSize) + "-step" +str(Distance) +"-e" + str(expIndex)
  inputScaling = 10 ** (expIndex)
  B=0 #初めに捨てる場合に使用

  # MSE
  name=project_name+"m"+str(mesh_code)+"MSE"
  print_MSE(Out[0,B:], Y[0, B:], inputScaling, name, is_save=is_save, is_show=is_show,out_path=out_path,now_str=now_str)

  #test
  #g,b,c、Yt,Y_Geo,Y_NCo
  name=project_name+"m"+str(mesh_code)+"Test"
  show_print_3array_for_test(Out[0,B:]/inputScaling,Y[0, B:]/inputScaling,Y_nco[0, B:]/inputScaling,name,is_save=is_save, is_show=is_show,out_path=out_path,now_str=now_str)

  #Wout
  name=project_name+"m"+str(mesh_code)+"Wout"
  show_print_array_for_wout(Wout[0,0:300],name,is_save=is_save,is_show=is_show,out_path=out_path,now_str=now_str)

  #UU,out2
  name=project_name+"m"+str(mesh_code)+"one_stepU"
  show_print_array_for_test(trainO[0,0:]/inputScaling, UU[0, B:]/inputScaling, name,is_save=is_save, is_show=is_show,out_path=out_path,now_str=now_str)

  #XX
  name=project_name+"m"+str(mesh_code)+"one_stepX"
  show_print_array(XX[0:100,0:500].T,name,is_save=is_save, is_show=is_show,out_path=out_path,now_str=now_str)

"""####print_one_data_GR(Free Rewite)"""

def print_one_data_GR(path, mesh_code, expIndex, Distance, inSize, project_name_main, is_save=False, is_show=True,out_path="", now_str="", XX_num=100):
  tmp = load_trained_data(path,expIndex,mesh_code,inSize)
  if type(tmp) != type({"A":1}):
    print(tmp)
    return
  (Win,W,X,Wout,x,Data) = tmp["trained_data"]

  tmp = load_test_data(path, mesh_code, expIndex, Distance, inSize)
  if type(tmp) != type((1,2)):
    print(tmp)
    return
  (Y,UU,XX,Out,trainO) = tmp

  tmp = load_NCo_data(path, mesh_code, expIndex, Distance, inSize)
  if type(tmp) != type((1,2)):
    print(tmp)
    return
  (Y_nco,UU_nco,XX_nco,Out_nco,trainO_nco) = tmp

  project_name = project_name_main +"Geo"+ "-C"+str(inSize) + "-step" +str(Distance) +"-e" + str(expIndex)
  inputScaling = 10 ** (expIndex)
  B=0 #初めに捨てる場合に使用

  # # MSE
  # name=project_name+"m"+str(mesh_code)+"MSE"
  # print_MSE(Out[0,B:], Y[0, B:], inputScaling, name, is_save=is_save, is_show=is_show,out_path=out_path,now_str=now_str)

  # #test
  # #g,b,c、Yt,Y_Geo,Y_NCo
  # name=project_name+"m"+str(mesh_code)+"Test"
  # show_print_3array_for_test(Out[0,B:]/inputScaling,Y[0, B:]/inputScaling,Y_nco[0, B:]/inputScaling,name,is_save=is_save, is_show=is_show,out_path=out_path,now_str=now_str)

  # #Wout
  # name=project_name+"m"+str(mesh_code)+"Wout"
  # show_print_array_for_wout(Wout[0,0:300],name,is_save=is_save,is_show=is_show,out_path=out_path,now_str=now_str)

  # #UU,out2
  # name=project_name+"m"+str(mesh_code)+"one_stepU"
  # show_print_array_for_test(trainO[0,0:]/inputScaling, UU[0, B:]/inputScaling, name,is_save=is_save, is_show=is_show,out_path=out_path,now_str=now_str)

  #XX
  name=project_name+"m"+str(mesh_code)+"one_stepX"
  show_print_array(XX[0:XX_num,0:500].T,name,is_save=is_save, is_show=is_show,out_path=out_path,now_str=now_str)

  #XX
  name=project_name+"m"+str(mesh_code)+"one_stepX"
  show_print_array(XX[0:XX_num,0:].T,name,is_save=is_save, is_show=is_show,out_path=out_path,now_str=now_str)

"""### meshに表示

**MSE_aveを求めるとき、inputScalingの逆変換をしてから、MSEの式に入力してる！、また、MSEの中は、Ytで正規化してある**

#### get_ave_MSE **<5/9に更新!!!>**
"""

#1/Time * sigma(t in Time){(Y - Yt)**2 / Yt**2}
def get_ave_MSE(teacher, output):
  se = np.zeros(len(output))
  for i in range(len(output)):
    se[i] =  np.square( (output[i] - teacher[i]) / teacher[i])

  mse = np.sqrt(se.sum()/len(se))
  return mse

"""#### get_mse_map_LGR"""

#リザバーにすることができるメッシュのMSEをMAP
#sub_in: get_matrix_of_mesh, get_R_list,load_test_data, get_ave_mse
#in: Reservoir List, Matrix_of_mesh, expIndex,Distance,path,inSize
#out: Reservoir_list
#note: 上のget_mesh_list()やその子関数の出力が必要
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
        tmp_test_data = load_test_data(path, tmp_mesh, expIndex, Distance, inSize)

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

"""#### save_or_load_mse_map"""

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

"""#### get_mse_NCo_map"""

#リザバーにすることができるメッシュのMSEをMAP
#sub_in: get_matrix_of_mesh, get_R_list,load_test_data, get_ave_mse
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
        tmp_test_data = load_NCo_data(path, tmp_mesh, expIndex, Distance, inSize)

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

"""#### get_diff_map"""

def get_diff_map(grl, gmom,G_map, N_map):
  gmom_mat = gmom["mat"]
  mse_mat = np.ones(gmom_mat.shape)
  set_grl = set(grl)
  for y in range(gmom_mat.shape[0]):
    for x in range(gmom_mat.shape[1]):
      mse_mat[y,x] = G_map[y,x] - N_map[y,x]

  return mse_mat

"""#### get_mat_array"""

def get_mat_array(mat):
  y_mat_array = np.ones(mat.shape)
  x_mat_array = np.ones(mat.shape)
  for y in range(mat.shape[0]):
    for x in range(mat.shape[1]):
      y_mat_array[y,x] = y
      x_mat_array[y,x] = x

  return {"y": y_mat_array, "x": x_mat_array}

"""#### print_mesh_code"""

def print_mesh_code(mesh_code,gmom,gnl,df):
  gmom_mat = gmom["mat"]
  exist_mat = gmom_mat.copy()
  #Data
  grmr = get_raw_mesh_array(df)
  set_grmr = set(grmr)
  #Reservoir
  grl = get_R_list(grmr, gmom, gnl)
  set_grl = set(grl)

  for y in range(gmom_mat.shape[0]):
    for x in range(gmom_mat.shape[1]):
      tmp = gmom_mat[y,x]
      if tmp in set_grmr:
        if tmp in set_grl:
          if tmp == mesh_code:
            exist_mat[y,x] = 1 #this mesh
          else:
            exist_mat[y,x] = 2 #R
        else:
          exist_mat[y,x] = 1 #D
      else:
        exist_mat[y,x] = 0 #ND

  #Fig Params
  plt.rcParams["font.size"] =15
  Figsize = (40,20)
  plt.rcParams['image.cmap'] = 'bwr'

  fig = plt.figure(figsize=Figsize)
  ax = fig.add_subplot(111, title=str(mesh_code))
  im = ax.imshow(exist_mat)
  c = fig.colorbar(im)

  #grid
  major_ticks = np.arange(0, gmom_mat.shape[0], 20)
  minor_ticks = np.arange(0, gmom_mat.shape[0], 10)
  ax.set_xticks(major_ticks)
  ax.set_xticks(minor_ticks, minor=True)
  ax.set_yticks(major_ticks)
  ax.set_yticks(minor_ticks, minor=True)
  ax.grid(which='minor', color="white", linestyle="--")
  ax.grid(which='major', color="yellow")

  # im.set_clim(0,4000) #limits
  fig.show()