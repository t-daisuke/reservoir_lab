# -*- coding: utf-8 -*-
"""Create_Train_Data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rrhB0H6iwGs9PQItoLg-UY3JZRD9K7Zj

# コード規則

*   ほかの関数の出力をgloval変数っぽくつかうときは大文字
*   in,out,subin(ほかの関数をつかう)は明記

# Initial

## import
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg 
import random
import os
import time
import gc
# numpy.linalg is also an option for even fewer dependencies
#日付を取得してデータを取得
import datetime

"""## function"""



"""# Main(基本的に上から先に実行される前提)

## Create or Get DataFrame

### create DF
"""

path = './KDDI/'

df_path = path+"df/"
if not os.path.isdir(df_path): os.mkdir(df_path)
if os.path.isfile(df_path+"df.csv") : df = pd.read_csv(df_path+"df.csv")
else:
  file = []
  file.append(path)
  file.append(path + 'KLD100101_1.csv')
  file.append(path + 'KLD100102_1.csv')
  file.append(path + 'KLD100103_1.csv')
  file.append(path + 'KLD100104_1.csv')
  file.append(path + 'KLD100105_1.csv')
  file.append(path + 'KLD100106_1.csv')
  
  df1 = pd.read_csv(file[1],header=0)
  #***は10人以下なので0~10人にした
  for i in range(len(df1)):
    if df1.iat[i,7] == "***":
      df1.iat[i,7] = random.uniform(0,10)
    else:
      df1.iat[i,7] = float(df1.iat[i,7])
    
    if df1.iat[i,8] == "***":
      df1.iat[i,8] =  random.uniform(0,10)
    else:
      df1.iat[i,8] = float(df1.iat[i,8])
  df1["sum_population"] = df1["stay_pred_population"] + df1["move_pred_population"]

  df2 = pd.read_csv(file[2],header=0)
  for i in range(len(df2)):
    if df2.iat[i,7] == "***":
      df2.iat[i,7] = random.uniform(0,10)
    else:
      df2.iat[i,7] = float(df2.iat[i,7])
    
    if df2.iat[i,8] == "***":
      df2.iat[i,8] =  random.uniform(0,10)
    else:
      df2.iat[i,8] = float(df2.iat[i,8])
  df2["sum_population"] = df2["stay_pred_population"] + df2["move_pred_population"]

  df3 = pd.read_csv(file[3],header=0)
  for i in range(len(df3)):
    if df3.iat[i,7] == "***":
      df3.iat[i,7] = random.uniform(0,10)
    else:
      df3.iat[i,7] = float(df3.iat[i,7])
    
    if df3.iat[i,8] == "***":
      df3.iat[i,8] =  random.uniform(0,10)
    else:
      df3.iat[i,8] = float(df3.iat[i,8])
  df3["sum_population"] = df3["stay_pred_population"] + df3["move_pred_population"]

  df4 = pd.read_csv(file[4],header=0)
  for i in range(len(df4)):
    if df4.iat[i,7] == "***":
      df4.iat[i,7] = random.uniform(0,10)
    else:
      df4.iat[i,7] = float(df4.iat[i,7])
    
    if df4.iat[i,8] == "***":
      df4.iat[i,8] =  random.uniform(0,10)
    else:
      df4.iat[i,8] = float(df4.iat[i,8])
  df4["sum_population"] = df4["stay_pred_population"] + df4["move_pred_population"]

  df5 = pd.read_csv(file[5],header=0)
  for i in range(len(df5)):
    if df5.iat[i,7] == "***":
      df5.iat[i,7] = random.uniform(0,10)
    else:
      df5.iat[i,7] = float(df5.iat[i,7])
    
    if df5.iat[i,8] == "***":
      df5.iat[i,8] =  random.uniform(0,10)
    else:
      df5.iat[i,8] = float(df5.iat[i,8])
  df5["sum_population"] = df5["stay_pred_population"] + df5["move_pred_population"]

  df6 = pd.read_csv(file[6],header=0)
  for i in range(len(df6)):
    if df6.iat[i,7] == "***":
      df6.iat[i,7] = random.uniform(0,10)
    else:
      df6.iat[i,7] = float(df6.iat[i,7])
    
    if df6.iat[i,8] == "***":
      df6.iat[i,8] =  random.uniform(0,10)
    else:
      df6.iat[i,8] = float(df6.iat[i,8])
  df6["sum_population"] = df6["stay_pred_population"] + df6["move_pred_population"]

  df = df1.copy()
  df = df.append(df2)
  df = df.append(df3)
  df = df.append(df4)
  df = df.append(df5)
  df = df.append(df6)
  df.to_csv(df_path+"df.csv", index=False, header=True)

"""### cut_mesh_list"""

#Smeshのリストの分をrawから選定
#in: raw_mesh(df.なんとか)
def cut_mlist(m_array,Smesh_list=[45]):
  mset=set(m_array)
  #5339-{45,46,35}-{00~99}-{1~4}
  Pmesh = 5339
  mesh_list=[]

  for sm in Smesh_list:
    for i in range(100):
      tmp_mesh=Pmesh*10000 + sm*100 + i
      for j in [1,2,3,4]:
        mesh = tmp_mesh*10 + j
        if mesh in mset:
          mesh_list.append(mesh)

  return mesh_list

"""### get_raw_mesh_array"""

def get_raw_mesh_array(df):
  return df["mesh_code"].unique()

# #TEST
# len(cut_mlist(get_raw_mesh_array(df),[46]))

"""## Get_mesh_list(New)

### TODO

*   index_tpl = GMoM["dic"][mesh_O]で、GMoMにないものを求めるとエラーになる→GMoMは大きめに求める
*   このときのエラー処理ができてない

### 子関数

#### get_matrix_of_mesh
"""

#長方形のS_meshをもつ、1/2地域メッシュまでのメッシュを配列にしたものと、mesh: indexのdictをかえす
#NOTE: 十分に大きいメッシュコードを想定しなければいけない
#input: Smesh_list([[]], 長方形)
#output: {mat: matrix, dic: dict}
#index → mesh と mesh → index
def get_matrix_of_mesh(Smesh_l=[[54,55,56,57],[44,45,46,47],[34,35,36,37],[24,25,26,27]], Pmesh=5339, Amesh_n=[[3,4],[1,2]]):
  tmp_l = []
  for sm_tate in Smesh_l:
    for sub_tm in [i for i in range(9,-1,-1)]:
      #9X; 8X; ,,, 0X
      for a_l in Amesh_n:
        #3434 or 1212

        #よこ
        for sm in sm_tate:
          #45...46...
          for tm in range(sub_tm*10, sub_tm*10+10):
            #X0 ... X9
            tmp1 = Pmesh*100 + sm
            tmp2 = tmp1*100 + tm 
            for a in a_l:
              tmp3 = tmp2*10 + a
              tmp_l.append(tmp3)
  
  mat = np.array(tmp_l).reshape(20*len(Smesh_l),20*len(Smesh_l[0]))

  dic={}
  for y in range(mat.shape[0]):
    for x in range(mat.shape[1]):
      dic[mat[y,x]] = (y,x)

  return {"mat": mat, "dic": dic}

# #TEST
# tmp = (1,2)
# print(tmp)
# A = get_matrix_of_mesh()
# a_tmp = A["mat"][tmp[0],tmp[1]]
# print(a_tmp)
# a2_tmp = A["dic"][a_tmp]
# print(a2_tmp)

"""#### get_n_list"""

#入力次元を与えたら、そのまわりのneighbor_listを返す
#in: dim(int 入力次元,5,1,9,25,(奇数)**2)必要に応じて加筆
#out: neighbor_list([(int y, int x)])
#note: 中心セルは必ず頭になる
#ex: [(0, 0), (-1, 0), (0, 1), (0, -1), (1, 0)]
def get_n_list(dim):
  n_list=[(0,0)] #はじめはセルO(自分自身)
  if dim == 1:
    return n_list
  if dim == 5:
    #例
    n_list.append((-1,0)) #N
    n_list.append((0,1)) #E
    n_list.append((0,-1)) #W
    n_list.append((1,0)) #S
    return n_list
  
  sqrt_dim = int(np.sqrt(dim)) #dimは奇数の**2を想定, sqrtは奇数
  abs_mesh = sqrt_dim//2 #-abs_mesh ... abs_meshになる
  for y in range(-abs_mesh,abs_mesh+1,1):
    for x in range(-abs_mesh,abs_mesh+1,1):
      if(y,x) == (0,0):
        continue
      n_list.append((y,x))
  return n_list

"""### get_mesh_list"""

#中心メッシュコードから、まわりのメッシュコードを返す
#sub_in: get_matrix_of_mesh(), get_n_list()
#in: mesh_O(int 中心メッシュコード), dim  入力次元のメッシュが中心メッシュに対していくつのindexか)
#out: mesh_list
#note: 上の子関数の出力はこの関数のそとから出力でもらう
def get_mesh_list(mesh_O, GMoM, GNL):
  #mesh → index
  index_tpl = GMoM["dic"][mesh_O]

  #index → neighbor_index_list
  index_tpl_list=[]
  for n_index_tpl in GNL:
    index_tpl_list.append((index_tpl[0]+n_index_tpl[0], index_tpl[1]+n_index_tpl[1]))

  #neighbor_index_list → mesh_list
  mesh_list=[]
  for index_tpl in index_tpl_list:
    mesh_list.append(GMoM["mat"][index_tpl[0], index_tpl[1]])

  return mesh_list

#TEST
# gmom = get_matrix_of_mesh()
# gnl = get_n_list(9)
# get_mesh_list(533945002,gmom,gnl)

"""## resercoir_list, resercoir_list_dict(New)

### get_reservoir_list
"""

#get_mesh_listで作成されるメッシュのうち、周りのメッタシュのデーがmesh_arrayにある=リザバーにすることができるメッシュのリストを返す
#sub_in: get_mesh_list()、直接的にはget_matrix_of_mesh(), get_n_list()
#in: data_mesh_array(arrayになる df["mesh_code"].unique()) データがあるmeshのarray
#out: Reservoir_list
#note: 上のget_mesh_list()やその子関数の出力が必要
def get_R_list(data_mesh_array, GMoM, GNL):
  r_list=[]
  for m in data_mesh_array:
    tmp_l=get_mesh_list(m,GMoM,GNL)
    if not (False in np.in1d(np.array(tmp_l), data_mesh_array)):
      r_list.append(tmp_l[0])
  return r_list

# # TEST
# gmom = get_matrix_of_mesh()
# gnl = get_n_list(9)
# dma = get_raw_mesh_array(df) 
# # dma = cut_mlist(dma,[45]) #Smesh == 45に固定
# print(len(dma))
# Rl = get_R_list(dma, gmom, gnl)
# print(len(Rl))

"""### get_reservoir_list_dict(NEW)"""

#リザバーメッシュのうち(周りにデータがあるメッシュ){mesh: Rmesh, Rmesh, -Rmesh, Rmesh}みたいなdictを作成
#TestのSelf OrgnizeでReservoirの返り値を自分の予想でするか(自分自身のセルと、データしかないせる)、ほかのRからとるかの時に使う
#in: reservoir_list
#out: self_orgnize_dict
def get_Rlist_dict(R_list, GMoM, GNL):
  #TODO
  SOdict = {}
  R_set = set(R_list)
  for r in R_list:
    m_l = get_mesh_list(r,GMoM,GNL)
    tmp_l=[]
    for m in m_l:
      if not m in R_set:
        #Dataのみのセル→ひとつでもあるとNTR
        tmp_l.append(-m)
      else:
        #TR
        tmp_l.append(m)
    
    SOdict[r] = tmp_l


  return SOdict

#  #TEST
# gmom = get_matrix_of_mesh()
# gnl = get_n_list(9)
# dma = get_raw_mesh_array(df) 
# # dma = cut_mlist(dma,[45]) #Smesh == 45に固定
# print(len(dma))
# Rl = get_R_list(dma, gmom, gnl)
# print(len(Rl))
# grld = get_Rlist_dict(Rl,gmom,gnl)
# print(len(grld.keys()))
# tmp=533934493
# print(tmp in Rl)
# print(grld[tmp])
# tmp=533934482
# print(tmp in Rl)
# print(grld[tmp]) #KeyError Reservoirじゃないから

"""## メッシュの要素の調査(in_dim: 9)

### matrix create
"""

# gmom = get_matrix_of_mesh()
# gnl = get_n_list(9)
# #Out_put_matrix
# gmom_mat = gmom["mat"]
# exist_mat = gmom_mat.copy()
# #Data
# grmr = get_raw_mesh_array(df)
# set_grmr = set(grmr)
# #Reservoir
# grl = get_R_list(grmr, gmom, gnl)
# set_grl = set(grl)

# for y in range(gmom_mat.shape[0]):
#   for x in range(gmom_mat.shape[1]):
#     tmp = gmom_mat[y,x]
#     if tmp in set_grmr:
#       if tmp in set_grmr:
#         exist_mat[y,x] = 2 #R
#       else:
#         exist_mat[y,x] = 1 #D
#     else:
#       exist_mat[y,x] = 0 #ND

"""### fig"""

# #Fig Params
# plt.rcParams["font.size"] =15
# Figsize = (25,12.5)
# plt.rcParams['image.cmap'] = 'bwr'

# fig = plt.figure(figsize=Figsize)
# ax = fig.add_subplot(111, title="mesh_map")
# im = ax.imshow(exist_mat)
# c = fig.colorbar(im)

# #grid
# major_ticks = np.arange(0, gmom_mat.shape[0], 20)
# minor_ticks = np.arange(0, gmom_mat.shape[0], 10)
# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks, minor=True)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)
# ax.grid(which='minor', color="white", linestyle="--")
# ax.grid(which='major', color="yellow")

# # im.set_clim(0,4000) #limits
# fig.show()

"""## Create Training Data Set(Doing)

### 子関数
"""

#線形補完する
def complement_linear(start, end, interval = 60):
  #intervalの分、start~endまでを線形補完し、listで返す
  differ = (end - start)/interval
  comp = [i*differ + start for i in range(interval)]
  return comp

#mesh_listからTraining用のraw_dataをつくる
def create_data_from_mesh_list(df,mesh_list):
  test = df.sort_values(['mesh_code','yyyymm','hour','holiday_flg','gender'])
  data_list = [] #np.array
  test_list = [] #df
  #中央、上、下、？、_
  for tmp_mesh_code in mesh_list:
    #gender,holidayを全て足し合わせて、/2にした
    test_mesh = test[test['mesh_code'] == tmp_mesh_code]
    if len(test_mesh) == 0: print("mesh_code_ERROR")

    tmp_dic = dict(sum_id = [i for i in range(len(test_mesh)//4)])
    df_sum = pd.DataFrame(data=tmp_dic)

    n=0
    mesh_array=test_mesh['mesh_code'].to_numpy()
    sum_array=test_mesh['sum_population'].to_numpy()
    hour_array=test_mesh['hour'].to_numpy()
    day_array=test_mesh['yyyymm'].to_numpy()

    meshcode_list=[]
    sum_list=[]
    hour_list=[]
    day_list=[]
    for i in df_sum['sum_id'].to_numpy():
      n=4*i
      meshcode_list.append(mesh_array[n])
      sum_list.append(sum_array[n])
      day_list.append(day_array[n])
      hour_list.append(hour_array[n])
      for j in [1,2,3]:
        n = 4*i + j
        sum_list[-1] = sum_list[-1] + sum_array[n]
      sum_list[-1] = sum_list[-1]/2

    df_sum['mesh_code'] = np.array(meshcode_list)
    df_sum['sum_population'] = np.array(sum_list)
    df_sum['hour'] = np.array(hour_list)
    df_sum['day'] = np.array(day_list)

    min = [i for _ in range(len(df_sum)-1) for i in range(60)]
    min.append(0) #201901/0:00~2019/06/23:00 (最後は終点がないから補填できない)
    tmp_dic = dict(minutes = min) 
    df_linear = pd.DataFrame(data=tmp_dic)

    l_mesh_list=[]
    l_hour_list=[]
    l_day_list=[]
    l_sum_list=[]

    for df_sum_i in range(len(df_sum)-1):
      tmp = complement_linear(sum_list[df_sum_i],sum_list[df_sum_i+1],60)
      for i in range(60):
        l_mesh_list.append(meshcode_list[df_sum_i])
        l_hour_list.append(hour_list[df_sum_i])
        l_day_list.append(day_list[df_sum_i])

        l_sum_list.append(tmp[i])
      
    l_mesh_list.append(mesh_list[-1])
    l_hour_list.append(hour_list[-1])
    l_day_list.append(day_list[-1])
    l_sum_list.append(sum_list[-1])

    df_linear['mesh_code'] = np.array(l_mesh_list)
    df_linear['sum_population'] = np.array(l_sum_list)
    df_linear['hour'] = np.array(l_hour_list)
    df_linear['day'] = np.array(l_day_list)

    test_list.append(df_linear)
    data_list.append(l_sum_list)
  data = np.array(data_list[0])
  for i in range(1,len(data_list)):
    data = np.vstack((data, np.array(data_list[i])))
  return data

"""### train_GR

途中でやめると、途中のデータを読み出してエラーになることがある。そのときはファイルを消してください
"""

#in :res_params, path(path/train/e/C/...にほぞんされる)
def train_LGR(path,res_params,raw_data,expIndex,mesh_code,is_update = False):
  (leakingRate, resSize, spectralRadius, inSize, outSize, initLen, trainLen, testLen, reg, seed_num) = res_params

  train_path=path+"seed" + str(seed_num) + "/"
  if not os.path.isdir(train_path): os.mkdir(train_path)
  train_path=train_path+"train/"
  if not os.path.isdir(train_path): os.mkdir(train_path)
  train_path=train_path+str(mesh_code)+"/"
  if not os.path.isdir(train_path): os.mkdir(train_path)
  train_path=train_path+"e"+str(expIndex)+"/"
  if not os.path.isdir(train_path): os.mkdir(train_path)
  train_path=train_path+"C"+str(inSize)+"/"
  if not os.path.isdir(train_path): os.mkdir(train_path)

  #mesh_codeのデータがある→読み出し
  Tdata_file = train_path + "Tdata"
  if os.path.isfile(Tdata_file+".npz") and (not is_update):
    Tdata = np.load(Tdata_file+".npz")
    (Win,W,X,Wout,x,Data) = (Tdata["Win"],Tdata["W"],Tdata["X"],Tdata["Wout"],Tdata["x"],Tdata["Data"])
    return (Win,W,X,Wout,x,Data)

  #Train
  Data = raw_data * 10**expIndex
  Data = Data.astype(np.float64)
  #trainは1 timeずつ
  In = Data[0:inSize,0:trainLen+testLen-1] #入力
  Out = Data[0:outSize,1:trainLen+testLen] #出力
  a = leakingRate
  np.random.seed(seed_num)
  Win = (np.random.rand(resSize,1+inSize) - 0.5) * 1 # -0.5~0.5の一様分布
  W = np.random.rand(resSize,resSize) - 0.5 
  X = np.zeros((1+resSize,trainLen-initLen))
  Yt = Out[0:outSize,initLen:trainLen] #init ~ train-1でtrain(train-init分)

  # normalizing and setting spectral radius (correct, slow):
  rhoW = max(abs(linalg.eig(W)[0]))
  W *= spectralRadius / rhoW
  print("Training...")
  # run the reservoir with the Data and collect X
  x = np.zeros((resSize,1))
  for t in range(trainLen):
    u = In[0:inSize,t:t+1]
    x = (1-a)*x + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) ) #瞬間の値
    if t >= initLen:
        X[:,t-initLen] = np.vstack((1,x))[:,0]
  Wout = linalg.solve( np.dot(X,X.T) + reg*np.eye(1+resSize), np.dot(X,Yt.T) ).T

  #save
  np.savez_compressed(Tdata_file,Win=Win,W=W,X=X,Wout=Wout,x=x,Data=Data)

  return (Win,W,X,Wout,x,Data)

"""### create_train_data"""

#Trainingを回して、保存させる
#subin: {get_matrix_of_mesh, get_neighbor_list, get_raw_mesh_array}get_resercoir_list(), get_mesh_list
#in: path,res_params,df,expIndex
#out: なし
def create_Tdata(path,res_params,df,expIndex, is_update=False):
  gmom = get_matrix_of_mesh()
  gnl = get_n_list(res_params[3]) #たぶんinSize
  dma = get_raw_mesh_array(df) 

  Rlist = get_R_list(dma,gmom,gnl)
  for index,mesh_code in enumerate(Rlist):
    gml = get_mesh_list(mesh_code, gmom, gnl)
    raw_data = create_data_from_mesh_list(df,gml)
    _=train_LGR(path,res_params,raw_data,expIndex,mesh_code, is_update=is_update)
    print(str(100 * index/len(Rlist) ))
  print("Train Data Saved")
  return

"""## Trainの実行部分部分"""

res_params = (1, 1000, 0.75, 9, 9, 24*60, 3*24*60, 2*24*60-60+1, 1e-8, 2)
expIndex = -9.5
is_up = False
print(path)
print(res_params)

start_time = time.perf_counter() #Start
print("Start")
create_Tdata(path,res_params,df,expIndex,is_update = is_up)
end_time = time.perf_counter() #End
print("Save Train Data:"+ str(end_time - start_time) + "s")

