# -*- coding: utf-8 -*-
"""Test_ver1.0

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19IQ4ZP0LRw-ZxwJ3KHu8UklGxT2gYdj2

# Version

*   1.0ver
*   はじめのやつ

# コード規則

*   ほかの関数の出力をgloval変数っぽくつかうときは大文字
*   in,out,subin(ほかの関数をつかう)は明記
*   create DFでpathを定義+実行部分

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

"""# Train_function(基本的に上から先に実行される前提)

## Create or Get DataFrame

### create DF
"""

print(gc.collect())

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

"""## Get_mesh_list

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

"""## resercoir_list, resercoir_list_dict

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

"""### get_reservoir_list_dict


"""

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

"""# Test

## 子関数

### load_trained_data
"""

#Trainingによって保存されたものを呼び出す
#in: expIndex,mesh_code,inSize
#out: {"Tdata":(Win,W,X,Wout,x,Data), "path":train_path}
def load_Tdata(path,expIndex,mesh_code,inSize,seed_num):
  
  train_path=path+"seed" + str(seed_num) + "/"
  if not os.path.isdir(train_path): return("Error" + str(mesh_code))
  train_path=train_path+"train/"
  if not os.path.isdir(train_path): return("Error" + str(mesh_code))
  train_path=train_path+str(mesh_code)+"/"
  if not os.path.isdir(train_path): return("Error" + str(mesh_code))
  train_path=train_path+"e"+str(expIndex)+"/"
  if not os.path.isdir(train_path): return("Error" + str(mesh_code))
  train_path=train_path+"C"+str(inSize)+"/"
  if not os.path.isdir(train_path): return("Error" + str(mesh_code))

  #読み出し
  Tdata_file = train_path + "Tdata"
  if os.path.isfile(Tdata_file+".npz"):
    Tdata = np.load(Tdata_file+".npz")
    (Win,W,X,Wout,x,Data) = (Tdata["Win"],Tdata["W"],Tdata["X"],Tdata["Wout"],Tdata["x"],Tdata["Data"])
    return {"Tdata":(Win,W,X,Wout,x,Data), "path":train_path}
  else: return("Error" + str(mesh_code))

def dis_in_out(Data,inSize,outSize,dis):
  W = int(np.ceil(Data.shape[1]/dis))-1 #In,Outの横の長さ
  In = np.zeros((inSize,W)) #Dataの横の長さをdis分割してお尻だけ削る
  Out = np.zeros((outSize,W)) #頭だけ削る
  for i in range(W):
    In[0:inSize, i] = Data[0:inSize, i*dis] #頭からdisおきに格納
    Out[0:outSize, i] = Data[0:outSize, i*dis + dis] #Inに対してdis後が正解
  return (In,Out)

"""## メイン関数

### LGR
"""

#GeoReservoir実行、結果を保存
#Trainingはファイルに保存されてる前提
#Rlist_dictを制限すれば
#subin:  get_Rlist_dict(この子関数であるR_list,GMoM,GNL,(get_mesh_listは中で使ってる)は外に出てる)
#sub_func: load_Tdata, dis_in_out
#in:path,res_params,expIndex,Distance,Rlist_dict
#out: なし
@profile
def test_GR(path,res_params,expIndex,Distance,Rlist_dict):
  start_time = time.time()
  #trainを全てのセルで終えてる前提
  (leakingRate, resSize, spectralRadius, inSize, outSize, initLen, trainLen, testLen, reg, seed_num) = res_params

  #各testの時に学習するmeshのdict: Rlist_dict(input)

  Rlist = list(Rlist_dict.keys()) #Reservoir Mesh list

  print(str((time.time() - start_time)//1) + "s " + "LargeGeoReservoir Start...")

  All_R_dict={} #(Win,W,X,Wout,Data) path,x,u,(Y,UU,XX,In,Out,trainO)全てを格納する
  load_time=0
  for t,r in enumerate(Rlist):
    tmp = load_Tdata(path,expIndex,r,inSize,seed_num)
    if type(tmp) != type({"A":1}):
      print(tmp)
      return
    (Win,W,X,Wout,x,Data) = tmp["Tdata"]

    #save to All_R_dict
    tmp_dict={}
    tmp_dict["Win"] = Win
    tmp_dict["W"] = W
    tmp_dict["Wout"] = Wout
    # tmp_dict["Data"] = Data
    tmp_dict["X"] = X

    tmp_dict["x"] = x

    (In, Out) = dis_in_out(Data[0:,trainLen:trainLen+testLen],inSize,outSize,Distance)
    tmp_dict["In"] = In
    tmp_dict["Out"] = Out
    tmp_dict["trainO"] = Data[0:outSize,trainLen:trainLen+testLen]

    tmp_dict["u"] = In[0:inSize,0:1]
    
    tmp_dict["Y"] = np.zeros((outSize,testLen//Distance))
    tmp_dict["UU"] = np.zeros((outSize,testLen//Distance * Distance))
    tmp_dict["XX"] = np.zeros((resSize,testLen//Distance * Distance))

    All_R_dict[r] = tmp_dict

    rate = (1000*(t+1)) / len(Rlist) //1 /10
    
    if load_time == 0:
      load_time=(time.time() - start_time)//1

    if rate*10//1 %100 == 0:
      print(str(rate) +"% done @ " + str((time.time() - start_time)//1) + " s in " + str(load_time))

  print()
  print(str((time.time() - start_time)//1) + "s: " +str(load_time)+ "s load. Loaded and Initialized...")
  print(gc.collect())

  a=leakingRate
  print("Compute Geo reservoir...")
  for t in range(testLen//Distance):
    sst=time.time()
    check_time=sst
    for d_i in range(Distance):
      ####Compute Each
      
      #Check Time
      # print("d_i is " + str(d_i))
      #print(str((time.time() - check_time)) + "s @AAA")
      check_time=time.time()
      
      for r in Rlist:
        tmp_dict = All_R_dict[r] #params

        Win = tmp_dict["Win"]
        W = tmp_dict["W"]
        Wout = tmp_dict["Wout"]

        u = tmp_dict["u"]
        x = tmp_dict["x"]
        UU = tmp_dict["UU"]
        XX = tmp_dict["XX"]
        
        tmp1u=np.vstack((1,u))
        x = (1-a)*x + a*np.tanh( np.dot( Win, tmp1u ) + np.dot( W, x ) )
        tmp1x= np.vstack((1,x))
        u = np.dot( Wout,tmp1x)
        tmp_dict["x"] = x
        tmp_dict["u"] = u
           
        #4D
        XX[:,t*Distance+d_i] = x[0:,0] 
        UU[:,t*Distance+d_i] = u[0:,0]
        tmp_dict["XX"] = XX
        tmp_dict["UU"] = UU
      
      #Check Time
      #print(str((time.time() - check_time)) + "s @BBB")
      check_time=time.time()
      
      ####Self Organize
      for r in Rlist:
        u = All_R_dict[r]["u"]
        mlist = Rlist_dict[r]

        # 周囲のmのReservorから値を取得
        for i,m in enumerate(mlist):
          if i == 0:
            #index = 0の時は自分自身
            continue
          elif m < 0:
            #m is Not Reservoir mesh
            continue
          else:
            u[i,0] = All_R_dict[m]["u"][0,0]
        
        tmp_dict["u"] = u

      #Check Time
      #print(str((time.time() - check_time)) + "s @CCC")
      check_time=time.time()
    
    #Check Time
    #print(str((time.time() - check_time)) + "s @DDD")
    check_time=time.time()
       
    #set Y
    for r in Rlist:
      All_R_dict[r]["Y"][:,t]  = All_R_dict[r]["u"][0:,0]
    
    #Check Time
    #print(str((time.time() - check_time)) + "s @EEE")
    check_time=time.time()
    
    #for next time
    if t+2 < All_R_dict[Rlist[0]]["In"].shape[1]:
      for r in Rlist:
        In = All_R_dict[r]["In"]
        All_R_dict[r]["u"] = In[0:inSize,t+1:t+2]
    
    rate=(1000*t/(testLen//Distance))//1 /10
    if t % 10 == 0:
      print(str(rate) +"% done @ " + str((time.time() - start_time)//1) + " s in　" + str((time.time() - sst)//1) + "s")
    # print(str((time.time() - sst)//1) + "s "+str( (1000*t/(testLen//Distance))//1 /10 ) +"% done")

  print(str((time.time() - start_time)//1) + "s " + "Coputed !!")
  print(gc.collect())
  print("Saving...")

  for r in Rlist:
    tmp_dict = All_R_dict[r] #params

    Y = tmp_dict["Y"]
    UU = tmp_dict["UU"]
    XX = tmp_dict["XX"]
    Out = tmp_dict["Out"]
    trainO = tmp_dict["trainO"]
    
    Test_path=path+"Test/"
    if not os.path.isdir(Test_path): os.mkdir(Test_path)
    Test_path=Test_path+"seed"+str(seed_num)+"/"
    if not os.path.isdir(Test_path): os.mkdir(Test_path)
    Test_path=Test_path+str(r)+"/"
    if not os.path.isdir(Test_path): os.mkdir(Test_path)
    Test_path=Test_path+"e"+str(expIndex)+"/"
    if not os.path.isdir(Test_path): os.mkdir(Test_path)
    Test_path=Test_path+"C"+str(inSize)+"/"
    if not os.path.isdir(Test_path): os.mkdir(Test_path)
    Test_path = Test_path + str(Distance) + "step/"
    if not os.path.isdir(Test_path): os.mkdir(Test_path)
    test_file = Test_path + "test_data"

    np.savez_compressed(test_file, Y=Y, UU=UU, XX=XX, Out=Out, trainO=trainO)

  print(print(str((time.time() - start_time)//1) + "s " + "All Completed"))
  return

"""# まわすところ"""
path = './KDDI/'
res_params = (1, 1000, 0.75, 9, 9, 24*60, 3*24*60, 2*24*60-60+1, 1e-8,1)
expIndex = -9.5
dis = 30
print("Res Params")
print(res_params)
print("Exp Index")
print(expIndex)
print("Dis")
print(dis)
print(path)
gmom = get_matrix_of_mesh()
gnl = get_n_list(res_params[3])
dma = get_raw_mesh_array(df) #すべてのメッシュ
# dma = cut_mlist(dma,[45]) #Smesh == 45に固定
print("Data mesh:" + str(len(dma)))
Rl = get_R_list(dma, gmom, gnl)
print("Reservoir mesh:" + str(len(Rl)))
grld = get_Rlist_dict(Rl,gmom,gnl)

test_GR(path,res_params,expIndex,dis,grld)

