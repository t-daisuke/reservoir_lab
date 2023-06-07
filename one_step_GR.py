#Ver June

#fileの保存の名前はgeo_res_paramsを採用
####
#note
####

"""
1次元と9次元で領域が変わるので困る
trainとtestのデータが、geoをncoで異なる
"""

# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from train_func import *
from test_func import dis_in_out
from print_func import check_gr_data_existed, load_geo_test_data, load_nco_test_data
from print_func import display_time #, create_mse_maps

import os
import time

import concurrent.futures
import threading

import pdb

#########
#Train
#########

def train_1step_GR(main_path, res_params, raw_data_subset, mesh_code, is_update=False):
    (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
     initLen, trainLen, testLen,
     reg, seed_num, conectivity) = res_params
    
    if inSize != 9 or outSize != 1:
        print("In Out Error! @ GR")
        print((inSize, outSize))
        return
    
    train_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        train_path += "-" + str(res_params[prm_i])
    train_path += "/"

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    train_path += "1step_GR_train/"
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
        print("make " + str(train_path))

    # mesh_codeのデータがある→読み出し
    trained_file = train_path + str(mesh_code)
    if os.path.isfile(trained_file+".npz") and (not is_update):
        trained_data = np.load(trained_file+".npz")
        (Win, W, X, Wout, x, Data) = (
            trained_data["Win"], trained_data["W"], trained_data["X"], trained_data["Wout"],
            trained_data["x"], trained_data["Data"])
        return (Win, W, X, Wout, x, Data)

    # Train
    Data = raw_data_subset * 10**expIndex
    Data = Data.astype(np.float64)
    # trainは1 timeずつ
    In = Data[0:inSize, 0:trainLen+testLen-1]  # 入力
    Out = Data[0:outSize, 1:trainLen+testLen]  # 出力
    a = leakingRate
    np.random.seed(seed_num)
    Win = (np.random.rand(resSize, 1+inSize) - 0.5) * 2  # -1~1の一様分布
    W = create_sparse_rand_matrix(resSize, resSize, conectivity)
    rhoW = max(linalg.eigh(W)[0])
    W *= spectralRadius / rhoW
    X = np.zeros((1+resSize, trainLen-initLen))
    Yt = Out[0:outSize, initLen:trainLen]  # init ~ train-1でtrain(train-init分)

    
    # run the reservoir with the Data and collect X
    x = np.zeros((resSize, 1))
    for t in range(trainLen):
        u = In[0:inSize, t:t+1]
        x = (1-a)*x + a*np.tanh(Win@np.vstack((1, u)) + W @ x)
        if t >= initLen:
            X[:, t-initLen] = np.vstack((1, x))[:, 0]
    Wout = linalg.solve(np.dot(X, X.T) + reg *
                        np.eye(1+resSize), np.dot(X, Yt.T)).T

    # save
    np.savez_compressed(trained_file, Win=Win, W=W, X=X,
                        Wout=Wout, x=x, Data=Data)

    return (Win, W, X, Wout, x, Data)

def train_1step_GR_for_NCO(main_path, res_params, raw_data_subset, mesh_code, is_update=False):
    (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
     initLen, trainLen, testLen,
     reg, seed_num, conectivity) = res_params
    
    if inSize != 1 or outSize != 1:
        print("In Out Error! @ NCO")
        print((inSize, outSize))
        return
    
    train_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        train_path += "-" + str(res_params[prm_i])
    train_path += "/"

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    train_path += "1step_NCO_train/"
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
        print("make " + str(train_path))

    # mesh_codeのデータがある→読み出し
    trained_file = train_path + str(mesh_code)
    if os.path.isfile(trained_file+".npz") and (not is_update):
        trained_data = np.load(trained_file+".npz")
        (Win, W, X, Wout, x, Data) = (
            trained_data["Win"], trained_data["W"], trained_data["X"], trained_data["Wout"],
            trained_data["x"], trained_data["Data"])
        return (Win, W, X, Wout, x, Data)

    # Train
    Data = raw_data_subset * 10**expIndex
    Data = Data.astype(np.float64)
    # trainは1 timeずつ
    In = Data[0:inSize, 0:trainLen+testLen-1]  # 入力
    Out = Data[0:outSize, 1:trainLen+testLen]  # 出力
    a = leakingRate
    np.random.seed(seed_num)
    Win = (np.random.rand(resSize, 1+inSize) - 0.5) * 2  # -1~1の一様分布
    W = create_sparse_rand_matrix(resSize, resSize, conectivity)
    rhoW = max(linalg.eigh(W)[0])
    W *= spectralRadius / rhoW
    X = np.zeros((1+resSize, trainLen-initLen))
    Yt = Out[0:outSize, initLen:trainLen]  # init ~ train-1でtrain(train-init分)

    
    # run the reservoir with the Data and collect X
    x = np.zeros((resSize, 1))
    for t in range(trainLen):
        u = In[0:inSize, t:t+1]
        x = (1-a)*x + a*np.tanh(Win@np.vstack((1, u)) + W @ x)
        if t >= initLen:
            X[:, t-initLen] = np.vstack((1, x))[:, 0]
    Wout = linalg.solve(np.dot(X, X.T) + reg *
                        np.eye(1+resSize), np.dot(X, Yt.T)).T

    # save
    np.savez_compressed(trained_file, Win=Win, W=W, X=X,
                        Wout=Wout, x=x, Data=Data)

    return (Win, W, X, Wout, x, Data)

def create_one_step_local_area_trained_data(main_path, geo_res_params, nco_res_params, df, Smesh_list,repeat_num=60,is_update=False):
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(geo_res_params[4])  # inSize
    print("Local area trained at " + str(Smesh_list))
    all_dma = get_raw_mesh_array(df)
    dma = cut_mlist(all_dma,Smesh_list)
    
    local_area_path = main_path + "smesh"
    for smesh in Smesh_list:
        local_area_path += "-" + str(smesh)
    local_area_path += "/"
    
    if not os.path.isdir(local_area_path):
        os.mkdir(local_area_path)
        print("make " + str(local_area_path))

    Rlist = get_R_list(dma, gmom, gnl)
    start_time = time.time()
    subsection_time = time.time()
    for index, mesh_code in enumerate(Rlist):
        gml = get_mesh_list(mesh_code, gmom, gnl)
        raw_data_subset = create_subset_from_data_and_mesh_list(df, gml)
        real_data = extract_data_every_n(raw_data_subset,60)
        repeated_data = repeat_data_columns(real_data,repeat_num)
        _ = train_1step_GR(local_area_path, geo_res_params, repeated_data,
                      mesh_code, is_update=is_update)
        
        rate = 100 * index/len(Rlist)
        if sprit_printer(index,len(Rlist),sprit_num=20):
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s")
        subsection_time = time.time()
        
    gnl = get_n_list(nco_res_params[4])  # inSize
    print("Local area trained at " + str(Smesh_list))
    all_dma = get_raw_mesh_array(df)
    dma = cut_mlist(all_dma,Smesh_list)
    
    local_area_path = main_path + "smesh"
    for smesh in Smesh_list:
        local_area_path += "-" + str(smesh)
    local_area_path += "/"
    
    if not os.path.isdir(local_area_path):
        os.mkdir(local_area_path)
        print("make " + str(local_area_path))

    Rlist = get_R_list(dma, gmom, gnl)
    start_time = time.time()
    subsection_time = time.time()
    for index, mesh_code in enumerate(Rlist):
        gml = get_mesh_list(mesh_code, gmom, gnl)
        raw_data_subset = create_subset_from_data_and_mesh_list(df, gml)
        real_data = extract_data_every_n(raw_data_subset,60)
        repeated_data = repeat_data_columns(real_data,repeat_num)
        _ = train_1step_GR_for_NCO(local_area_path, nco_res_params, repeated_data,
                      mesh_code, is_update=is_update)
        rate = 100 * index/len(Rlist)
        if sprit_printer(index,len(Rlist),sprit_num=20):
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s")
        subsection_time = time.time()
    print("Train Data Saved")
    return
#########
#Test
#########

def load_one_step_trained_data(main_path, res_params, mesh_code, train_folder):
    train_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        train_path += "-" + str(res_params[prm_i])
    train_path += "/"+train_folder+"/"

    # read mesh_code
    trained_file = train_path + str(mesh_code)
    if os.path.isfile(trained_file+".npz"):
        trained_data = np.load(trained_file+".npz")
        (Win, W, X, Wout, x, Data) = (
            trained_data["Win"], trained_data["W"], trained_data["X"], trained_data["Wout"],
            trained_data["x"], trained_data["Data"])
        # return (Win, W, X, Wout, x, Data)
        return {"trained_data": (Win, W, X, Wout, x, Data), "path": train_path}
    else:
        print(train_path)
        print("ERROR at " + str(mesh_code))

def test_GR(main_path, res_params, Distance, Rlist_dict):
    start_time = time.time()
    # trainを全てのセルで終えてる前提
    (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
     initLen, trainLen, testLen,
     reg, seed_num, conectivity) = res_params
    
    if inSize != 9 or outSize != 1 or Distance != 1:
        print("In Out Error! @ GR")
        print((inSize, outSize))
        return

    #各testの時に学習するmeshのdict: Rlist_dict(input)

    Rlist = list(Rlist_dict.keys())  # Reservoir Mesh list

    print(str((time.time() - start_time)//1) +
          "s " + "LargeGeoReservoir Start...")

    # (Win,W,X,Wout,Data) main_path,x,u,(Y,UU,XX,In,Out,trainO)全てを格納する
    All_R_dict = {}
    subsection_time = time.time()
    sectiontime = time.time()
    for t, r in enumerate(Rlist):
        tmp = load_one_step_trained_data(main_path, res_params, r, "1step_GR_train")
        if type(tmp) != type({"A": 1}):
            print(tmp)
            return
        (Win, W, X, Wout, x, Data) = tmp["trained_data"]

        # save to All_R_dict
        tmp_dict = {}
        tmp_dict["Win"] = Win
        tmp_dict["W"] = W
        tmp_dict["Wout"] = Wout
        # tmp_dict["Data"] = Data
        tmp_dict["X"] = X

        tmp_dict["x"] = x

        (In, Out) = dis_in_out(
            Data[0:, trainLen:trainLen+testLen], inSize, outSize, Distance)
        tmp_dict["In"] = In
        tmp_dict["Out"] = Out
        tmp_dict["trainO"] = Data[0:outSize, trainLen:trainLen+testLen]

        tmp_dict["u"] = In[0:inSize, 0:1]

        tmp_dict["Y"] = np.zeros((outSize, testLen//Distance))
        tmp_dict["UU"] = np.zeros((outSize, testLen//Distance * Distance))
        tmp_dict["XX"] = np.zeros((resSize, testLen//Distance * Distance))

        All_R_dict[r] = tmp_dict
        
        rate = 100 * t/len(Rlist)
        if sprit_printer(t,len(Rlist),sprit_num=5):
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s and this section time is " "{:.2f}".format(time.time() - sectiontime) + " s")
        subsection_time = time.time()

    print("Load Ended")
    print("{:.2f}".format(time.time() -start_time) + " s passed, this section " + "{:.2f}".format(time.time() - sectiontime) + " s")
    sectiontime = time.time()
    subsection_time = time.time()
    print("Initialize and Compute Geo reservoir...")

    a = leakingRate
    for t in range(testLen//Distance):
        for d_i in range(Distance):
            # Compute Each

            for r in Rlist:
                tmp_dict = All_R_dict[r]  # params

                Win = tmp_dict["Win"]
                W = tmp_dict["W"]
                Wout = tmp_dict["Wout"]

                u = tmp_dict["u"]
                x = tmp_dict["x"]
                UU = tmp_dict["UU"]
                # XX = tmp_dict["XX"]

                tmp1u = np.vstack((1, u))
                x = (1-a)*x + a*np.tanh(Win@tmp1u + W @ x)
                tmp1x = np.vstack((1, x))
                u = Wout@tmp1x
                # u = np.tanh(Wout@tmp1x)
                tmp_dict["x"] = x
                tmp_dict["u"] = u

                # 4D
                # XX[:,t*Distance+d_i] = np.vstack((x))[:,0]
                UU[:, t*Distance+d_i] = u[0:, 0]
                # tmp_dict["XX"] = XX
                tmp_dict["UU"] = UU

            # Self Organize
            for r in Rlist:
                print("Error?")
                return

        # set Y
        for r in Rlist:
            All_R_dict[r]["Y"][:, t] = All_R_dict[r]["u"][0:, 0]

        # for next time
        if t+2 < All_R_dict[Rlist[0]]["In"].shape[1]:
            for r in Rlist:
                In = All_R_dict[r]["In"]
                All_R_dict[r]["u"] = In[0:inSize, t+1:t+2]

        rate = 100*t/(testLen//Distance)
        if sprit_printer(t,(testLen//Distance),sprit_num=20):
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s and this section time is " "{:.2f}".format(time.time() - sectiontime) + " s")
        subsection_time = time.time()

    print("Coputed !!")
    print("{:.2f}".format(time.time() -start_time) + " s passed, this section " + "{:.2f}".format(time.time() - sectiontime) + " s")
    sectiontime = time.time()
    subsection_time = time.time()
    print("Saving...")

    for t, r in enumerate(Rlist):
        tmp_dict = All_R_dict[r]  # params

        Y = tmp_dict["Y"]
        UU = tmp_dict["UU"]
        XX = tmp_dict["XX"]
        Out = tmp_dict["Out"]
        trainO = tmp_dict["trainO"]
        
        test_path = main_path + str(res_params[0])
        for prm_i in range(1,len(res_params)):
            test_path += "-" + str(res_params[prm_i])
        test_path += "/"
        test_path += str(Distance)+"step-test/"
        if not os.path.isdir(test_path):
            os.mkdir(test_path)
            print("make " + str(test_path))
        
        test_file = test_path + str(r)
        
        np.savez_compressed(test_file, Y=Y, UU=UU, XX=XX,
                            Out=Out, trainO=trainO)
        
        rate = 100 * t/len(Rlist)
        if sprit_printer(t,len(Rlist),sprit_num=5):
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s and this section time is " "{:.2f}".format(time.time() - sectiontime) + " s")
        subsection_time = time.time()

    print("All completed")
    print("{:.2f}".format(time.time() -start_time) + " s passed, this section " + "{:.2f}".format(time.time() - sectiontime) + "s")
    return

# NotCoopolateReservoir実行、結果を保存
# Trainingはファイルに保存されてる前提
# Rlist_dictを制限すれば
# subin:  get_Rlist_dict(この子関数であるR_list,GMoM,GNL,(get_mesh_listは中で使ってる)は外に出てる)
# sub_func: load_trained_data, dis_in_out
# in:path,res_params,expIndex,Distance,Rlist_dict
#out: なし


def test_NCOGR(main_path, res_params, Distance, Rlist_dict):
    start_time = time.time()
    # trainを全てのセルで終えてる前提
    (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
     initLen, trainLen, testLen,
     reg, seed_num, conectivity) = res_params
    
    if inSize != 1 or outSize != 1 or Distance != 1:
        print("In Out Error! @ NCO")
        print((inSize, outSize))
        return
    
    #各testの時に学習するmeshのdict: Rlist_dict(input)

    Rlist = list(Rlist_dict.keys())  # Reservoir Mesh list

    print(str((time.time() - start_time)//1) +
          "s " + "NotCoopGeoReservoir Start...")
    
    subsection_time = time.time()
    for t_r, r in enumerate(Rlist):
        tmp = load_one_step_trained_data(main_path, res_params, r, "1step_NCO_train")
        if type(tmp) != type({"A": 1}):
            print(tmp)
            return
        (Win, W, X, Wout, x, Data) = tmp["trained_data"]

        (In, Out) = dis_in_out(
            Data[0:, trainLen:trainLen+testLen], inSize, outSize, Distance)

        trainO = Data[0:outSize, trainLen:trainLen+testLen]

        u = In[0:inSize, 0:1]

        Y = np.zeros((outSize, testLen//Distance))
        UU = np.zeros((outSize, testLen//Distance * Distance))
        XX = np.zeros((resSize, testLen//Distance * Distance))

        a = leakingRate

        # NoCop
        for t in range(testLen//Distance):
            for d_i in range(Distance):
                # Compute
                x = (1-a)*x + a * \
                    np.tanh(Win@np.vstack((1, u)) + W @ x)
                # u = np.tanh(Wout@np.vstack((1, x)))
                u = Wout@np.vstack((1, x))

                # 4D
                # XX[:, t*Distance+d_i] = np.vstack((x))[:, 0]
                UU[:, t*Distance+d_i] = u[0:, 0]

                # Self Organize
                # なし

                # set Y
                Y[:, t] = u[0:, 0]

            # for next time
            if t+2 < In.shape[1]:
                u = In[0:inSize, t+1:t+2]
        
        test_path = main_path + str(res_params[0])
        for prm_i in range(1,len(res_params)):
            test_path += "-" + str(res_params[prm_i])
        test_path += "/"
        test_path += str(Distance)+"step-test-nco/"
        if not os.path.isdir(test_path):
            os.mkdir(test_path)
            print("make " + str(test_path))
        
        test_file = test_path + str(r)
        
        np.savez_compressed(test_file, Y=Y, UU=UU, XX=XX,
                            Out=Out, trainO=trainO)

        rate = 100 * t_r/len(Rlist)
        if sprit_printer(t_r,len(Rlist),sprit_num=10):
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s")
        subsection_time = time.time()

    print("All completed")
    print("{:.2f}".format(time.time() -start_time) + " s passed")
    return

def create_local_gr_test_data(main_path, geo_res_params, nco_res_params, distance, df, Smesh_list):
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(geo_res_params[4])
    print("Local area trained at " + str(Smesh_list))
    all_dma = get_raw_mesh_array(df)
    dma = cut_mlist(all_dma,Smesh_list)
    Rl = get_R_list(dma, gmom, gnl)
    grld = get_Rlist_dict(Rl,gmom,gnl)
    local_area_path = main_path + "smesh"
    for smesh in Smesh_list:
        local_area_path += "-" + str(smesh)
    local_area_path += "/"
    if not os.path.isdir(local_area_path):
        print("path ERROR")
        return
    
    print("Data mesh:" + str(len(dma)))
    print("Reservoir mesh:" + str(len(Rl)))
    test_GR(local_area_path, geo_res_params, distance, grld)
    
    gnl = get_n_list(nco_res_params[4])
    print("Local area trained at " + str(Smesh_list))
    all_dma = get_raw_mesh_array(df)
    dma = cut_mlist(all_dma,Smesh_list)
    Rl = get_R_list(dma, gmom, gnl)
    grld = get_Rlist_dict(Rl,gmom,gnl)
    local_area_path = main_path + "smesh"
    for smesh in Smesh_list:
        local_area_path += "-" + str(smesh)
    local_area_path += "/"
    if not os.path.isdir(local_area_path):
        print("path ERROR")
        return
    print("Data mesh:" + str(len(dma)))
    print("Reservoir mesh:" + str(len(Rl)))
    test_NCOGR(local_area_path, nco_res_params, distance, grld)
    return

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

def get_copy_gr_data(main_path, saved_test_path, geo_res_params,nco_res_params, distance, mesh_list):
  is_existed = True
  for m in mesh_list:
    is_existed = is_existed and check_gr_data_existed(m, main_path, geo_res_params, distance)
  if is_existed:
    print("Existed:" + str(mesh_list))
    print(str(geo_res_params))
    return
  
  save_path = main_path + "gr_data/"
  if not os.path.isdir(save_path):
      os.mkdir(save_path)
      print("make " + str(save_path))
  
  for mesh_code in mesh_list:
    tmp_geo_test_data = load_geo_test_data(saved_test_path, geo_res_params, distance, mesh_code)

    if type(tmp_geo_test_data) != type((1,2)):
      #ERROE
      print(tmp_geo_test_data)
      return tmp_geo_test_data
    
    tmp_nco_test_data = load_nco_test_data(saved_test_path, nco_res_params, distance, mesh_code)

    if type(tmp_nco_test_data) != type((1,2)):
      #ERROE
      print(tmp_nco_test_data)
      return tmp_nco_test_data
    
    tmp_train_data = load_one_step_trained_data(saved_test_path, geo_res_params, mesh_code,"1step_GR_train")
    if type(tmp_train_data) != type({"A": 1}):
        print(tmp_train_data)
        return
    else:
      tmp_train_data = tmp_train_data["trained_data"]
    
    (Win, W, X, Wout, x, Data) = tmp_train_data
    
    (Ygeo,UUgeo,XXgeo,Outgeo,trainOgeo) = tmp_geo_test_data
    
    (Ynco,UUnco,XXnco,Outnco,trainOnco) = tmp_nco_test_data
    
    gr_data_name = save_path + str(geo_res_params[0])
    for prm_i in range(1,len(geo_res_params)):
        gr_data_name += "-" + str(geo_res_params[prm_i])
    gr_data_name += "-" + str(distance) + str(mesh_code)
    # np.savez_compressed(gr_data_name,geo_test_data = tmp_geo_test_data, nco_test_data = tmp_nco_test_data, train_data = tmp_train_data)
    np.savez_compressed(gr_data_name,
                        Win=Win, W=W, X=X,Wout=Wout, x=x, Data=Data,
                        Ygeo=Ygeo, UUgeo=UUgeo, XXgeo=XXgeo,Outgeo=Outgeo, trainOgeo=trainOgeo,
                        Ynco=Ynco, UUnco=UUnco, XXnco=XXnco,Outnco=Outnco, trainOnco=trainOnco)
    print("Saved!" + str(gr_data_name))
    

    
  print("Copy Data Complete!")
  return

if __name__ == '__main__':
    # Variable

    df_path = "./df/"
    main_path = './all_prgrm_output/'
    saved_test_path = './all_prgrm_output/'

    if os.path.isfile(df_path+"df.csv"):
        df = pd.read_csv(df_path+"df.csv")
    else:
        print("NO DF FILE")

    # (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
    #  initLen, trainLen, testLen,
    #  reg, seed_num, conectivity) = res_params
    Smesh_list = [45]
    center_mesh_mat = (24,35) #ikebukuro 533945774 9*9
    d_list=[1]
    is_up = False
    # cnct_list = [0.001, 0.01, 0.1, 1.0]
    # sc_list = [-1, -3, -5, -7, -9, -11, -13]
    cnct_list = [0.001]
    sc_list = [-1]
    
    all_program_start_time = time.perf_counter()
    
    for d in d_list:       
        for cone in cnct_list:
            for sc in sc_list:
                geo_res_params = (sc, 1, 1000, 0.75, 9, 1,
                    24*60, 3*24*60, 2*24*60-60+1,
                    1e-8, 2, cone)
                nco_res_params = (sc, 1, 1000, 0.75, 1, 1,
                    24*60, 3*24*60, 2*24*60-60+1,
                    1e-8, 2, cone)
                distance = d
                
                main_path = './all_prgrm_output/'
                ###################Reservoir
                for_time = time.perf_counter()
                
                start_time = time.perf_counter() #Start
                print("Start Train")

                #TODO repeat_numはファイル名やres_paramsに含めていません。
                create_one_step_local_area_trained_data(main_path,geo_res_params, nco_res_params,df,Smesh_list,repeat_num=60,is_update = is_up)

                print("Save Train Data:")
                display_time(time.perf_counter() - start_time)

                start_time = time.perf_counter()  # Start
                print("Start Test")
                print(str(geo_res_params) + "d:" + str(distance)) 
                print()

                create_local_gr_test_data(main_path, geo_res_params, nco_res_params, distance, df, Smesh_list)

                print("Save Test Data:" )
                display_time(time.perf_counter() - start_time)
                
                
                #####################Print
                main_path = './'
                
                start_time = time.perf_counter()  # Start
                print("Start mse create")
                
                # create_mse_maps(main_path, saved_test_path, geo_res_params, distance, df, Smesh_list)
                
                print("Save mse create Test Data:" )
                display_time(time.perf_counter() - start_time)
                
                start_time = time.perf_counter()  # Start
                print("Start copy gr data")
                
                copy_gr_data(main_path, saved_test_path, geo_res_params, nco_res_params, distance, Smesh_list, center_mesh_mat)
                
                print("Save copy gr data Test Data:" )
                display_time(time.perf_counter() - start_time)
                
                print("For Time:")
                display_time(time.perf_counter() - for_time)
                print(str(geo_res_params) + "d:" + str(distance)) 
                print()
    
    print("All Time:")
    display_time(time.perf_counter() - all_program_start_time)