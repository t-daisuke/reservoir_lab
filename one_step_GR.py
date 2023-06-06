#Ver June

# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from train_func import *

import os
import time

import concurrent.futures
import threading

import pdb


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

def create_local_area_trained_data(main_path, res_params, df, Smesh_list,is_update=False):
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(res_params[4])  # inSize
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
        _ = train_1step_GR(local_area_path, res_params, raw_data_subset,
                      mesh_code, is_update=is_update)
        
        _ = train_1step_GR_for_NCO(local_area_path, res_params, raw_data_subset,
                      mesh_code, is_update=is_update)
        rate = 100 * index/len(Rlist)
        if sprit_printer(index,len(Rlist),sprit_num=20):
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s")
        subsection_time = time.time()
    print("Train Data Saved")
    return