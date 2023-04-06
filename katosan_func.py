import pandas as pd
import numpy as np
import os
from train_func import *

def train_GR_k(main_path, res_params, raw_data_subset, mesh_code, is_update=False):
    (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
     initLen, trainLen, testLen,
     reg, seed_num, conectivity) = res_params
    np.random.seed(seed_num)
    
    train_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        train_path += "-" + str(res_params[prm_i])
    train_path += "/"

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    train_path += "train/"
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
    In = Data[0:inSize, 0:trainLen+testLen-1]
    Out = Data[0:outSize, 1:trainLen+testLen]
    inputScaled = In
    teacherScaled = Out
    
    ####KATOSAN
    
    # Setting input weight -1~1, uniform distribution
    Win = np.random.rand(resSize, inSize) * 2 - 1
    # Start setting reservoir weight
    W = create_sparse_rand_matrix(resSize, resSize, conectivity)
    # Compute eigenvalues w and eigenvectors v
    w, v = np.linalg.eig(W)
    # print("w: {0}".format(w))
    # Setting Spectral radius
    W = W * spectralRadius / max(abs(w))
    
     # training Wout
    # xx as single reservoir state, XX as reservoir states matrix when training
    XX = np.zeros((resSize, np.shape(inputScaled)[1]))
    print('shape of XX: {0}'.format(np.shape(XX)))
    xx = np.zeros((resSize, 1))
    print('shape of xx: {0}'.format(np.shape(xx)))
    for t in range(np.shape(inputScaled)[1]):
        # need to reshape otherwise shape of u would be (inSize, )
        # We want (inSize, 1)
        # Be smart, Python!!!
        u = inputScaled[:, t].reshape(inSize, 1)
        xx = (1 - leakingRate)*xx + leakingRate*np.tanh(np.dot(Win, u) + np.dot(W, xx)) + (0 * (np.random.rand(resSize, 1) - 0.5))
        XX[:, t] = xx[:, 0]
    allX = XX
    # ignore initial state
    transient = 100
    XX = XX[:, transient:]
    # compute Wout
    Wout = np.dot(np.arctanh(teacherScaled[:, transient:]), np.linalg.pinv(XX))
    
    # save
    X = allX
    x = XX
    np.savez_compressed(trained_file, Win=Win, W=W, X=X,
                        Wout=Wout, x=x, Data=Data)

    return (Win, W, X, Wout, x, Data)