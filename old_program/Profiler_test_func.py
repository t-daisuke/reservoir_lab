# -*- coding: utf-8 -*-
# Ver0324

# import
from line_profiler import LineProfiler
#ref
#https://qiita.com/aratana_tamutomo/items/aa3b723a3dd7a44e45d6

import pandas as pd
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

import os
import time

from train_func import *
from print_func import *

"""TEST FUNC"""

"""# Test

## 子関数

### load_trained_data
"""

# Trainingによって保存されたものを呼び出す
# in: expIndex,mesh_code,inSize
#out: {"trained_data":(Win,W,X,Wout,x,Data), "path":train_path}


def load_trained_data(main_path, res_params, mesh_code):
    train_path = main_path + str(res_params[0])
    for prm_i in range(1,len(res_params)):
        train_path += "-" + str(res_params[prm_i])
    train_path += "/train/"

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


def dis_in_out(Data, inSize, outSize, dis):
    W = int(np.ceil(Data.shape[1]/dis))-1  # In,Outの横の長さ
    In = np.zeros((inSize, W))  # Dataの横の長さをdis分割してお尻だけ削る
    Out = np.zeros((outSize, W))  # 頭だけ削る
    for i in range(W):
        In[0:inSize, i] = Data[0:inSize, i*dis]  # 頭からdisおきに格納
        Out[0:outSize, i] = Data[0:outSize, i*dis + dis]  # Inに対してdis後が正解
    return (In, Out)


"""## メイン関数

### LGR
"""

# GeoReservoir実行、結果を保存
# Trainingはファイルに保存されてる前提
# Rlist_dictを制限すれば
# subin:  get_Rlist_dict(この子関数であるR_list,GMoM,GNL,(get_mesh_listは中で使ってる)は外に出てる)
# sub_func: load_trained_data, dis_in_out
# in:main_path,res_params,expIndex,Distance,Rlist_dict
#out: なし


def test_GR(main_path, res_params, Distance, Rlist_dict):
    start_time = time.time()
    # trainを全てのセルで終えてる前提
    (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
     initLen, trainLen, testLen,
     reg, seed_num, conectivity) = res_params

    #各testの時に学習するmeshのdict: Rlist_dict(input)

    Rlist = list(Rlist_dict.keys())  # Reservoir Mesh list

    print(str((time.time() - start_time)//1) +
          "s " + "LargeGeoReservoir Start...")

    # (Win,W,X,Wout,Data) main_path,x,u,(Y,UU,XX,In,Out,trainO)全てを格納する
    All_R_dict = {}
    subsection_time = time.time()
    sectiontime = time.time()
    for t, r in enumerate(Rlist):
        tmp = load_trained_data(main_path, res_params, r)
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
        if sprit_printer(t,len(Rlist),sprit_num=10):
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
                tmp_dict["x"] = x
                tmp_dict["u"] = u

                # 4D
                # XX[:,t*Distance+d_i] = np.vstack((x))[:,0]
                UU[:, t*Distance+d_i] = u[0:, 0]
                # tmp_dict["XX"] = XX
                tmp_dict["UU"] = UU

            # Self Organize
            for r in Rlist:
                u = All_R_dict[r]["u"]
                mlist = Rlist_dict[r]

                # 周囲のmのReservorから値を取得
                for i, m in enumerate(mlist):
                    if i == 0:
                        # index = 0の時は自分自身
                        continue
                    elif m < 0:
                        # m is Not Reservoir mesh
                        continue
                    else:
                        u[i, 0] = All_R_dict[m]["u"][0, 0]

                tmp_dict["u"] = u

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
        
        # np.savez_compressed(test_file, Y=Y, UU=UU, XX=XX,
        #                     Out=Out, trainO=trainO)
        
        rate = 100 * t/len(Rlist)
        if sprit_printer(t,len(Rlist),sprit_num=10):
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

    #各testの時に学習するmeshのdict: Rlist_dict(input)

    Rlist = list(Rlist_dict.keys())  # Reservoir Mesh list

    print(str((time.time() - start_time)//1) +
          "s " + "NotCoopGeoReservoir Start...")
    
    subsection_time = time.time()
    for t_r, r in enumerate(Rlist):
        tmp = load_trained_data(main_path, res_params, r)
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
        
        # np.savez_compressed(test_file, Y=Y, UU=UU, XX=XX,
        #                     Out=Out, trainO=trainO)

        rate = 100 * t_r/len(Rlist)
        if sprit_printer(t_r,len(Rlist),sprit_num=40):
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s")
        subsection_time = time.time()
        
        if (time.time() -start_time) > 60*3:
            print("Temporaly END")
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s")
            return

    print("All completed")
    print("{:.2f}".format(time.time() -start_time) + " s passed")
    return

def create_gr_test_data(main_path, res_params, distance, df):
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(res_params[4])
    dma = get_raw_mesh_array(df) #すべてのメッシュ
    # dma = cut_mlist(dma,[45]) #Smesh == 45に固定
    Rl = get_R_list(dma, gmom, gnl)
    grld = get_Rlist_dict(Rl,gmom,gnl)
    
    print("Data mesh:" + str(len(dma)))
    print("Reservoir mesh:" + str(len(Rl)))
    test_GR(main_path, res_params, distance, grld)
    return

def create_nco_test_data(main_path, res_params, distance, df):
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(res_params[4])
    dma = get_raw_mesh_array(df) #すべてのメッシュ
    # dma = cut_mlist(dma,[45]) #Smesh == 45に固定
    Rl = get_R_list(dma, gmom, gnl)
    grld = get_Rlist_dict(Rl,gmom,gnl)
    
    print("Data mesh:" + str(len(dma)))
    print("Reservoir mesh:" + str(len(Rl)))
    test_NCOGR(main_path, res_params, distance, grld)
    return
"""TEST FUNC END"""

if __name__ == '__main__':
    # Variable

    df_path = "./df/"
    main_path = './'

    if os.path.isfile(df_path+"df.csv"):
        df = pd.read_csv(df_path+"df.csv")
    else:
        print("NO DF FILE")

    # (expIndex, leakingRate, resSize, spectralRadius, inSize, outSize,
    #  initLen, trainLen, testLen,
    #  reg, seed_num, conectivity) = res_params
    res_params = (-9.5, 1, 1000, 0.75, 9, 9,
                  24*60, 3*24*60, 2*24*60-60+1,
                  1e-8, 2, 0.01)
    distance = 30
    is_up = False
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(res_params[4])
    dma = get_raw_mesh_array(df)
    Rl = get_R_list(dma, gmom, gnl)
    grld = get_Rlist_dict(Rl,gmom,gnl)
    print("Data mesh:" + str(len(dma)))
    print("Reservoir mesh:" + str(len(Rl)))
    

    prof = LineProfiler()
    prof.add_function(test_NCOGR)
    prof.runcall(test_NCOGR,main_path, res_params, distance, grld)
    prof.print_stats()
    
    profiler_path = "./profiler/nco_profile"
    date = get_current_date(profiler_path)
    filename = f"{date}-v1"  # 初期のファイル名

    # すでに同じ名前のファイルがある場合、新しいファイル名を作成する
    if os.path.isfile(filename):
        version = 1
        while True:
            version += 1
            new_filename = f"{date}-v{version}"
            if not os.path.isfile(new_filename):
                filename = new_filename
                break

    # ファイルを保存する
    with open(filename, "w") as f:
        prof.print_stats(f)