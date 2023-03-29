# -*- coding: utf-8 -*-
# Ver0324

# import
import pandas as pd
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

import os
import time

"""TRAIN FUNC"""

# Smeshのリストの分をrawから選定
# in: raw_mesh(df.なんとか)


def cut_mlist(m_array, Smesh_list=[45]):
    mset = set(m_array)
    # 5339-{45,46,35}-{00~99}-{1~4}
    Pmesh = 5339
    mesh_list = []

    for sm in Smesh_list:
        for i in range(100):
            tmp_mesh = Pmesh*10000 + sm*100 + i
            for j in [1, 2, 3, 4]:
                mesh = tmp_mesh*10 + j
                if mesh in mset:
                    mesh_list.append(mesh)

    return mesh_list


"""### get_raw_mesh_array"""


def get_raw_mesh_array(df):
    return df["mesh_code"].unique()


"""## Get_mesh_list(New)

*   index_tpl = GMoM["dic"][mesh_O]で、GMoMにないものを求めるとエラーになる→GMoMは大きめに求める
*   このときのエラー処理ができてない

### 子関数

#### get_matrix_of_mesh
"""

# 長方形のS_meshをもつ、1/2地域メッシュまでのメッシュを配列にしたものと、mesh: indexのdictをかえす
#NOTE: 十分に大きいメッシュコードを想定しなければいけない
#input: Smesh_list([[]], 長方形)
#output: {mat: matrix, dic: dict}
# index → mesh と mesh → index


def get_matrix_of_mesh(Smesh_l=[[54, 55, 56, 57], [44, 45, 46, 47], [34, 35, 36, 37], [24, 25, 26, 27]], Pmesh=5339, Amesh_n=[[3, 4], [1, 2]]):
    tmp_l = []
    for sm_tate in Smesh_l:
        for sub_tm in [i for i in range(9, -1, -1)]:
            # 9X; 8X; ,,, 0X
            for a_l in Amesh_n:
                #3434 or 1212

                # よこ
                for sm in sm_tate:
                    # 45...46...
                    for tm in range(sub_tm*10, sub_tm*10+10):
                        # X0 ... X9
                        tmp1 = Pmesh*100 + sm
                        tmp2 = tmp1*100 + tm
                        for a in a_l:
                            tmp3 = tmp2*10 + a
                            tmp_l.append(tmp3)

    mat = np.array(tmp_l).reshape(20*len(Smesh_l), 20*len(Smesh_l[0]))

    dic = {}
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            dic[mat[y, x]] = (y, x)

    return {"mat": mat, "dic": dic}


"""#### get_n_list"""

# 入力次元を与えたら、そのまわりのneighbor_listを返す
# in: dim(int 入力次元,5,1,9,25,(奇数)**2)必要に応じて加筆
# out: neighbor_list([(int y, int x)])
#note: 中心セルは必ず頭になる
#ex: [(0, 0), (-1, 0), (0, 1), (0, -1), (1, 0)]


def get_n_list(dim):
    n_list = [(0, 0)]  # はじめはセルO(自分自身)
    if dim == 1:
        return n_list
    if dim == 5:
        # 例
        n_list.append((-1, 0))  # N
        n_list.append((0, 1))  # E
        n_list.append((0, -1))  # W
        n_list.append((1, 0))  # S
        return n_list

    sqrt_dim = int(np.sqrt(dim))  # dimは奇数の**2を想定, sqrtは奇数
    abs_mesh = sqrt_dim//2  # -abs_mesh ... abs_meshになる
    for y in range(-abs_mesh, abs_mesh+1, 1):
        for x in range(-abs_mesh, abs_mesh+1, 1):
            if(y, x) == (0, 0):
                continue
            n_list.append((y, x))
    return n_list


"""### get_mesh_list"""

# 中心メッシュコードから、まわりのメッシュコードを返す
# sub_in: get_matrix_of_mesh(), get_n_list()
# in: mesh_O(int 中心メッシュコード), dim  入力次元のメッシュが中心メッシュに対していくつのindexか)
#out: mesh_list
#note: 上の子関数の出力はこの関数のそとから出力でもらう


def get_mesh_list(mesh_O, GMoM, GNL):
    # mesh → index
    index_tpl = GMoM["dic"][mesh_O]

    # index → neighbor_index_list
    index_tpl_list = []
    for n_index_tpl in GNL:
        index_tpl_list.append(
            (index_tpl[0]+n_index_tpl[0], index_tpl[1]+n_index_tpl[1]))

    # neighbor_index_list → mesh_list
    mesh_list = []
    for index_tpl in index_tpl_list:
        mesh_list.append(GMoM["mat"][index_tpl[0], index_tpl[1]])

    return mesh_list


"""## resercoir_list, resercoir_list_dict(New)

### get_reservoir_list
"""

# get_mesh_listで作成されるメッシュのうち、周りのメッタシュのデーがmesh_arrayにある=リザバーにすることができるメッシュのリストを返す
# sub_in: get_mesh_list()、直接的にはget_matrix_of_mesh(), get_n_list()
# in: data_mesh_array(arrayになる df["mesh_code"].unique()) データがあるmeshのarray
#out: Reservoir_list
# note: 上のget_mesh_list()やその子関数の出力が必要


def get_R_list(data_mesh_array, GMoM, GNL):
    r_list = []
    for m in data_mesh_array:
        tmp_l = get_mesh_list(m, GMoM, GNL)
        if not (False in np.in1d(np.array(tmp_l), data_mesh_array)):
            r_list.append(tmp_l[0])
    return r_list


"""### get_reservoir_list_dict(NEW)"""

# リザバーメッシュのうち(周りにデータがあるメッシュ){mesh: Rmesh, Rmesh, -Rmesh, Rmesh}みたいなdictを作成
# TestのSelf OrgnizeでReservoirの返り値を自分の予想でするか(自分自身のセルと、データしかないせる)、ほかのRからとるかの時に使う
# in: reservoir_list
#out: self_orgnize_dict


def get_Rlist_dict(R_list, GMoM, GNL):
    SOdict = {}
    R_set = set(R_list)
    for r in R_list:
        m_l = get_mesh_list(r, GMoM, GNL)
        tmp_l = []
        for m in m_l:
            if not m in R_set:
                # Dataのみのセル→ひとつでもあるとNTR
                tmp_l.append(-m)
            else:
                # TR
                tmp_l.append(m)

        SOdict[r] = tmp_l

    return SOdict


"""## メッシュの要素の調査(in_dim: 9)

### matrix create
"""


def create_fig_pop_mesh_map(df):
    gmom = get_matrix_of_mesh()
    gnl = get_n_list(9)
    # Out_put_matrix
    gmom_mat = gmom["mat"]
    exist_mat = gmom_mat.copy()
    # Data
    grmr = get_raw_mesh_array(df)
    set_grmr = set(grmr)
    # Reservoir
    grl = get_R_list(grmr, gmom, gnl)
    set_grl = set(grl)

    for y in range(gmom_mat.shape[0]):
        for x in range(gmom_mat.shape[1]):
            tmp = gmom_mat[y, x]
            if tmp in set_grmr:
                if tmp in set_grmr:
                    exist_mat[y, x] = 2  # R
                else:
                    exist_mat[y, x] = 1  # D
            else:
                exist_mat[y, x] = 0  # ND

    """### fig"""

    # Fig Params
    plt.rcParams["font.size"] = 15
    Figsize = (25, 12.5)
    plt.rcParams['image.cmap'] = 'bwr'

    fig = plt.figure(figsize=Figsize)
    ax = fig.add_subplot(111, title="mesh_map")
    im = ax.imshow(exist_mat)
    c = fig.colorbar(im)

    # grid
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
    return

"""TRAIN FUNC END"""

"""TEST FUNC"""

"""# Test

## 子関数

### load_trained_data
"""

# Trainingによって保存されたものを呼び出す
# in: expIndex,mesh_code,inSize
#out: {"Tdata":(Win,W,X,Wout,x,Data), "path":train_path}


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
        return {"Tdata": (Win, W, X, Wout, x, Data), "path": train_path}
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
        (Win, W, X, Wout, x, Data) = tmp["Tdata"]

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
        if rate*10000//1 % 100000 == 0:
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
                x = (1-a)*x + a*np.tanh(np.dot(Win, tmp1u) + np.dot(W, x))
                tmp1x = np.vstack((1, x))
                u = np.dot(Wout, tmp1x)
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
        if rate*10000//1 % 100000 == 0:
            print("{:.2f}".format(rate) + "% done " + "{:.2f}".format(time.time() -start_time) + " s passed, this subset needs " + "{:.2f}".format(time.time() - subsection_time)
                  + " s and this section time is " "{:.2f}".format(time.time() - sectiontime) + " s")
        subsection_time = time.time()

    print("Coputed !!")
    print("{:.2f}".format(time.time() -start_time) + " s passed, this section " + "{:.2f}".format(time.time() - sectiontime) + " s")
    sectiontime = time.time()
    subsection_time = time.time()
    print("Saving...")

    for r in Rlist:
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
        if rate*10000//1 % 100000 == 0:
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


def test_NCoGR(path, res_params, expIndex, Distance, Rlist_dict):
    s_time = time.time()
    print(str(s_time) + "s " + "Programe Start...")
    # trainを全てのセルで終えてる前提
    (leakingRate, resSize, spectralRadius, inSize, outSize,
     initLen, trainLen, testLen, reg, seed_num) = res_params

    #各testの時に学習するmeshのdict: Rlist_dict(input)

    Rlist = list(Rlist_dict.keys())  # Reservoir Mesh list

    now_time = time.time()
    print(str((now_time - s_time)//1) + "s " + "NotCoopGeoReservoir Start...")

    for t_r, r in enumerate(Rlist):
        subsection_time = time.time()

        tmp = load_trained_data(path, expIndex, r, inSize, seed_num)
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
                    np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
                u = np.dot(Wout, np.vstack((1, x)))

                # 4D
                XX[:, t*Distance+d_i] = np.vstack((x))[:, 0]
                UU[:, t*Distance+d_i] = u[0:, 0]

                # Self Organize
                # なし

                # set Y
                Y[:, t] = u[0:, 0]

            # for next time
            if t+2 < In.shape[1]:
                u = In[0:inSize, t+1:t+2]

        Test_path = path+"NCo/"
        if not os.path.isdir(Test_path):
            os.mkdir(Test_path)
        Test_path = Test_path+"seed"+str(seed_num)+"/"
        if not os.path.isdir(Test_path):
            os.mkdir(Test_path)
        Test_path = Test_path+str(r)+"/"
        if not os.path.isdir(Test_path):
            os.mkdir(Test_path)
        Test_path = Test_path+"e"+str(expIndex)+"/"
        if not os.path.isdir(Test_path):
            os.mkdir(Test_path)
        Test_path = Test_path+"C"+str(inSize)+"/"
        if not os.path.isdir(Test_path):
            os.mkdir(Test_path)
        Test_path = Test_path + str(Distance) + "step/"
        if not os.path.isdir(Test_path):
            os.mkdir(Test_path)
        test_file = Test_path + "test_data"

        np.savez_compressed(test_file, Y=Y, UU=UU, XX=XX,
                            Out=Out, trainO=trainO)

        rate = (10000*(t_r+1)) / len(Rlist) // 1 / 100

        print(str((time.time() - subsection_time) * 100 // 1 / 100) +
              "s "+str(rate) + "% done")

    now_time = time.time()
    print(print(str((now_time - s_time)//1) + "s " + "All Completed"))
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
    test_NCoGR(main_path, res_params, distance, grld)
    return
"""TEST FUNC END"""
