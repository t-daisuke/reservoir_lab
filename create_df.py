import random
import pandas as pd
import numpy as np
import os


def main():
    df_path = "./KDDI/df/"
    path = './KDDI/'

    if not os.path.isdir(df_path):
        os.mkdir(df_path)
    if os.path.isfile(df_path+"df.csv") : print("DF already exists")
    else:
        file = []
        file.append(path)
        file.append(path + 'KLD100101_1.csv')
        file.append(path + 'KLD100102_1.csv')
        file.append(path + 'KLD100103_1.csv')
        file.append(path + 'KLD100104_1.csv')
        file.append(path + 'KLD100105_1.csv')
        file.append(path + 'KLD100106_1.csv')

        df1 = pd.read_csv(file[1], header=0)
        # ***は10人以下なので0~10人にした
        for i in range(len(df1)):
            if df1.iat[i, 7] == "***":
                df1.iat[i, 7] = random.uniform(0, 10)
            else:
                df1.iat[i, 7] = float(df1.iat[i, 7])

            if df1.iat[i, 8] == "***":
                df1.iat[i, 8] = random.uniform(0, 10)
            else:
                df1.iat[i, 8] = float(df1.iat[i, 8])
        df1["sum_population"] = df1["stay_pred_population"] + \
            df1["move_pred_population"]

        df2 = pd.read_csv(file[2], header=0)
        for i in range(len(df2)):
            if df2.iat[i, 7] == "***":
                df2.iat[i, 7] = random.uniform(0, 10)
            else:
                df2.iat[i, 7] = float(df2.iat[i, 7])

            if df2.iat[i, 8] == "***":
                df2.iat[i, 8] = random.uniform(0, 10)
            else:
                df2.iat[i, 8] = float(df2.iat[i, 8])
        df2["sum_population"] = df2["stay_pred_population"] + \
            df2["move_pred_population"]

        df3 = pd.read_csv(file[3], header=0)
        for i in range(len(df3)):
            if df3.iat[i, 7] == "***":
                df3.iat[i, 7] = random.uniform(0, 10)
            else:
                df3.iat[i, 7] = float(df3.iat[i, 7])

            if df3.iat[i, 8] == "***":
                df3.iat[i, 8] = random.uniform(0, 10)
            else:
                df3.iat[i, 8] = float(df3.iat[i, 8])
        df3["sum_population"] = df3["stay_pred_population"] + \
            df3["move_pred_population"]

        df4 = pd.read_csv(file[4], header=0)
        for i in range(len(df4)):
            if df4.iat[i, 7] == "***":
                df4.iat[i, 7] = random.uniform(0, 10)
            else:
                df4.iat[i, 7] = float(df4.iat[i, 7])

            if df4.iat[i, 8] == "***":
                df4.iat[i, 8] = random.uniform(0, 10)
            else:
                df4.iat[i, 8] = float(df4.iat[i, 8])
        df4["sum_population"] = df4["stay_pred_population"] + \
            df4["move_pred_population"]

        df5 = pd.read_csv(file[5], header=0)
        for i in range(len(df5)):
            if df5.iat[i, 7] == "***":
                df5.iat[i, 7] = random.uniform(0, 10)
            else:
                df5.iat[i, 7] = float(df5.iat[i, 7])

            if df5.iat[i, 8] == "***":
                df5.iat[i, 8] = random.uniform(0, 10)
            else:
                df5.iat[i, 8] = float(df5.iat[i, 8])
        df5["sum_population"] = df5["stay_pred_population"] + \
            df5["move_pred_population"]

        df6 = pd.read_csv(file[6], header=0)
        for i in range(len(df6)):
            if df6.iat[i, 7] == "***":
                df6.iat[i, 7] = random.uniform(0, 10)
            else:
                df6.iat[i, 7] = float(df6.iat[i, 7])

            if df6.iat[i, 8] == "***":
                df6.iat[i, 8] = random.uniform(0, 10)
            else:
                df6.iat[i, 8] = float(df6.iat[i, 8])
        df6["sum_population"] = df6["stay_pred_population"] + \
            df6["move_pred_population"]

        df = df1.copy()
        df = df.append(df2)
        df = df.append(df3)
        df = df.append(df4)
        df = df.append(df5)
        df = df.append(df6)
        df.to_csv(df_path+"df.csv", index=False, header=True)
        

if __name__ == '__main__':
    main()
