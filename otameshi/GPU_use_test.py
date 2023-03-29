import numpy as np
import cupy as cp

@profile
def main():
    A_cpu = np.random.randn(1000, 2000)
    B_cpu = np.random.randn(2000, 3000)
    for _ in range(2000):
        # numpy を使ってCPUで行列積を計算する
        AB_cpu = np.dot(A_cpu, B_cpu)

        # np.ndarray から GPU 上のメモリにデータを移動する
        A_gpu = cp.asarray(A_cpu)
        B_gpu = cp.asarray(B_cpu)

        # cupy を使って GPU で行列積を計算する
        AB_gpu = cp.dot(A_gpu, B_gpu)

        # メインメモリ上にデータを移動する
        AB_cpu2 = AB_gpu.get()  # AB_cpu2 は np.ndarray 型
        
        if _ % 100 == 0:
            print(_//100)

if __name__ == "__main__":
    main()