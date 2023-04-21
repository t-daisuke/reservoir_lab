# import
import pandas as pd
import os
from train_func import *
from print_func import *

def main():
    saved_folder = "./"
    cnct_list = [0, 1, 2, 3]
    scaling_list = [-1, -3, -5, -7, -9, -11, -13]
    distance = 32
    names = ["diff", "geo", "nco"]
    res_params = [-9, 1, 100, 0.75, 9, 9,
                  24 * 60, 3 * 24 * 60, 2 * 24 * 60 - 60 + 1,
                  1e-8, 2, 10 ** (-1.0 * 3)]

    figsize, cmap_mse, cmap = set_graph_params()

    fig, axs = plt.subplots(len(scaling_list), len(cnct_list), figsize=(figsize[0], figsize[1] * len(scaling_list)))
    fig.subplots_adjust(bottom=0.2, wspace=0.3, hspace=0.3)

    for j, s in enumerate(scaling_list):
        for i, c in enumerate(cnct_list):
            res_params[0] = s
            res_params[11] = 10 ** (-1.0 * c)

            _, map = load_mse_map(names[1], saved_folder, res_params, distance)

            ax = axs[j][i]
            im = plot_graph(ax, map, cmap_mse, j, i, scaling_list, cnct_list, res_params, distance, name=names[1])

    cax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
    fig.colorbar(im, cax=cax, orientation='horizontal')

    plt.show()

if __name__ == '__main__':
    main()

    
    