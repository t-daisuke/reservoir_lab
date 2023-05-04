import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


def save_figures_to_pdf(t, distance, save_folder, pdf_filename):
    with PdfPages(pdf_filename) as pdf:
        for d in range(distance):
            img_path = os.path.join(save_folder, f'd_{d}_t_{t}.png')
            img = Image.open(img_path)
            pdf.savefig(fig=plt.figure(figsize=(18, 12)), bbox_inches=img.getbbox())
            plt.close()
            
def main():
    # 保存された図をPDFにまとめる
    t = 15
    distance = 32  # distance の値を適切な値に変更してください
    save_folder = './debug_fig/'
    pdf_filename = f't{t}_d{distance}figures.pdf'
    save_figures_to_pdf(t, distance, save_folder, pdf_filename)
    

if __name__ == '__main__':
    main()