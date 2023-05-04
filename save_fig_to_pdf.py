import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from matplotlib.transforms import Bbox

# def save_figures_to_pdf(t, distance, save_folder, pdf_filename):
#     with PdfPages(pdf_filename) as pdf:
#         for d in range(distance):
#             img_path = os.path.join(save_folder, f'd_{d}_t_{t}.png')
#             img = Image.open(img_path)
#             bbox = Bbox.from_bounds(*img.getbbox())
            
#             fig, ax = plt.subplots(figsize=(18, 12))
#             ax.imshow(img)
#             ax.axis('off')
#             pdf.savefig(fig, bbox_inches=bbox)
#             plt.close(fig)
def save_figures_to_pdf(t, distance, save_folder, pdf_filename):
    with PdfPages(pdf_filename) as pdf:
        for d in range(distance):
            img_path = os.path.join(save_folder, f'd_{d}_t_{t}.png')
            img = Image.open(img_path)
            
            dpi = 100
            width, height = img.size
            figsize = width / dpi, height / dpi
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.imshow(np.asarray(img))
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
            plt.close(fig)


            
def main():
    # 保存された図をPDFにまとめる
    t = 16
    distance = 32  # distance の値を適切な値に変更してください
    save_folder = './debug_fig/'
    pdf_filename = f't{t}_d{distance}figures.pdf'
    save_figures_to_pdf(t, distance, save_folder, pdf_filename)
    

if __name__ == '__main__':
    main()