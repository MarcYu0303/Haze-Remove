"""
Created by Yu Ran
2022/10/14
program of question 1 in project 1
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plot_histogram(pic, pic_name):
    r, g, b = pic.split()

    plt.figure(f'{pic_name} Red Channel')
    ar = np.array(r).flatten()
    plt.hist(ar, bins=256, density=True, facecolor='r', edgecolor='r')
    plt.savefig(f'./result/{pic_name}_red_hist.jpg')
    plt.title(f'{pic_name} Red Channel')
    plt.show()

    plt.figure(f'{pic_name} Green Channel')
    ag = np.array(g).flatten()
    plt.hist(ag, bins=256, density=True, facecolor='g', edgecolor='g')
    plt.savefig(f'./result/{pic_name}_green_hist.jpg')
    plt.title(f'{pic_name} Green Channel')
    plt.show()

    plt.figure(f'{pic_name} Blue Channel')
    ab = np.array(b).flatten()
    plt.hist(ab, bins=256, density=True, facecolor='b', edgecolor='b')
    plt.savefig(f'./result/{pic_name}_blue_hist.jpg')
    plt.title(f'{pic_name} Blue Channel')
    plt.show()


if __name__ == '__main__':
    # file_dirct = './db/IEI2019/'
    # pic_names = ['H22.png', 'H26.jpg', 'R1.jpg']
    #
    # for i in range(len(pic_names)):
    #     dirct = file_dirct + pic_names[i]
    #     pic = Image.open(dirct)
    #     plot_histogram(pic, pic_names[i][:-4])
    pic = Image.open('./result/1024/H26_HE.jpg')
    plot_histogram(pic, 'H26')


