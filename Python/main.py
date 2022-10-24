import numpy as np
import cv2
import copy
import os
from dark_channel_prior import HazelRemove as darkchannel
from guided_filter import HazelRemove as darkchannel_guided_filter
from histogram_equalization import Histogram_Eqaulization as histogram_equalization

if __name__ == '__main__':
    path_name = './db/IEI2019'
    file = os.listdir('./db/IEI2019')
    for fname in file:
        print(fname)
        pic_name = fname[:-4]
        img = cv2.imread(f'./db/IEI2019/{fname}')
        model1 = darkchannel(img=img)
        model2 = darkchannel_guided_filter(img=img)

        result1 = model1.remove_hazel_DC()
        result2 = model2.remove_hazel_DC()

        cv2.imwrite(f'./result/1021/DCP_{pic_name}.jpg', result1)
        cv2.imwrite(f'./result/1021/DCP_GF_{pic_name}.jpg', result2)
        cv2.imwrite(f'./result/1021/HE_{pic_name}.jpg', histogram_equalization(img))