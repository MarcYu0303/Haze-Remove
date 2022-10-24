"""
Created by Yu Ran
2022/10/11
"""
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt


class HazelRemove():
    def __init__(self, img):
        self.img = img
        self.filter_length = 15
        self.darkImg = self.get_dark_channel(img)
        self.num_pixels = len(img) * len(img[0])
        self.B_A, self.G_A, self.R_A = self.get_A_val()
        self.img_shape = img[:, :, 0].shape

    def display_dark_channel(self):
        cv2.imshow('dark img', self.darkImg)
        cv2.waitKey(0)

    def get_dark_channel(self, img):
        darkImg = np.min(img, 2)
        # darkImg = self.min_filter(darkImg, self.filter_length)
        darkImg = cv2.erode(darkImg, (90, 90))
        return darkImg

    def get_A_val(self):
        hist = cv2.calcHist([self.darkImg, 0], [0], None, [256], [0, 256])
        count = 0
        for i in range(254, 0, -1):
            count += hist[i]
            if count >= (self.num_pixels * 0.001):
                break
        threshold = i + 1
        max_element = np.array([0, 0, 0])

        for i in range(len(self.darkImg)):
            for j in range(len(self.darkImg[0])):
                if self.darkImg[i, j] >= threshold:
                    for channel in range(3):
                        max_element[channel] = max(max_element[channel], self.img[i, j, channel])
        max_element[max_element > 240] = 240
        print('A value for channel B, G, A', max_element[0], max_element[1], max_element[2])
        return max_element[0], max_element[1], max_element[2]

    def remove_hazel_DC(self, w=0.95):
        A = [self.B_A, self.G_A, self.R_A]
        output = np.zeros(self.img.shape)
        normalized_img = copy.copy(self.img).astype(np.float64)
        normalized_img[:, :, 0] = normalized_img[:, :, 0] / 255
        normalized_img[:, :, 1] = normalized_img[:, :, 1] / 255
        normalized_img[:, :, 2] = normalized_img[:, :, 2] / 255
        normalized_dark_img = self.get_dark_channel(img=normalized_img)
        for channel in range(3):
            tx = np.ones(self.img_shape) - w * normalized_dark_img
            tx[tx < 0.1] = 0.1
            temp1 = self.img[:, :, channel] - np.ones(self.img_shape) * A[channel]
            output[:, :, channel] = temp1 / tx + np.ones(self.img_shape) * A[channel]
        output = np.clip(output, 0, 255).astype(np.uint8)
        cv2.imshow('result', output)
        cv2.waitKey(0)
        return output

    def min_filter(self, img, K_size):
        height, width = img.shape
        pad = K_size // 2
        out_img = img.copy()
        pad_img = np.zeros((height + pad * 2, width + pad * 2))
        pad_img[pad: pad + height, pad: pad + width] = img.copy()

        for y in range(height):
            for x in range(width):
                out_img[y, x] = np.min(pad_img[y:y + K_size, x:x + K_size])
        return out_img


if __name__ == '__main__':
    img = cv2.imread('./db/IEI2019/H6.jpg')
    print(img.shape)
    model = HazelRemove(img=img)

    result = model.remove_hazel_DC()
    # cv2.imwrite(f'./result/H22_darkchannel_0.95w_9filter.jpg', result)
