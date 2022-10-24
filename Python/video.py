import cv2
import glob
import numpy as np
import cv2
import copy

class HazelRemove():
    def __init__(self, img):
        self.img = img
        self.darkImg = np.min(img, 2)
        # self.darkImg = cv2.erode(np.min(img, 2), (15, 15))
        self.num_pixels = len(img) * len(img[0])
        self.B_A, self.G_A, self.R_A = self.get_A_val()
        self.img_shape = img[:, :, 0].shape

    def display_dark_channel(self):
        cv2.imshow('dark img', self.darkImg)
        cv2.waitKey(0)

    def get_dark_channel(self, img):
        darkImg = np.min(img, 2)
        guided_darkImg = guidedfilter(darkImg, cv2.erode(darkImg, (15, 15)), 8, 500)
        return guided_darkImg

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
        # print('A value for channel B, G, A', max_element[0], max_element[1], max_element[2])
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
        # cv2.imshow('result', output)
        # cv2.waitKey(0)
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

def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
    imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
    imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0 : r+1] = imCum[:, r : 2*r+1]
    imDst[:, r+1 : cols-r] = imCum[:, 2*r+1 : cols] - imCum[:, 0 : cols-2*r-1]
    imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]

    return imDst

def guidedfilter(I, p, r, eps):
    (rows, cols) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)

    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N

    q = meanA * I + meanB
    return q


def readtest():
    videoname = 'db/IEV2022/airplane.mp4'
    capture = cv2.VideoCapture(videoname)
    if capture.isOpened():
        while True:
            ret, img = capture.read()  # img 就是一帧图片
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            if not ret: break  # 当获取完最后一帧就结束
    else:
        print('视频打开失败！')


def writetest():
    videoname = 'db/IEV2022/airplane.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(videoname, fourcc, 1.0, (1280, 960), True)
    imgpaths = glob.glob('*.jpg')
    for path in imgpaths:
        print(path)
        img = cv2.imread(path)
        writer.write(img)  # 读取图片后一帧帧写入到视频中
    writer.release()


def makevideo():
    videoinpath = 'db/IEV2022/airplane.mp4'
    videooutpath = 'result.avi'
    capture = cv2.VideoCapture(videoinpath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(videooutpath, fourcc, 23.0, (960, 432), True) # 指定编码器, 帧率， 每一帧大小
    if capture.isOpened():
        while True:
            ret, img_src = capture.read()
            if not ret: break
            model = HazelRemove(img=img_src)
            img_out = model.remove_hazel_DC()  # 自己写函数op_one_img()逐帧处理
            writer.write(img_out)
    else:
        print('视频打开失败！')
    writer.release()


if __name__ == '__main__':
    makevideo()
