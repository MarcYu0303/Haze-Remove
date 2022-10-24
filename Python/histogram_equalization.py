import cv2
import matplotlib.pyplot as plt

# def Histogram_Eqaulization(img):
#     blue = img[:, :, 0]
#     green = img[:, :, 1]
#     red = img[:, :, 2]
#     blue_equ = cv2.equalizeHist(blue)
#     green_equ = cv2.equalizeHist(green)
#     red_equ = cv2.equalizeHist(red)
#     equ = cv2.merge([blue_equ, green_equ, red_equ])
#     return equ

if __name__ == '__main__':
    img = cv2.imread('./db/IEI2019/H26.jpg')

    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    blue_equ = cv2.equalizeHist(blue)
    green_equ = cv2.equalizeHist(green)
    red_equ = cv2.equalizeHist(red)
    equ = cv2.merge([blue_equ, green_equ, red_equ])

    cv2.imshow("1", img)
    cv2.imshow("2", equ)
    plt.figure("Oringinal")
    plt.hist(img.ravel(), 256)
    plt.figure("Result")
    plt.hist(equ.ravel(), 256)
    plt.show()
    cv2.imwrite('./result/1024/H26_HE.jpg', equ)
    cv2.waitKey(0)
    cv2.destroyallwindows()