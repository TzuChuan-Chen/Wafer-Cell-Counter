import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema


if __name__ == '__main__':

    img_title = ['20070907_160746_Cell_40.png', '20070907_160814_Cell_40.png', '20070907_161304_Cell_40.png',
                 '20070907_162752_Cell_40.png', '20070907_163631_Cell_28.png', '20070907_164013_Cell_79.png',
                 '20070907_170041_Cell_114.png', '20070907_171232_Wafer_100.png', '20070907_171317_Wafer_100.png']

    img = cv2.imread(f'HW_Image/{img_title[0]}', cv2.IMREAD_GRAYSCALE)

    #sobel y
    y_sobel = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    cv2.imwrite('sobel_y.png', y_sobel)



    # blur = cv2.medianBlur(img, 9)
    #img vs blur
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # plt.imshow(blur, cmap='gray')
    # plt.show()
    # copy_img = np.copy(blur)
    # y_sobel = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # y_sobel = cv2.convertScaleAbs(y_sobel)
    # dst = cv2.equalizeHist(y_sobel)

    # img_ref = dst[:, 300]

    # start_idx = np.where(img_ref > 220)[0][0]
    # end_idx = np.where(img_ref > 220)[0][-1]
    # # print(end_idx)
    # if np.mean(img_ref[:500]) > 100:
    #     start_idx = 520

    # if end_idx > 5000:  # 4160 上方物體長度
    #     end_idx -= 4180

    # # print(start_idx)
    # # print(end_idx)
    # roi_img = copy_img[start_idx: end_idx]

    # vt_1 = roi_img[:, 50]
    # vt_2 = roi_img[:, 200]
    # vt_3 = roi_img[:, 500]
    # vt_4 = roi_img[:, 800]
    # vt_5 = roi_img[:, 950]

    # cv2.imwrite('ROI_img.png', img[start_idx: end_idx])

    # # 找出峰值
    # peaks1, _ = find_peaks(vt_1, height=15, width=5, distance=5)
    # peaks2, _ = find_peaks(vt_2, height=15, width=5, distance=5)
    # peaks3, _ = find_peaks(vt_3, height=15, width=5, distance=5)
    # peaks4, _ = find_peaks(vt_4, height=15, width=5, distance=5)
    # peaks5, _ = find_peaks(vt_5, height=15, width=5, distance=5)
    # peak_list = [len(peaks1), len(peaks2), len(peaks3), len(peaks4), len(peaks5)]

    # # print(peak_list)
    # print('總共有', round(np.mean(peak_list[0])), '片')
    # # print('總共有', np.argmax(np.bincount(peak_list)), '片')

    # plt.imshow(dst, cmap='gray')
    # plt.show()
    # # plt.plot(peaks1, vt_1[peaks1], "x", color="red")
    # plt.vlines(peaks3-20, 0, 1020, color="red")
    # plt.plot(np.zeros_like(vt_3), "--", color="gray")
    # pimg = plt.imread('ROI_img.png')

    # plt.imshow(np.rot90(pimg), cmap='gray')
    # plt.plot(vt_3)
    # plt.show()
