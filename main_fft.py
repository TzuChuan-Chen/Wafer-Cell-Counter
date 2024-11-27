import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 讀取圖像
img = cv.imread('HW_Image/20070907_171232_Wafer_100.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# 傅立葉轉換
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 創建純高通濾波器遮罩
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols, 2), np.uint8)
r = 30  # 濾波器半徑，可以調整
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
mask[mask_area] = 0

# 應用高通濾波器
fshift = dft_shift * mask

# 顯示濾波後的頻譜
magnitude_spectrum_filtered = 20*np.log(cv.magnitude(fshift[:,:,0], fshift[:,:,1]))

# 反傅立葉轉換
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])

# 增強對比度以便觀察
img_back_enhanced = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
cv.imwrite('results/20070907_171232_Wafer_100_enhanced.png', img_back_enhanced)
# 顯示結果
plt.figure(figsize=(12,10))
plt.subplot(221),plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(magnitude_spectrum_filtered, cmap='gray')
plt.title('High-Pass Frequency Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(img_back, cmap='gray')
plt.title('High Frequency Image'), plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(img_back_enhanced, cmap='gray')
plt.title('Enhanced High Frequency Image'), plt.xticks([]), plt.yticks([])

plt.show()