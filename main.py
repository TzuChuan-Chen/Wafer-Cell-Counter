import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import rotate

class CellCounter:
    def __init__(self):
        self.debug = True
        
    def preprocess_image(self, img):
        # 自適應中值濾波
        blur = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        
        # Sobel邊緣檢測
        y_sobel = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        y_sobel = np.absolute(y_sobel)
        y_sobel = np.uint8(y_sobel)
        
        # 自適應直方圖均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(y_sobel)
        
        return enhanced
        
    def detect_roi(self, img):
        # 投影到Y軸
        y_proj = np.sum(img, axis=1)
        
        # 使用OTSU自動找閾值
        thresh = threshold_otsu(y_proj)
        
        # 找到有意義的區域
        valid_rows = np.where(y_proj > thresh)[0]
        start_idx = valid_rows[0]
        end_idx = valid_rows[-1]
        
        # 加入安全邊界
        margin = 50
        start_idx = max(0, start_idx - margin)
        end_idx = min(img.shape[0], end_idx + margin)
        
        return start_idx, end_idx
        
    def detect_rotation(self, img):
        # 使用霍夫變換檢測主要線條
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                if angle < 45:  # 假設cell大致垂直
                    angles.append(angle)
            
            if angles:
                return np.median(angles)
        return 0
        
    def count_cells(self, img):
        # 預處理
        processed = self.preprocess_image(img)
        
        # 檢測ROI
        start_idx, end_idx = self.detect_roi(processed)
        roi = processed[start_idx:end_idx, :]
        
        # 檢測和修正旋轉
        angle = self.detect_rotation(roi)
        if abs(angle) > 0.5:
            roi = rotate(roi, angle)
            
        # 在多個位置取樣
        num_samples = 7
        sample_positions = np.linspace(0.1, 0.9, num_samples) * roi.shape[1]
        sample_positions = sample_positions.astype(int)
        
        # 動態確定峰值檢測參數
        mean_intensity = np.mean(roi)
        std_intensity = np.std(roi)
        
        peak_counts = []
        for pos in sample_positions:
            profile = roi[:, pos]
            
            # 動態設定參數
            height = mean_intensity + 0.5 * std_intensity
            distance = int(roi.shape[0] / 200)  # 基於預期的cell數量
            
            peaks, _ = find_peaks(profile, 
                                height=height,
                                distance=distance,
                                prominence=std_intensity)
            peak_counts.append(len(peaks))
            
            if self.debug:
                plt.figure(figsize=(10, 5))
                plt.plot(profile)
                plt.plot(peaks, profile[peaks], "x")
                plt.title(f"Sample at position {pos}")
                plt.show()
        
        # 使用中位數作為最終結果
        cell_count = int(np.median(peak_counts))
        
        return cell_count, roi

def main():
    counter = CellCounter()
    
    img_title = ['20070907_160746_Cell_40.png', '20070907_160814_Cell_40.png']
    
    for title in img_title:
        img = cv2.imread(f'HW_Image/{title}', cv2.IMREAD_GRAYSCALE)
        count, roi = counter.count_cells(img)
        
        print(f"Image: {title}")
        print(f"Estimated cell count: {count}")
        
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        
        plt.subplot(122)
        plt.imshow(roi, cmap='gray')
        plt.title(f"ROI (Detected {count} cells)")
        plt.show()

if __name__ == '__main__':
    main()