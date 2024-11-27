import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def process_image(img_path):
    # 讀取圖像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 預處理
    blur = cv2.medianBlur(img, 9)
    copy_img = np.copy(blur)
    
    # Sobel邊緣檢測和直方圖均衡化
    y_sobel = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    y_sobel = cv2.convertScaleAbs(y_sobel)
    dst = cv2.equalizeHist(y_sobel)
    
    # 使用參考列找到ROI區域
    img_ref = dst[:, 300]
    start_idx = np.where(img_ref > 220)[0][0]
    end_idx = np.where(img_ref > 220)[0][-1]
    
    # 調整ROI邊界
    if np.mean(img_ref[:500]) > 100:
        start_idx = 520
    if end_idx > 5000:
        end_idx -= 4180
        
    # 裁切ROI區域
    roi_img = copy_img[start_idx: end_idx]
    
    # 儲存ROI圖像
    cv2.imwrite('results/ROI_img.png', img[start_idx: end_idx])
    
    return roi_img, dst, start_idx, end_idx

def analyze_vertical_lines(roi_img):
    # 選擇多個垂直切片位置
    positions = [50, 200, 500, 800, 950]
    peak_counts = []
    all_peaks = []
    
    # 分析每個垂直切片
    for pos in positions:
        vertical_line = roi_img[:, pos]
        peaks, _ = find_peaks(vertical_line, 
                            height=15,    # 最小高度
                            width=5,      # 最小寬度
                            distance=5)   # 最小間距
        peak_counts.append(len(peaks))
        all_peaks.append(peaks)
    
    # 計算平均片數
    avg_count = round(np.mean(peak_counts))
    
    return avg_count, peak_counts, all_peaks, positions

def visualize_results(roi_img, dst, peaks, positions, avg_count, base_name):
    # 創建更大的圖表
    plt.figure(figsize=(20, 15))
    
    # 調整子圖的相對大小
    grid = plt.GridSpec(3, 1, height_ratios=[1, 2, 1.5])
    
    # 顯示處理後的完整圖像
    plt.subplot(grid[0])
    plt.imshow(dst, cmap='gray')
    plt.title('Processed Image', fontsize=14)
    
    # 顯示ROI區域和檢測結果
    plt.subplot(grid[1])
    plt.imshow(roi_img, cmap='gray')
    # 只使用中間的切片（position[2] = 500）
    center_peaks = peaks[2]  # 使用中間的切片結果
    for peak in center_peaks:
        plt.axhline(y=peak, color='r', alpha=0.5)
    plt.title(f'ROI Region with Detected Lines (Count: {len(center_peaks)})', fontsize=14)
    
    # 顯示中間切片的投影曲線
    plt.subplot(grid[2])
    vertical_line = roi_img[:, positions[2]]  # 使用中間切片
    # 正規化投影曲線到0-1範圍
    normalized_line = (vertical_line - np.min(vertical_line)) / (np.max(vertical_line) - np.min(vertical_line))
    plt.plot(normalized_line, color='b', linewidth=2, label=f'Position {positions[2]}')
    plt.plot(center_peaks, normalized_line[center_peaks], "rx", markersize=10)
    plt.title('Normalized Vertical Projection with Peaks', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # 調整圖片之間的間距
    plt.tight_layout(pad=3.0)
    
    # 儲存高解析度的圖片
    plt.savefig(f'results/{base_name}_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # 確保結果資料夾存在
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    img_title = ['20070907_171232_Wafer_100_enhanced.png']

    for title in img_title:
        try:
            print(f"\n處理圖片: {title}")
            image_path = f'HW_Image/{title}'
            
            # 處理圖像
            roi_img, dst, start_idx, end_idx = process_image(image_path)
            
            # 分析垂直線
            avg_count, peak_counts, peaks, positions = analyze_vertical_lines(roi_img)
            
            # 視覺化結果
            base_name = title.split('.')[0]
            visualize_results(roi_img, dst, peaks, positions, avg_count, base_name)
            
            # 儲存統計資訊
            count_from_filename = int(title.split('_')[-1].split('.')[0])
            print(f"檔案名稱中的數量: {count_from_filename}")
            print(f"偵測到的平均數量: {avg_count}")
            print(f"各切片偵測數量: {peak_counts}")
            
        except Exception as e:
            print(f"處理圖片 {title} 時發生錯誤: {str(e)}")