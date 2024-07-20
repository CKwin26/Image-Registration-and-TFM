import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_points_and_arrows(image_path1, image_path2, excel_path, output_path1, output_path2):
    # 读取图像
    img1 = Image.open(image_path1).convert("L")
    img2 = Image.open(image_path2).convert("L")
    img_array1 = np.array(img1)
    img_array2 = np.array(img2)

    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 检查Excel文件是否包含所需列
    if 'x1' not in df.columns or 'y1' not in df.columns or 'x2' not in df.columns or 'y2' not in df.columns:
        raise ValueError("Excel文件中必须包含'x1', 'y1', 'x2'和'y2'列")

    # 绘制第一个图像上的关键点和箭头
    plt.figure(figsize=(img_array1.shape[1]/100, img_array1.shape[0]/100), dpi=100)
    plt.imshow(img_array1, cmap='gray')
    plt.scatter(df['x1'], df['y1'], c='green', edgecolor='white', s=5)  # 绿色点和箭头
    for i in range(len(df)):
        
        plt.arrow(df['x1'][i], df['y1'][i], df['x2'][i]-df['x1'][i], df['y2'][i]-df['y1'][i],
                  color='red', head_width=5, head_length=3, length_includes_head=True)  # 红色箭头

    # 保存第一个图像
    plt.axis('off')  # 隐藏坐标轴
    plt.savefig(output_path1, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"保存到 {output_path1}")

    # 绘制第二个图像上的关键点
    plt.figure(figsize=(img_array2.shape[1]/100, img_array2.shape[0]/100), dpi=100)
    plt.imshow(img_array2, cmap='gray')
    plt.scatter(df['x2'], df['y2'], c='green', edgecolor='white', s=10)  # 绿色点

    # 保存第二个图像
    plt.axis('off')  # 隐藏坐标轴
    plt.savefig(output_path2, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"保存到 {output_path2}")

def main(image_paths, excelp, picp1, picp2):
    image_path1 = image_paths[0]
    image_path2 = image_paths[1]
    plot_points_and_arrows(image_path1, image_path2, excelp, picp1, picp2)

if __name__ == "__main__":
    image_paths = [
        r'C:\\Users\\austa\\Downloads\\eighth_0.4\\Day 0 + 88bit.tif',
        r'C:\\Users\\austa\\Downloads\\eighth_0.4\\Day 4 + 88bit.tif'
    ]
    excelp = r'C:\\Users\\austa\\Downloads\nineth_ana\\matches_data.xlsx'
    picp1 = 'C:\\Users\\austa\\Downloads\\reference_image_with_manual_kp_and_arrows.png'
    picp2 = 'C:\\Users\\austa\\Downloads\\moving_image_with_manual_kp.png'
    main(image_paths, excelp, picp1, picp2)
