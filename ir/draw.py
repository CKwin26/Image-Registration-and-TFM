import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_points_and_arrows(image_path1, image_path2, excel_path, output_path1, output_path2):
    # read img
    img1 = Image.open(image_path1).convert("L")
    img2 = Image.open(image_path2).convert("L")
    img_array1 = np.array(img1)
    img_array2 = np.array(img2)

    # get excel
    df = pd.read_excel(excel_path)

    # check name
    if 'x1' not in df.columns or 'y1' not in df.columns or 'x2' not in df.columns or 'y2' not in df.columns:
        raise ValueError("Excel文件中必须包含'x1', 'y1', 'x2'和'y2'列")

    # draw first
    plt.figure(figsize=(img_array1.shape[1]/100, img_array1.shape[0]/100), dpi=100)
    plt.imshow(img_array1, cmap='gray')
    plt.scatter(df['x1'], df['y1'], c='green', edgecolor='white', s=5)  # green dots and arrow behind
    for i in range(len(df)):
        plt.arrow(df['x1'][i], df['y1'][i], df['x2'][i]-df['x1'][i], df['y2'][i]-df['y1'][i],
                  color='red', head_width=5, head_length=3, length_includes_head=True)  # red arrow

    # save
    plt.axis('off')  # hid axis
    plt.savefig(output_path1, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"save to  {output_path1}")

    # save second
    plt.figure(figsize=(img_array2.shape[1]/100, img_array2.shape[0]/100), dpi=100)
    plt.imshow(img_array2, cmap='gray')
    plt.scatter(df['x2'], df['y2'], c='green', edgecolor='white', s=10)  # green dots

    # hid 2nd
    plt.axis('off')  # hide axis
    plt.savefig(output_path2, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"save to {output_path2}")

if __name__ == "__main__":
    image_path2 = 'C:\\Users\\austa\\Downloads\\nineth_ana\\ref.tif'
    image_path1 = 'C:\\Users\\austa\\Downloads\\nineth_ana\\al.tif'
    excel_path = 'C:\\Users\\austa\\Downloads\\nineth\\matched_kpwithcontrast.xlsx'
    output_path1 = 'C:\\Users\\austa\\Downloads\\nineth_ana\\manual1.png'
    output_path2 = 'C:\\Users\\austa\\Downloads\\nineth_ana\\manual2.png'
    plot_points_and_arrows(image_path1, image_path2, excel_path, output_path1, output_path2)
