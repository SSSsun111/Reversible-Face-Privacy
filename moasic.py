import os
import cv2
import numpy as np
from pathlib import Path
import time  # 👈 用于计时


def apply_mosaic(image, block_size=20):
    height, width = image.shape[:2]
    mosaic_img = image.copy()

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            h = min(block_size, height - y)
            w = min(block_size, width - x)
            block = image[y:y + h, x:x + w]

            avg_color = np.mean(block, axis=(0, 1)) if len(image.shape) == 3 else np.mean(block)
            mosaic_img[y:y + h, x:x + w] = avg_color

    return mosaic_img


def process_images(input_dir, output_dir, block_size=20):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    input_path = Path(input_dir)
    files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    latencies = []  # 👈 存放每张图片的处理时间

    for file in files:
        try:
            image = cv2.imread(str(file))
            if image is None:
                print(f"无法读取图像: {file}")
                continue

            # ----------------------------
            # ✅ 开始计时：只测马赛克算法本身
            # ----------------------------
            start = time.time()

            mosaic_image = apply_mosaic(image, block_size)

            end = time.time()
            latency_ms = (end - start) * 1000  # 转为毫秒
            latencies.append(latency_ms)

            # ----------------------------
            # 保存图像（不计入延迟）
            # ----------------------------
            output_file = Path(output_dir) / file.name
            cv2.imwrite(str(output_file), mosaic_image)

            print(f"已处理: {file.name}, 延迟: {latency_ms:.2f} ms")

        except Exception as e:
            print(f"处理 {file} 时出错: {e}")

    # ----------------------------
    # 最后输出平均延迟
    # ----------------------------
    if len(latencies) > 0:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n平均延迟: {avg_latency:.2f} ms/张（仅算法执行时间）")


if __name__ == "__main__":
    input_directory = r"D:\Desktop\test"
    output_directory = r"D:\Desktop\REALDATA3"
    mosaic_block_size = 30

    print("开始处理图像...")
    process_images(input_directory, output_directory, mosaic_block_size)
    print("图像处理完成！")
