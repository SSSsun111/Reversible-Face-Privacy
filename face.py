import face_recognition
import numpy as np
import os
from PIL import Image


def compare_faces(image_path1, image_path2, tolerance=0.6):
    """
    比较两张人脸图片是否为同一人

    参数:
        image_path1: 第一张人脸图片路径
        image_path2: 第二张人脸图片路径
        tolerance: 人脸相似度阈值，越小越严格，默认为0.6

    返回:
        is_different: 布尔值，True表示不是同一人，False表示可能是同一人
        distance: 两张人脸的欧氏距离
        message: 处理结果信息
    """
    try:
        # 加载图片
        image1 = face_recognition.load_image_file(image_path1)
        image2 = face_recognition.load_image_file(image_path2)

        # 检测人脸
        face_locations1 = face_recognition.face_locations(image1)
        face_locations2 = face_recognition.face_locations(image2)

        # 检查是否检测到人脸
        if len(face_locations1) == 0:
            return False, None, "第一张图片未检测到人脸"
        if len(face_locations2) == 0:
            return False, None, "第二张图片未检测到人脸"

        # 提取人脸特征编码
        face_encodings1 = face_recognition.face_encodings(image1, face_locations1)
        face_encodings2 = face_recognition.face_encodings(image2, face_locations2)

        # 检查是否成功提取到人脸编码
        if len(face_encodings1) == 0:
            return False, None, "第一张图片无法提取人脸特征"
        if len(face_encodings2) == 0:
            return False, None, "第二张图片无法提取人脸特征"

        face_encoding1 = face_encodings1[0]
        face_encoding2 = face_encodings2[0]

        # 计算人脸特征向量之间的距离
        face_distance = face_recognition.face_distance([face_encoding1], face_encoding2)[0]

        # 根据距离判断是否为同一人
        is_same_person = face_distance <= tolerance

        return not is_same_person, face_distance, "成功比对"

    except Exception as e:
        return False, None, f"错误: {str(e)}"


def compare_same_filename_images(directory_path1, directory_path2, tolerance=0.5):
    """
    比较两个目录中相同文件名的图片，判断是否为同一个人

    参数:
        directory_path1: 第一个图片目录路径
        directory_path2: 第二个图片目录路径
        tolerance: 人脸相似度阈值

    返回:
        results: 比较结果列表
    """
    # 获取两个目录下所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # 获取第一个目录中的图片文件
    files_dir1 = {}
    for file in os.listdir(directory_path1):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            files_dir1[file] = os.path.join(directory_path1, file)

    # 获取第二个目录中的图片文件
    files_dir2 = {}
    for file in os.listdir(directory_path2):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            files_dir2[file] = os.path.join(directory_path2, file)

    # 找出两个目录中相同的文件名
    common_files = set(files_dir1.keys()) & set(files_dir2.keys())

    print(f"在两个目录中找到 {len(common_files)} 个相同文件名的图片")

    # 比较结果列表
    results = []

    # 比较相同文件名的图片
    for file in common_files:
        img_path1 = files_dir1[file]
        img_path2 = files_dir2[file]

        is_different, distance, message = compare_faces(img_path1, img_path2, tolerance)

        result = {
            'filename': file,
            'image1': img_path1,
            'image2': img_path2,
            'is_different': is_different,
            'distance': distance,
            'message': message
        }

        results.append(result)

        # 输出比较结果 - 修复了None值的格式化问题
        if message.startswith("错误") or distance is None:
            print(f"{file}: {message}")
        else:
            if is_different:
                print(f"{file}: 不是同一个人（距离: {distance:.4f}）")
            else:
                print(f"{file}: 可能是同一个人（距离: {distance:.4f}）")

    return results


def save_results_to_file(results, output_file):
    """
    将比较结果保存到文件

    参数:
        results: 比较结果列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("文件名,是否为不同人,距离,消息\n")

        for result in results:
            is_different_str = "是" if result['is_different'] else "否"
            distance_str = f"{result['distance']:.4f}" if result['distance'] is not None else "N/A"

            f.write(f"{result['filename']},{is_different_str},{distance_str},{result['message']}\n")

    print(f"结果已保存到: {output_file}")


# 主程序
if __name__ == "__main__":
    # 指定两个图片目录路径
    directory_path1 = "/root/autodl-tmp/pulse/yuantu"  # 修改为你的第一个目录
    directory_path2 = "/root/autodl-tmp/pulse/runs"  # 修改为你的第二个目录

    # 设置结果输出文件
    output_file = "/root/autodl-tmp/pulse/face_comparison_results.csv"

    # 确保目录存在
    if not os.path.exists(directory_path1):
        print(f"错误: 目录不存在 - {directory_path1}")
    elif not os.path.exists(directory_path2):
        print(f"错误: 目录不存在 - {directory_path2}")
    else:
        print(f"开始比较 {directory_path1} 和 {directory_path2} 中相同文件名的图片...")
        results = compare_same_filename_images(directory_path1, directory_path2)

        # 统计结果
        total = len(results)
        different_count = sum(1 for r in results if r['is_different'])
        same_count = total - different_count
        error_count = sum(1 for r in results if r['message'].startswith("错误") or r['distance'] is None)

        print("\n比较结果汇总:")
        print(f"总共比较: {total} 对图片")
        print(f"不是同一个人: {different_count} 对")
        print(f"可能是同一个人: {same_count} 对")
        print(f"出错或无法比较: {error_count} 对")

        # 保存结果到文件
        save_results_to_file(results, output_file)