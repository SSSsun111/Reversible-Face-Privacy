import os
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Dict

from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ========= 全局初始化 ArcFace（InsightFace） =========
# model: 选择官方集合 'buffalo_l'（包含检测、关键点与ArcFace特征）
# ctx_id: -1 用 CPU；0 用第一块 GPU
APP = FaceAnalysis(name="buffalo_l")
APP.prepare(ctx_id=0, det_size=(640, 640))  # 如无GPU改成 ctx_id=-1

def _load_image_bgr(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return img

def _get_arcface_embedding(img_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    返回 (embedding_512维, message)。若失败，embedding 返回 None，message 说明原因。
    """
    faces = APP.get(img_bgr)
    if len(faces) == 0:
        return None, "未检测到人脸"

    # 选最大的人脸（与 face_recognition 默认行为接近）
    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    face = faces[0]

    # InsightFace已经给出对齐特征 face.embedding（通常已L2归一化）
    emb = face.normed_embedding if hasattr(face, "normed_embedding") else face.embedding

    if emb is None or emb.size == 0:
        return None, "无法提取人脸特征"

    # 确保是 float32 且 L2 归一化（保险起见再归一化一次）
    emb = emb.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return None, "特征为零向量"
    emb = emb / norm
    return emb, "成功"

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 定义为 1 - cos_sim，值越小越像
    return 1.0 - _cosine_similarity(a, b)

def compare_faces(image_path1: str, image_path2: str, tolerance: float = 0.4):
    """
    用 ArcFace(InsightFace) 比较两张人脸是否同一人。
    这里的 tolerance 是 **余弦距离阈值**（1 - cos_sim），默认 0.4（越小越严格）。
    常见经验：0.3~0.5 之间可调，取决于你数据的清晰度与场景。
    返回:
        is_different: True 代表“不是同一人”；False 代表“可能是同一人”
        distance: 余弦距离（越小越像）
        message: 文本信息
    """
    try:
        img1 = _load_image_bgr(image_path1)
        img2 = _load_image_bgr(image_path2)

        emb1, msg1 = _get_arcface_embedding(img1)
        if emb1 is None:
            return False, None, f"第一张图片：{msg1}"

        emb2, msg2 = _get_arcface_embedding(img2)
        if emb2 is None:
            return False, None, f"第二张图片：{msg2}"

        dist = _cosine_distance(emb1, emb2)
        is_same = dist <= tolerance
        return (not is_same), dist, "成功比对（ArcFace/余弦距离）"

    except Exception as e:
        return False, None, f"错误: {str(e)}"

def compare_same_filename_images(directory_path1: str, directory_path2: str, tolerance: float = 0.4):
    """
    比较两个目录中相同文件名的图片（ArcFace 版）
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    files_dir1 = {f: os.path.join(directory_path1, f)
                  for f in os.listdir(directory_path1)
                  if any(f.lower().endswith(ext) for ext in image_extensions)}
    files_dir2 = {f: os.path.join(directory_path2, f)
                  for f in os.listdir(directory_path2)
                  if any(f.lower().endswith(ext) for ext in image_extensions)}

    common_files = set(files_dir1.keys()) & set(files_dir2.keys())
    print(f"在两个目录中找到 {len(common_files)} 个相同文件名的图片")

    results = []
    for file in sorted(common_files):
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

        if message.startswith("错误") or distance is None:
            print(f"{file}: {message}")
        else:
            if is_different:
                print(f"{file}: 不是同一个人（余弦距离: {distance:.4f}）")
            else:
                print(f"{file}: 可能是同一个人（余弦距离: {distance:.4f}）")

    return results

def save_results_to_file(results: List[Dict], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("文件名,是否为不同人,余弦距离(1-cos),消息\n")
        for r in results:
            is_different_str = "是" if r['is_different'] else "否"
            distance_str = f"{r['distance']:.4f}" if r['distance'] is not None else "N/A"
            f.write(f"{r['filename']},{is_different_str},{distance_str},{r['message']}\n")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    directory_path1 = "/root/autodl-tmp/pulse/yuantu"
    directory_path2 = "/root/autodl-tmp/pulse/runs"
    output_file = "/root/autodl-tmp/pulse/face_comparison_results_arcface.csv"

    if not os.path.exists(directory_path1):
        print(f"错误: 目录不存在 - {directory_path1}")
    elif not os.path.exists(directory_path2):
        print(f"错误: 目录不存在 - {directory_path2}")
    else:
        print(f"开始比较 {directory_path1} 和 {directory_path2} 中相同文件名的图片（ArcFace）...")
        results = compare_same_filename_images(directory_path1, directory_path2, tolerance=0.4)

        total = len(results)
        different_count = sum(1 for r in results if r['distance'] is not None and r['is_different'])
        same_count = sum(1 for r in results if r['distance'] is not None and not r['is_different'])
        error_count = sum(1 for r in results if r['message'].startswith("错误") or r['distance'] is None)

        print("\n比较结果汇总:")
        print(f"总共比较: {total} 对图片")
        print(f"不是同一个人: {different_count} 对")
        print(f"可能是同一个人: {same_count} 对")
        print(f"出错或无法比较: {error_count} 对")

        save_results_to_file(results, output_file)
