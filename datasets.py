import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import config
import os
from collections import defaultdict

args = config.Args()

# 对数据集图像进行处理
transform = T.Compose([
    T.RandomCrop(128),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

# 使用 albumentations 库对图像进行处理
transform_A = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.augmentations.transforms.ChannelShuffle(0.3),
    ToTensorV2()
])

transform_A_valid = A.Compose([
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])

transform_A_test = A.Compose([
    A.LongestMaxSize(max_size=1024),  # 将最长边缩放到1024，保持比例
    A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT),  # 填充至目标尺寸
    ToTensorV2()
])

# 原始 DIV2K 数据集路径
DIV2K_path = "/root/autodl-tmp/dataset"

# 新的 cover 和 secret 数据集路径
cover_path = "/root/autodl-tmp/image/cover"
secret_path = "/root/autodl-tmp/image/secret"

batchsize = 12


# 原始 DIV2K 数据集类
class DIV2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path + "/DIV2K_train_HR" + "/*." + "png")))
        else:
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path + "/DIV2K_valid_HR" + "/*." + "png")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item / 255.0
        return item

    def __len__(self):
        return len(self.files)


# 新的 Cover 数据集类
class Cover_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        # 直接从 cover_path 加载所有图像
        self.files = natsorted(
            sorted(glob.glob(cover_path + "/*." + "png") +
                   glob.glob(cover_path + "/*." + "jpg") +
                   glob.glob(cover_path + "/*." + "jpeg")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index % len(self.files)])  # 确保索引有效
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item / 255.0
        return item

    def __len__(self):
        return len(self.files)


# 新的 Secret 数据集类
class Secret_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        # 直接从 secret_path 加载所有图像
        self.files = natsorted(
            sorted(glob.glob(secret_path + "/*." + "png") +
                   glob.glob(secret_path + "/*." + "jpg") +
                   glob.glob(secret_path + "/*." + "jpeg")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index % len(self.files)])  # 确保索引有效
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item / 255.0
        return item

    def __len__(self):
        return len(self.files)


# 新增：按文件名匹配的配对数据集类
class Paired_Dataset(Dataset):
    def __init__(self, cover_path, secret_path, transforms_=None, mode='test'):
        self.cover_path = cover_path
        self.secret_path = secret_path
        self.transform = transforms_
        self.mode = mode

        # 获取所有封面图像路径
        cover_files = (glob.glob(cover_path + "/*." + "png") +
                       glob.glob(cover_path + "/*." + "jpg") +
                       glob.glob(cover_path + "/*." + "jpeg"))

        # 获取所有秘密图像路径
        secret_files = (glob.glob(secret_path + "/*." + "png") +
                        glob.glob(secret_path + "/*." + "jpg") +
                        glob.glob(secret_path + "/*." + "jpeg"))

        # 创建文件名到路径的映射
        cover_name_to_path = {os.path.splitext(os.path.basename(f))[0]: f for f in cover_files}
        secret_name_to_path = {os.path.splitext(os.path.basename(f))[0]: f for f in secret_files}

        # 找出共同的文件名
        common_names = set(cover_name_to_path.keys()) & set(secret_name_to_path.keys())

        # 创建配对列表
        self.pairs = [(cover_name_to_path[name], secret_name_to_path[name]) for name in natsorted(common_names)]

        # 如果没有共同文件名，则使用所有可能的组合
        if not self.pairs:
            print("警告：没有找到名称匹配的图像对。使用所有可能的组合。")
            self.cover_files = natsorted(sorted(cover_files))
            self.secret_files = natsorted(sorted(secret_files))
            # 如果两个列表长度不同，使用较短的列表长度
            min_len = min(len(self.cover_files), len(self.secret_files))
            self.pairs = [(self.cover_files[i % len(self.cover_files)],
                           self.secret_files[i % len(self.secret_files)])
                          for i in range(min_len)]

    def __getitem__(self, index):
        cover_path, secret_path = self.pairs[index]

        # 读取封面图像
        cover_img = cv2.imread(cover_path)
        cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
        cover_trans = self.transform(image=cover_img)
        cover_item = cover_trans['image']
        cover_item = cover_item / 255.0

        # 读取秘密图像
        secret_img = cv2.imread(secret_path)
        secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2RGB)
        secret_trans = self.transform(image=secret_img)
        secret_item = secret_trans['image']
        secret_item = secret_item / 255.0

        return cover_item, secret_item, os.path.basename(cover_path)  # 同时返回文件名用于调试

    def __len__(self):
        return len(self.pairs)


# 修改原有的所有dataloader定义...（保持不变）

# 修改测试集loader，使用新的配对数据集
paired_test_loader = DataLoader(
    Paired_Dataset(cover_path, secret_path, transforms_=transform_A_test, mode="test"),
    batch_size=1,
    shuffle=False,  # 测试时不打乱顺序
    pin_memory=True,
    num_workers=1
)

# 原有的其他dataloader保持不变
DIV2K_train_cover_loader = DataLoader(
    Cover_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_train_secret_loader = DataLoader(
    Secret_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

# 验证集 loader
DIV2K_val_cover_loader = DataLoader(
    Cover_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

DIV2K_val_secret_loader = DataLoader(
    Secret_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.single_batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

# 保留原有的测试loader以便向后兼容
DIV2K_test_cover_loader = DataLoader(
    Cover_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

DIV2K_test_secret_loader = DataLoader(
    Secret_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

# 多批次 loader - 保留原有的 DIV2K 数据集用于兼容性
DIV2K_multi_train_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.multi_batch_iteration,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)

DIV2K_multi_val_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.multi_batch_iteration,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)

DIV2K_multi_test_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=args.test_multi_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)