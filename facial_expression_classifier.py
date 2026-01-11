import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 表情类别定义
EMOTION_LABELS = {
    0: 'neutral',  # 中立
    1: 'happy',  # 高兴
    2: 'sad',  # 悲伤
    3: 'surprise',  # 惊讶
    4: 'anger',  # 生气
    5: 'fear',  # 恐惧
    6: 'disgust'  # 厌恶
}


class FacialExpressionDataset(Dataset):
    """面部表情数据集"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历所有表情类别文件夹
        for label, emotion in EMOTION_LABELS.items():
            emotion_dir = self.root_dir / emotion
            if emotion_dir.exists():
                for img_path in emotion_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(label)
                for img_path in emotion_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(label)

        print(f"加载了 {len(self.images)} 张图片")
        print(f"类别分布: {dict(zip(*np.unique(self.labels, return_counts=True)))}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ExpressionClassifier(nn.Module):
    """基于ResNet的表情分类器"""

    def __init__(self, num_classes=7, pretrained=True):
        super(ExpressionClassifier, self).__init__()
        # 使用预训练的ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        # 修改最后的全连接层
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=20, device='cuda'):
    """训练模型"""
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 60)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())

        print(f'验证 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_expression_model.pth')
            print(f'保存最佳模型，准确率: {best_acc:.4f}')

    return history


def evaluate_model(model, test_loader, device='cuda'):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds,
                                target_names=list(EMOTION_LABELS.values())))

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(EMOTION_LABELS.values()),
                yticklabels=list(EMOTION_LABELS.values()))
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # 计算准确率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\n总体准确率: {accuracy:.4f}")

    return accuracy, all_preds, all_labels


def plot_training_history(history):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss曲线
    ax1.plot(history['train_loss'], label='训练Loss')
    ax1.plot(history['val_loss'], label='验证Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy曲线
    ax2.plot(history['train_acc'], label='训练Acc')
    ax2.plot(history['val_acc'], label='验证Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def main():
    # 配置
    path_fake = "/root/autodl-tmp/pulse/runs"
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"使用设备: {device}")

    # 数据增强和标准化
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    print("加载数据集...")
    train_dataset = FacialExpressionDataset(
        os.path.join(path_fake, 'train'),
        transform=train_transform
    )
    val_dataset = FacialExpressionDataset(
        os.path.join(path_fake, 'val'),
        transform=val_transform
    )
    test_dataset = FacialExpressionDataset(
        os.path.join(path_fake, 'test'),
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    # 创建模型
    print("创建模型...")
    model = ExpressionClassifier(num_classes=len(EMOTION_LABELS), pretrained=True)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    print("\n开始训练...")
    history = train_model(model, train_loader, val_loader, criterion,
                          optimizer, num_epochs=num_epochs, device=device)

    # 绘制训练曲线
    plot_training_history(history)

    # 加载最佳模型并测试
    print("\n加载最佳模型进行测试...")
    model.load_state_dict(torch.load('best_expression_model.pth'))
    accuracy, preds, labels = evaluate_model(model, test_loader, device=device)

    print(f"\n实验结论:")
    print(f"测试集准确率: {accuracy:.4f}")
    if accuracy > 0.6:
        print("✓ 匿名化后的人脸图像仍保留较强的表情语义信息")
    elif accuracy > 0.4:
        print("⚠ 匿名化后的人脸图像部分保留表情语义信息")
    else:
        print("✗ 匿名化严重影响了表情语义信息")


if __name__ == '__main__':
    main()