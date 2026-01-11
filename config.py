from dataclasses import dataclass
import torch
# 以类的方式定义参数
@dataclass
class Args:
    # training config
    # model config 
    image_size_train = 256
    image_size_test_single = 256
    image_size_test_multiple = 256
    num_secret = 1
    # optimer config
    lr = 2e-4
    warm_up_epoch = 20
    warm_up_lr_init = 5e-6
    # dataset paths
    DIV2K_path = '/root/autodl-tmp/dataset/DIV2K_train_HR'  # 原DIV2K路径
    cover_path = '/root/autodl-tmp/image/cover'  # 新增cover图像路径
    secret_path = '/root/autodl-tmp/image/secret'  # 新增secret图像路径
    single_batch_size = 12
    multi_batch_szie = 8
    multi_batch_iteration = (num_secret + 1) * 8
    test_multi_batch_size = num_secret + 1
    epochs = 6000  # 设置为总的epoch数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_freq = 10
    save_freq = 200
    train_next = 0  
    use_model = 'StegFormer-B'
    input_dim = 3
    norm_train = 'clamp'
    output_act = None
    path = '/root/autodl-tmp/StegFormer'
    model_name = 'StegFormer-B_4baseline'