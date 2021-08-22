# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm

# Log
import logging


logger = logging.getLogger()  # 不加名称设置root logger
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# 使用FileHandler输出到文件
fh = logging.FileHandler('HW03_byCRZ_log.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

# 使用StreamHandler输出到屏幕
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# 添加两个Handler
logger.addHandler(ch)
logger.addHandler(fh)
# logger.info('this is info message')
# logger.warning('this is warn message')


CONFIG = {
    "batch_size": 128,
    "train_set_path": "food-11/training/labeled",
    "valid_set_path": "food-11/validation",
    "unlabeled_set_path": "food-11/training/unlabeled",
    "test_set_path": "food-11/testing",

    "num_workers": 4,
    "num_epochs": 80,
    "do_semi": False,

    "device": "gpu",
    "save_path": "./checkpoints",
}


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64), # ?
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # FM大小: (64, 64)

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # FM大小: (32, 32)

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0), # FM大小: (8, 8)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256), # 目的？猜想：增强表达能力
            nn.ReLU(),
            nn.Linear(256, 11),
        )
    
    def forward(self, x):
        # x: [batch_size, 3, 128, 128]
        # logger.info(f"before cnn: {x.shape}")
        x = self.cnn_layers(x)
        # x: [batch_size, 256, 8, 8]
        # logger.info(f"before flatten: {x.shape}")
        x = torch.flatten(x, start_dim=1)
        # x: [batch_size, 256 * 8 * 8]
        # logger.info(f"before fc: {x.shape}")
        x = self.fc_layers(x)
        # x: [batch_size, 11]
        return x


def pre_datasets():
    TRAIN_TFM = transforms.Compose(
        [
            transforms.Resize(size=(128, 128)),
            # TODO
            transforms.ToTensor(),
        ]
    )
    TEST_TFM = transforms.Compose(
        [
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ]
    )
    train_set = DatasetFolder(
        root=CONFIG["train_set_path"],
        loader=Image.open,
        extensions="jpg",
        transform=TRAIN_TFM,
    )
    valid_set = DatasetFolder(
        root=CONFIG["valid_set_path"],
        loader=Image.open,
        extensions="jpg",
        transform=TEST_TFM,
    )
    unlabeled_set = DatasetFolder(
        root=CONFIG["unlabeled_set_path"],
        loader=Image.open,
        extensions="jpg",
        transform=TRAIN_TFM,
    )
    test_set = DatasetFolder(
        root=CONFIG["test_set_path"],
        loader=Image.open,
        extensions="jpg",
        transform=TEST_TFM,
    )
    # 1. 为什么遍历train_set的每一项，输出结果是一个tuple，不应该是Tensor吗？
    # 把transform参数隐去，输出的是PILImage类型。
    # 确实是Tensor。tuple是图片和标签合在一起了。
    # for item in train_set:
    #     print(type(item[0]), item)
    #     break
    # print(type(train_set))
    # 2. 标签跑到哪里去了呢？
    # 标签和图片一起构成了tuple，类别似乎是文件名第一个下划线之前的内容。
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    # print("pre id:", id(train_loader))
    # logging.info("处理数据完成")
    return train_loader, valid_loader, test_loader

def train(train_loader, valid_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if CONFIG["device"] == "cpu":
        device = "cpu"

    model = Classifier().to(device=device)
    model.device = device # ?

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    for epoch in range(CONFIG["num_epochs"]):
        logger.info(f"epoch: {epoch + 1} start.")

        if CONFIG["do_semi"]:
            pass

        model.train()

        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{CONFIG['num_epochs']:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()

        valid_loss = []
        valid_accs = []

        for batch in tqdm(valid_loader):
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch + 1:03d}/{CONFIG['num_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        logger.info("saving model...")
        torch.save(model, CONFIG["save_path"])


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = pre_datasets()
    # print("after id:", id(train_loader))
    train(train_loader, valid_loader)
