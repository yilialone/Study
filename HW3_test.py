import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
import re
from torchvision import transforms
from PIL import Image
from torch import nn
from tqdm import tqdm

train_tfm = transforms.Compose([
    transforms.Resize((256, 256)),  # 图片改变尺寸
    transforms.RandomRotation(60),  # 随机旋转(在-60到60之间随机进行旋转)
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomGrayscale(p=0.1),  # 以10%的概率将图像转为灰度

    transforms.ToTensor(),  # 将图像数据转为张量形式
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 图像标准归一化，数值为pytorch通用数值

])
test_tfm = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class CNN_dataset(Dataset):
    def __init__(self, filepath, tfm):
        self.list_images, self.list_labels = [], []
        self.transform = tfm
        for filename in os.listdir(filepath):
            # img=cv2.imread(filepath+"/"+filename)  # 这里不用opencv读取，如果用opencv读取需要转换成PIL.image，
            img = Image.open(filepath + "/" + filename)
            self.list_images.append(
                self.transform(img.copy()))  # open之后，实际图片并未加载到内存，主动调用action操作，将open后的对象copy到新对象，实现顺序读取、创建图片。
            img.close()

            label = re.findall(r'.*(?=_)', filename)
            self.list_labels.append(int(label[0]))

        self.list_labels = torch.tensor(self.list_labels)

    def __len__(self):
        return len(self.list_labels)

    def __getitem__(self, index):
        return self.list_images[index], self.list_labels[index]


class CNN_hw3(nn.Module):
    def __init__(self):
        super(CNN_hw3, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
            # [3,256,256] -> [64,256,256]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # -> [32,128,128]

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),  # -> [64,128,128]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # -> [64,64,64]

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),  # -> [128,64,64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # -> [128,32,32]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),  # -> [256,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # -> [256,16,16]

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),  # -> [512,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # ->[512,8,8]

        )
        self.layer_result = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 11)
        )

    def forward(self, x):
        out = self.layer(x)
        return self.layer_result(out)


def train_HW3(device, train_data, valid_data, epoch, moudel, loss_fun, optimizer):
    for i in range(epoch):
        train_loss, train_acc = [], []
        valid_loss, valid_acc = [], []
        for n, (image, label) in enumerate(tqdm(train_data)):
            image = image.to(device)
            label = label.to(device)

            pred = moudel(image)
            loss = loss_fun(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = torch.eq(pred.argmax(dim=-1), label).float().mean()

            train_loss.append(loss.item())
            train_acc.append(acc.item())

        print("epoch({}/{})train_loss:{}".format(i + 1, epoch, sum(train_loss) / len(train_loss)))

        for image, label in tqdm(valid_data):
            image, label = image.to(device), label.to(device)
            pred = moudel(image)
            loss = loss_fun(pred, label)
            acc = torch.eq(pred.argmax(dim=-1), label).float().mean()

            valid_loss.append(loss.item())
            valid_acc.append(acc.item())

        print("epoch({}/{})valid_loss:{}".format(i + 1, epoch, sum(valid_loss) / len(valid_loss)))
        print("epoch({}/{})valid_acc:{}".format(i + 1, epoch, sum(valid_acc) / len(valid_acc)))
#测试训练函数
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Mydataset = CNN_dataset("E:\\New desktop\\Debug log of the code\\reviewing neural networks\\ml2023spring-hw3\\train",
                            train_tfm)
    train_data_loader = DataLoader(dataset=Mydataset, batch_size=16, shuffle=True, num_workers=4)

    valid_dataset = CNN_dataset(
        "E:\\New desktop\\Debug log of the code\\reviewing neural networks\\ml2023spring-hw3\\valid",
        test_tfm)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)

    epoch = 1
    moudel = CNN_hw3()
    moudel.to(device)
    loss_fun = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam(moudel.parameters(), lr=1e-4)

    train_HW3(device, train_data_loader, valid_data_loader, epoch, moudel, loss_fun, optimizer)
