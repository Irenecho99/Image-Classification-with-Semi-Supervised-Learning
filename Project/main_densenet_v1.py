import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import os
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import  math
import torch.nn.functional as F

## random seed
seed = 10
def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(seed)

## DenseNet
class Bottleneck(nn.Module):

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([out, x], 1)

class Transition(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return F.avg_pool2d(out, 2)

class DenseNet(nn.Module):

    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes  = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes  = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes  = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes  = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes  = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes  = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.fc1 = nn.Sequential(nn.Linear(4096, 1024),nn.ReLU(inplace=True),nn.Linear(num_planes, num_classes))

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        return self.fc1(out)

def DenseNet121(num_classes):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_classes)

model = DenseNet121(200).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

## Dataset, DataLoader
class FlameSet(Data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.1), transforms.ToTensor()])
        self.dodata = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        if data.size(0) != 3:
            data = torch.cat((data, data, data), 0)
        data = self.dodata(data)
        return data

    def __len__(self):
        return len(self.imgs)


#Vali dataset
class ValiSet(Data.Dataset):
    def __init__(self, root, labels):

        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, 'val_{}.JPEG'.format(k)) for k in range(len(imgs))]
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.1), transforms.ToTensor()])
        self.dodata = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.labels = labels

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        if data.size(0) != 3:
            data = torch.cat((data, data, data), 0)
        data = self.dodata(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

#DataL Augumentation
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageFolder('./images_data/train', transform=data_transform)
labtrain_loader = Data.DataLoader(dataset=dataset, batch_size=20, shuffle=True)
unlabdataSet = FlameSet('./images_data/unlabel_from_train')
unlab_loader = Data.DataLoader(dataset=unlabdataSet, batch_size=20, shuffle=True)

with open('./images_data/val/val_annotations.txt', 'r', encoding='utf8') as fc:
    vali = fc.readlines()
valilabels = []
for line in vali:
    valilabels.append(line)
fc.close()
valilab = []
for i in valilabels:
    valilab.append(dataset.class_to_idx[i.split('\t')[1]])
dataSet = ValiSet('./images_data/val/images', valilab)
vali_loader = Data.DataLoader(dataset=dataSet, batch_size=20, shuffle=False)


vali_loss = []
for epoch in tqdm(range(270)):
    if epoch < 60:
        for data, target in labtrain_loader:  ##supervised
            data = data.cuda()
            target = target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
    elif (epoch < 240) & (epoch >= 60):  ##semi-supervised
        unlabdata = iter(unlab_loader)
        for train, label in labtrain_loader:
            train = train.cuda()
            label = label.cuda()
            unlab = next(unlabdata)
            unlab = unlab.cuda()
            train, label = Variable(train), Variable(label)
            unlab = Variable(unlab)
            output = model(unlab)
            fake_target = Variable(output.data.max(1)[1].view(-1))
            optimizer.zero_grad()
            output1 = model(train)
            output2 = model(unlab)
            loss = loss_func(output1, label) + ((epoch - 60) / 180) * 0.3 * loss_func(output2, fake_target)
            loss.backward()
            optimizer.step()
    elif epoch >= 240:
        unlabdata = iter(unlab_loader)
        for train, label in labtrain_loader:
            train = train.cuda()
            label = label.cuda()
            unlab = next(unlabdata)
            unlab = unlab.cuda()
            train, label = Variable(train), Variable(label)
            unlab = Variable(unlab)
            output = model(unlab)
            fake_target = Variable(output.data.max(1)[1].view(-1))
            optimizer.zero_grad()
            output1 = model(train)
            output2 = model(unlab)
            loss = loss_func(output1, label) + 0.3 * loss_func(output2, fake_target)
            loss.backward()
            optimizer.step()
    if epoch % 5 == 0: 
        acc = 0
        valiloss = 0
        for vali, target in vali_loader:
            vali = vali.cuda()
            target = target.cuda()
            vali, target = Variable(vali), Variable(target)
            output = model(vali)
            loss = loss_func(output, target)
            valiloss += loss.item()
            label = output.data.max(1)[1].view(-1)
            acc += sum(label == target).item()
        print('vali accuracy:{}'.format(acc / 10000))
        vali_loss.append(valiloss / 100)
torch.save(model.state_dict(), 'modeldense.pt',_use_new_zipfile_serialization=False) 
testdata=FlameSet('./images_data/test/images')
test_loader = Data.DataLoader(dataset=testdata, batch_size=20, shuffle=False)
m_state_dict = torch.load('modeldense.pt')  
model = DenseNet121(200)
model.load_state_dict(m_state_dict)
model = model.cuda()

labelsearch=dict([val,key] for key,val in dataset.class_to_idx.items()) 
testlab=[]
for testdata in test_loader:
    test = testdata.cuda()
    test = Variable(test)
    output = model(test)
    label = output.data.max(1)[1].view(-1)
    testlab.append(np.array(label.cpu()))
label=[]
for i in testlab:
    for h in i:
        label.append(labelsearch[h])
f = open('testpredict.txt',mode='w', encoding='utf8')
f.writelines(line+'\n'for line in label)
f.close()