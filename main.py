import numpy as np
import pandas as pd
import os
import cv2
from spilt_data import spilt_train,spilt_test
from readfile import readfile
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.autograd import Variable
from torchvision import datasets, models

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from run_test import train,test,run_pred

# spilt train video to img
spilt_train(path="train",frame=5)

#spilt test video to img
spilt_test(path ="test",frame=5)

#train list
frame_list = [i for i in os.listdir() if "frame" in i and "test" not in i]

#total train num
TOTAL_length = 0
for i in range(len(frame_list)):
    TOTAL_length +=  len(os.listdir(f"{frame_list[i]}"))

print(f"train_num : {TOTAL_length}")
#total test num
test_num  = len(os.listdir("test_frame"))
print(f"test_num : {test_num}")
#load file

x,y = readfile(num = TOTAL_length,label=True)

test_x = readfile(num = 160656,test=True,label=None)


#split data to train and validation

(X_train,x_val,y_train,y_val)=train_test_split(x,y,test_size=0.1)


#show all dataset shape
print(f"train x and y : {X_train.shape} , {y_train.shape}")
print(f"Val x and y :   {x_val.shape}   , {y_val.shape}")
print(f"test  x and y : {test_x.shape}  ")

#transforms img

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



# IMG_dataset
class img_Dataset(Dataset):
    def __init__(self,imgs,label,transform=None, target_transform=None):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.label = label

    def __getitem__(self, index):

        fn = self.imgs[index]


        if self.transform is not None:
            fn = self.transform(fn)

        if self.label is not None:
            label = self.label[index]
            return fn,label
        else:
            return fn

    def __len__(self):
        return len(self.imgs)

#DATALoader
train_data= img_Dataset(imgs = X_train , label= y_train,transform=preprocess)
val_data= img_Dataset(imgs = x_val   , label= y_val,transform=preprocess)
test_data = img_Dataset(imgs = test_x  , label =None ,transform=test_preprocess)

train_loader = DataLoader(train_data , batch_size=256  ,  shuffle = True)
val_loader = DataLoader( val_data , batch_size=256  ,  shuffle = True)

test_loader  = DataLoader(test_data  , batch_size = 256 , shuffle = False)

print(len(train_loader),len(val_loader),len(test_loader))

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


seed = 42
lr=0.001

model = models.resnet18(pretrained = False)
model.fc = nn.Linear(512,39)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=3, min_lr=0.000001)

# training
model_ft,model_path = train(model,train_loader,val_loader, optimizer, criterion, 40 ,scheduler,patience=4)

run_pred(model,model_path,test_loader)