import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from bank import tensor_print, DataLoader, BRAINDataset, realEEGNet, test, get_device, data_info, rand_lab_gen, rand_data_gen

'''
EEGNet re-build main script.

REFs:
    Original Paper: https://arxiv.org/pdf/1611.08024.pdf
    Keras network:  https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

'''

class depthwise_Conv2d(nn.Module):
    'ALT: https://github.com/rosinality/depthwise-conv-pytorch'
    # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/10
    def __init__(self, nin, kernels_per_layer, nout, chans):
        super(depthwise_Conv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=[chans, 1],
                                   padding=0, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        # print('Post Depth Conv DIMS: ', out.shape)
        return out

class separable_Conv2d(nn.Module):
    # https://discuss.pytorch.org/t/using-optimised-depthwise-convolutions/11819/14
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(separable_Conv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        # print('Post Sep Conv DIMS: ', x.shape)
        return x

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        'Parameters: '
        kernLength = 64
        F1 = 8
        F2 = 16
        D = 2
        chans = 2
        drop_rate = 0.5
        '------Pod 1'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=F1,
                               kernel_size=[1, kernLength])
        self.conv1.add_module('BN1', nn.BatchNorm1d(num_features=F1))
        self.conv1.add_module('DP1', depthwise_Conv2d(nin=F1, kernels_per_layer=D, nout=F1, chans=chans))
        self.conv1.add_module('BN2', nn.BatchNorm1d(num_features=F1))
        self.conv1.add_module('eLU1', nn.ELU(inplace=False))
        self.conv1.add_module('AP1', nn.AvgPool2d(kernel_size=[1, 4]))
        self.conv1.add_module('DO1', nn.Dropout(p=drop_rate, inplace=False))
        '------Pod 2'
        self.conv2 = separable_Conv2d(in_channels=F1, out_channels=F2,
                                         kernel_size=[1, 16])
        self.conv2.add_module('BN3', nn.BatchNorm1d(num_features=F2))
        self.conv2.add_module('eLU2', nn.ELU(inplace=False))
        self.conv2.add_module('AP2', nn.AvgPool2d(kernel_size=[1, 8]))
        self.conv2.add_module('DO2', nn.Dropout(p=drop_rate, inplace=False))
        '------FC'
        self.fc1 = nn.Linear(in_features=2592, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=2)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        printer = False
        'Layer 1: '
        x = self.conv1(x)
        tensor_print(x, indicator='Conv 1 DIMS: ', printer=printer)
        'Layer 2: '
        x = self.conv2(x)
        tensor_print(x, indicator='Conv 2 DIMS: ', printer=printer)
        'FLatten: '
        x = x.view(x.size(0), -1)
        tensor_print(x, indicator='Flattened DIMS: ', printer=printer)
        'FC 1: '
        x = self.fc1(x)
        tensor_print(x, indicator='FC 1 DIMS: ', printer=printer)
        'FC 2: '
        x = self.fc2(x)
        tensor_print(x, indicator='FC 2 DIMS: ', printer=printer)
        'Softmax: '
        x = x.type(torch.FloatTensor)
        x = self.softmax1(x)
        tensor_print(x, indicator='SFM 1 DIMS: ', printer=printer)
        return x

'Network Initialization'
net = EEGNet().cuda(0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.000001)

'Data Loading: Rand  /  French'
data_type = 'Rand'
if data_type == 'Rand':
    X_train, X_val, X_test = rand_data_gen(num_events=1000, num_samps=120, num_chans=2, verbose=0)
    y_train, y_val, y_test = rand_lab_gen(num_events=1000, verbose=0)

'---Data Info---'
data_info(X_train, y_train, X_val, y_val, X_test, y_test)

'---Establish device for compute'
device = get_device()

'---Initialize Training and Test dataset loading'
train_data = BRAINDataset(X_train, y_train)
val_data   = BRAINDataset(X_val, y_val)
test_data  = BRAINDataset(X_test, y_test)

'---Data Loading Functions---'
b_s = 32
trainloader = DataLoader(train_data, batch_size=b_s, shuffle=True)
valloader = DataLoader(val_data, batch_size=b_s, shuffle=True)
testloader = DataLoader(test_data, batch_size=b_s, shuffle=True)

'---Execute Traing---'
num_epochs = 10
model = realEEGNet(net, trainloader, valloader, device, optimizer,
           criterion, num_epochs=num_epochs, live_plot=True, verbose=1)

'---Test Network---'
test_acc = test(model, testloader, device, verbose=1)
