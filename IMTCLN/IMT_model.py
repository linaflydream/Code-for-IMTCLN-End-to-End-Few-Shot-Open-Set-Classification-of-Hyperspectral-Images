import torch
import torch.nn as nn
import torch.nn.init as init
class IMT(nn.Module):
    def __init__(self, CLASS_NUM, patch_size, n_bands):
        super(IMT, self).__init__()
        self.nc = CLASS_NUM+1
        if patch_size==5:
            self.conv1 = nn.Conv2d(n_bands, 100, 3)
            self.conv2 = nn.Conv2d(100, 50, 2)
            self.conv3 = nn.Conv2d(50, 30, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(30, self.nc)
        elif patch_size==3:
            self.conv1 = nn.Conv2d(n_bands, 100, 3,padding=1)
            self.conv2 = nn.Conv2d(100, 50, 2)
            self.conv3 = nn.Conv2d(50, 30, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(30, self.nc)
        elif patch_size==7:
            self.conv1 = nn.Conv2d(n_bands, 100, 3)
            self.conv2 = nn.Conv2d(100, 50, 3)
            self.conv3 = nn.Conv2d(50, 30, 3)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(30, self.nc)
        elif patch_size==9:
            self.conv1 = nn.Conv2d(n_bands, 100, 3)
            self.conv2 = nn.Conv2d(100, 50, 3)
            self.conv3 = nn.Conv2d(50, 30, 5)

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(30, self.nc)
        elif patch_size==11:
            self.conv1 = nn.Conv2d(n_bands, 25, 3)
            self.conv2 = nn.Conv2d(25, 25, 3)
            self.conv3 = nn.Conv2d(25, 25, 7)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(25, self.nc)
        self.fc2 = nn.Linear(self.nc, self.nc * 2)
        self.trans1 = nn.Softplus()
        self.trans2 = nn.Softmax(dim=1)
        self.trans3 = nn.Softmax(dim=2)

        
    def forward(self, x,flag=True):
            if flag:
                x = x.to(torch.float32)

                c1 = self.trans1(self.conv1(x))
                c2 = self.trans1(self.conv2(c1))
                c3 = self.trans1(self.conv3(c2))

                z = self.flatten(c3)
                
                h0 = self.fc1(z)
                soft_h = self.trans2(h0)
                return h0,soft_h ,z
            else:
                x = x.to(torch.float32)

                c1 = self.trans1(self.conv1(x))
                c2 = self.trans1(self.conv2(c1))
                c3 = self.trans1(self.conv3(c2))

                z = self.flatten(c3)
                
                h0 = self.fc1(z)
                soft_h = self.trans2(h0)
                return soft_h
   

    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, 128, self.patch_size, self.patch_size))
            x = self.relu(self.conv1(x))
            x = self.maxpool1(x)
            x = self.relu(self.conv2(x))
            x = self.maxpool2(x)
            x = x.view(x.shape[0], -1)
            s = x.size()[1]
        return s


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def kaiming_init(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0.03)
