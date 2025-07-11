import torch
import torch.nn as nn
from torchvision import models
from BCNN import BCNN
import MRNN


def init_linear(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class BaseCNN(nn.Module):

    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        outdim = 1
        if self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif self.config.backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=True).features

        self.representation = BCNN()
        self.fc = nn.Sequential(nn.Linear(512 * 512, 512),
                                nn.ReLU(),
                                nn.Linear(512,64),
                                nn.ReLU(),
                                nn.Linear(64,outdim))       
        self.fc.apply(init_linear)


        self.nlm1 = MRNN.MonotonicInvNet(block_num=1)
        self.nlm2 = MRNN.MonotonicInvNet(block_num=1)
        self.nlm3 = MRNN.MonotonicInvNet(block_num=1)
        self.nlm4 = MRNN.MonotonicInvNet(block_num=1)
        self.nlm5 = MRNN.MonotonicInvNet(block_num=1)
        self.nlm6 = MRNN.MonotonicInvNet(block_num=1)

        self.trans1 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.trans2 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.trans3 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.trans4 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.trans5 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)
        self.trans6 = MRNN.MonotonicFunc(3,[100,100,100], nb_steps=100)



    def forward(self, x, tag, minmos, maxmos, device,istest=False):
        
       if not istest :
            [b,d,c,h,w] = x.shape 
            x = x.view(-1,c,h,w)
            if self.config.backbone == 'resnet34':
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)
            elif self.config.backbone == 'vgg16':
                x = self.backbone(x)

            x = self.representation(x)

            x = self.fc(x) 

            x = x.view(b,d)
            DNN_x = x

            x = torch.split(x, 1, dim=1) 

            h = torch.zeros(x[0].shape[0], 2).to(device)
            h1 = torch.zeros(1, 2).to(device)
            y11,y22 = [], []
            dnn_min, dnn_max = [], []
            minmos = torch.tensor(minmos)
            maxmos = torch.tensor(maxmos)
            for sap_id in range(len(x)):
                x1 = x[sap_id]
                x2 = getattr(self,'trans'+str(sap_id+1),'None')(x1,h)
                y1,y2 = getattr(self, 'nlm'+str(sap_id+1), 'None')(x1,x2,h)
                mos_min = minmos[sap_id].to(device).unsqueeze(0).unsqueeze(0)
                mos_max = maxmos[sap_id].to(device).unsqueeze(0).unsqueeze(0)
                dnn_min_x1, dnn_min_x2 = getattr(self, 'nlm' + str(sap_id + 1), 'None')(mos_min,mos_min,h1,rev = True)

                dnn_max_x1, dnn_max_x2 = getattr(self, 'nlm' + str(sap_id + 1), 'None')(mos_max,mos_max,h1,rev = True)
                dnn_min.append(dnn_min_x1)
                dnn_max.append(dnn_max_x1)

                y11.append(y1[:,0])
                y22.append(y2[:,0])
            y11 = torch.stack(y11,dim=0) 
            y11 = torch.transpose(y11,1,0)
            y22 = torch.stack(y22,dim=0) 
            y22 = torch.transpose(y22,1,0)
            dnn_min = torch.cat(dnn_min, dim = 0).squeeze(1)
            dnn_max = torch.cat(dnn_max, dim = 0).squeeze(1)
    
            return y11,y22,DNN_x,dnn_min,dnn_max
       else:
            b,c,h,w = x.shape
            DNN_x = []
            if self.config.backbone == 'resnet34':
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)
            elif self.config.backbone == 'vgg16':
                x = self.backbone(x)

            x = self.representation(x)

            x = self.fc(x)           
            DNN_x = x
            y1 = torch.zeros_like(x)
            y2 = torch.zeros_like(x)
            h = torch.zeros(x.shape[0], 2).to(device)
            for i in range(b):
                tag_ = tag[i].data.item()
                x2 = getattr(self, 'trans' + str(tag_), 'None')(x, h)
                yy1,yy2 = getattr(self, 'nlm'+str(tag_), 'None')(x,x2,h)
                y1[i] = yy1[:,0]
                y2[i] = yy2[:,0]
            return y1,y2, DNN_x