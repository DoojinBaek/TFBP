import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli

class ConvNet(nn.Module):
    def __init__(self,nummotif,motiflen,poolType,mode,learning_rate,dropprob,beta1,beta2,beta3,beta4,device):
        super(ConvNet, self).__init__()
        self.device = device
        self.poolType=poolType
        self.mode=mode
        self.learning_rate=learning_rate
        self.dropprob=dropprob
        # for conv1
        self.wConv1=torch.randn(nummotif,4,motiflen).to(device)
        torch.nn.init.xavier_uniform_(self.wConv1)
        self.wConv1.requires_grad=True
        self.wRect1=torch.randn(nummotif).to(device)
        torch.nn.init.normal_(self.wRect1)
        self.wRect1=-self.wRect1
        self.wRect1.requires_grad=True
        # for conv2
        self.wConv2=torch.randn(nummotif,nummotif,motiflen).to(device)
        torch.nn.init.xavier_uniform_(self.wConv2)
        self.wConv2.requires_grad=True
        self.wRect2=torch.randn(nummotif).to(device)
        torch.nn.init.normal_(self.wRect1)
        self.wRect2=-self.wRect2
        self.wRect2.requires_grad=True
        # for FC
        self.wHidden=torch.randn(2*nummotif,32).to(device)
        self.wHiddenBias=torch.randn(32).to(device)
        if poolType=='maxavg':
            self.wHidden=torch.randn(2*nummotif,32).to(device)
        else:
            self.wHidden=torch.randn(nummotif,32).to(device)
            
        self.wNeu=torch.randn(33,1).to(device)
        self.wHiddenBias=torch.randn(32).to(device)
        torch.nn.init.xavier_uniform_(self.wNeu)
        (self.wNeu, self.wNeuBias) = self.wNeu.split(list(self.wNeu.size())[0]-1, dim=0)
        torch.nn.init.normal_(self.wHidden,mean=0,std=0.3)
        torch.nn.init.normal_(self.wHiddenBias,mean=0,std=0.3)

        self.wHidden.requires_grad=True
        self.wHiddenBias.requires_grad=True

        self.wNeu.requires_grad=True
        self.wNeuBias.requires_grad=True

        self.beta1=beta1
        self.beta2=beta2
        self.beta3=beta3
        self.beta4=beta4

    def forward_pass(self,x,mask=None,use_mask=False):
        # conv1
        conv1 = F.conv1d(x, self.wConv1, bias=self.wRect1, stride=1, padding=0)
        rect1 = conv1.clamp(min=0)
        # conv2
        conv2 = F.conv1d(rect1, self.wConv2, bias=self.wRect2, stride=1, padding=0)
        rect2 = conv2.clamp(min=0)
        maxPool, _ = torch.max(rect2, dim=2)
        if self.poolType=='maxavg':
            avgPool = torch.mean(rect2, dim=2)                     
            pool = torch.cat((maxPool, avgPool), 1)
        else:
            pool = maxPool

        hid=pool @ self.wHidden
        hid.add_(self.wHiddenBias)
        hid=hid.clamp(min=0)

        if self.mode=='training': 
            if  not use_mask:
                mask=bernoulli.rvs(self.dropprob, size=len(hid[0]))
                mask=torch.from_numpy(mask).float().to(self.device)
            hiddrop=hid*mask
            out=self.dropprob*(hiddrop @ self.wNeu)
            out.add_(self.wNeuBias)
        else:
            out=self.dropprob*(hid @ self.wNeu)
            out.add_(self.wNeuBias)
        return out,mask
       
    def forward(self, x):
        out,_=self.forward_pass(x)
        return out

class ConvNet_Base(nn.Module):
    def __init__(self,nummotif,motiflen,poolType,mode,learning_rate,dropprob,beta1,beta2,device):
        super(ConvNet_Base, self).__init__()
        self.device = device
        self.poolType=poolType
        self.mode=mode
        self.learning_rate=learning_rate
        self.dropprob=dropprob
        # for conv1
        self.wConv1=torch.randn(nummotif,4,motiflen).to(device)
        torch.nn.init.xavier_uniform_(self.wConv1)
        self.wConv1.requires_grad=True
        self.wRect1=torch.randn(nummotif).to(device)
        torch.nn.init.normal_(self.wRect1)
        self.wRect1=-self.wRect1
        self.wRect1.requires_grad=True
        # for conv2
        self.wConv2=torch.randn(nummotif,nummotif,motiflen).to(device)
        torch.nn.init.xavier_uniform_(self.wConv2)
        self.wConv2.requires_grad=True
        self.wRect2=torch.randn(nummotif).to(device)
        torch.nn.init.normal_(self.wRect1)
        self.wRect2=-self.wRect2
        self.wRect2.requires_grad=True

        self.beta1=beta1
        self.beta2=beta2

    def forward_pass(self,x,mask=None,use_mask=False):
        # conv1
        conv1 = F.conv1d(x, self.wConv1, bias=self.wRect1, stride=1, padding=0)
        rect1 = conv1.clamp(min=0)
        # conv2
        conv2 = F.conv1d(rect1, self.wConv2, bias=self.wRect2, stride=1, padding=0)
        rect2 = conv2.clamp(min=0)
        maxPool, _ = torch.max(rect2, dim=2)
        if self.poolType=='maxavg':
            avgPool = torch.mean(rect2, dim=2)                     
            pool = torch.cat((maxPool, avgPool), 1)
        else:
            pool = maxPool
        return pool
       
    def forward(self, x):
        out =self.forward_pass(x)
        return out

class FC(nn.Module):
    def __init__(self,nummotif,poolType,mode,learning_rate,dropprob,beta1,beta2,beta3,beta4,device):
        super(FC, self).__init__()
        self.device = device
        self.poolType=poolType
        self.mode=mode
        self.learning_rate=learning_rate
        self.dropprob=dropprob
        # for FC
        self.wHidden=torch.randn(2*nummotif,32).to(device)
        self.wHiddenBias=torch.randn(32).to(device)
        if poolType=='maxavg':
            self.wHidden=torch.randn(2*nummotif,32).to(device)
        else:
            self.wHidden=torch.randn(nummotif,32).to(device)
            
        self.wNeu=torch.randn(33,1).to(device)
        self.wHiddenBias=torch.randn(32).to(device)
        torch.nn.init.xavier_uniform_(self.wNeu)
        (self.wNeu, self.wNeuBias) = self.wNeu.split(list(self.wNeu.size())[0]-1, dim=0)
        torch.nn.init.normal_(self.wHidden,mean=0,std=0.3)
        torch.nn.init.normal_(self.wHiddenBias,mean=0,std=0.3)

        self.wHidden.requires_grad=True
        self.wHiddenBias.requires_grad=True

        self.wNeu.requires_grad=True
        self.wNeuBias.requires_grad=True

        self.beta3=beta3
        self.beta4=beta4

    def forward_pass(self,x,mask=None,use_mask=False):

        hid= x @ self.wHidden
        hid.add_(self.wHiddenBias)
        hid=hid.clamp(min=0)

        if self.mode=='training': 
            if  not use_mask:
                mask=bernoulli.rvs(self.dropprob, size=len(hid[0]))
                mask=torch.from_numpy(mask).float().to(self.device)
            hiddrop=hid*mask
            out=self.dropprob*(hiddrop @ self.wNeu)
            out.add_(self.wNeuBias)
        else:
            out=self.dropprob*(hid @ self.wNeu)
            out.add_(self.wNeuBias)
        return out,mask
       
    def forward(self, x):
        out,_=self.forward_pass(x)
        return out

class MTL_Model():
    def __init__(self, task1, task2, nummotif,motiflen,poolType,mode,learning_rate,dropprob,beta1,beta2,beta3,beta4, device):
        super(MTL_Model, self).__init__()

        # hard-shared layers
        self.net = ConvNet(nummotif,motiflen,poolType,mode,learning_rate,dropprob,beta1,beta2,beta3,beta4, device).to(device)
        # task specific
        self.net1 = ConvNet(nummotif,motiflen,poolType,mode,learning_rate,dropprob,beta1,beta2,beta3,beta4, device).to(device)
        self.net2 = ConvNet(nummotif,motiflen,poolType,mode,learning_rate,dropprob,beta1,beta2,beta3,beta4, device).to(device)
    
    def forward(self, x, task):
        pool = self.net(x)
        # task -> 64(batch size) by 1 tensor
        # pool -> 64 by 2d tensor
        
        return [self.net1(pool), self.net2(pool)]
