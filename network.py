import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli

class ConvNet(nn.Module):
    def __init__(self,nummotif,motiflen,poolType,neuType,mode,learning_rate,optimizer,dropprob,beta1,beta2,beta3, device, reverse_complemet_mode):
        super(ConvNet, self).__init__()
        self.device = device
        self.poolType=poolType
        self.neuType=neuType
        self.mode=mode
        self.learning_rate=learning_rate
        self.reverse_complemet_mode=reverse_complemet_mode
        self.wConv=torch.randn(nummotif,4,motiflen).to(device)
        torch.nn.init.xavier_uniform_(self.wConv)
        self.wConv.requires_grad=True
        self.wRect=torch.randn(nummotif).to(device)
        torch.nn.init.normal_(self.wRect)
        self.wRect=-self.wRect
        self.wRect.requires_grad=True
        self.dropprob=dropprob
        self.wHidden=torch.randn(2*nummotif,32).to(device)
        self.wHiddenBias=torch.randn(32).to(device)

        if neuType=='nohidden':
            if poolType=='maxavg':
                self.wNeu=torch.randn(2*nummotif+1,1).to(device)
            else:
                self.wNeu=torch.randn(nummotif+1,1).to(device)

            torch.nn.init.xavier_uniform_(self.wNeu)
            (self.wNeu, self.wNeuBias) = self.wNeu.split(list(self.wNeu.size())[0]-1, dim=0)

        else:
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
    
    def divide_two_tensors(self,x):
        l=torch.unbind(x)
        list1=[l[2*i] for i in range(int(x.shape[0]/2))]
        list2=[l[2*i+1] for i in range(int(x.shape[0]/2))]
        x1=torch.stack(list1,0)
        x2=torch.stack(list2,0)
        return x1,x2

    def forward_pass(self,x,mask=None,use_mask=False):
        conv=F.conv1d(x, self.wConv, bias=self.wRect, stride=1, padding=0)
        rect=conv.clamp(min=0)
        maxPool, _ = torch.max(rect, dim=2)
        if self.poolType=='maxavg':
            avgPool= torch.mean(rect, dim=2)                          
            pool=torch.cat((maxPool, avgPool), 1)
        else:
            pool=maxPool
        if(self.neuType=='nohidden'):
            if self.mode=='training': 
                if  not use_mask:
                    mask=bernoulli.rvs(self.dropprob, size=len(pool[0]))
                    mask=torch.from_numpy(mask).float().to(self.device)
                pooldrop=pool*mask
                out=pooldrop @ self.wNeu
                out.add_(self.wNeuBias)
            else:
                out=self.dropprob*(pool @ self.wNeu)
                out.add_(self.wNeuBias)       
        else:
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
        if not  self.reverse_complemet_mode:
            out,_=self.forward_pass(x)
        else:
            x1,x2=self.divide_two_tensors(x)
            out1,mask=self.forward_pass(x1)
            out2,_=self.forward_pass(x2,mask,True)
            out=torch.max(out1, out2)
        
        return out

class MTL_Model():
    def __init__(self, nummotif,motiflen,poolType,neuType,mode,learning_rate,optimizer,dropprob,beta1,beta2,beta3, device, reverse_complemet_mode,):
        super(MTL_Model, self).__init__()

        # 이 부분을 base network랑 specific한 network로 나눠서 작성해야함
        # -> self.base = ~
        # -> self.net1 = self.base + specific
        # -> self.net2 = self.base + specific
        # loss 식에 넣을 때 self.base.~ 들이랑 self.net1.~, self.net2.~ 부분들 다 넣어줘야함
        self.net1 = ConvNet(nummotif,motiflen,poolType,neuType,mode,learning_rate,optimizer,dropprob,beta1,beta2,beta3, device, reverse_complemet_mode).to(device)
        self.net2 = ConvNet(nummotif,motiflen,poolType,neuType,mode,learning_rate,optimizer,dropprob,beta1,beta2,beta3, device, reverse_complemet_mode).to(device)
    
    def forward(self, x):
        return [self.net1(x), self.net2(x)]
