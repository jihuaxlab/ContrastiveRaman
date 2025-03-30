import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=300,
                    help='Dimension of representations')
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=1200,
                    help='Num of spectra dimension')
parser.add_argument('--model', type=str, default='CLR',
                    help='Model')                   

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

set_seed(args.seed,args.cuda)

def mask(x):
    xx = torch.zeros_like(x)
    for i in range(x.shape[0]):
        a = np.random.randint(0,args.d)
        delta = np.random.uniform(0.3,0.7)
        if a>(1-delta)*args.d: b = args.d
        else: b = int(a+delta*args.d)
        xx[i,:,:a] = x[i,:,:a]
        xx[i,:,b:] = x[i,:,b:]
    
    return xx

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1
        )
        self.pool1 = nn.AvgPool1d(2) # d/2 dim
        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1
        )
        self.pool2 = nn.AvgPool1d(2) # d/4 dim
        self.conv = nn.Sequential(
            self.conv1,self.pool1,
            self.conv2,self.pool2,
        )
        self.lin = nn.Linear(args.d//4,args.c)

    def forward(self,x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.lin(x)
        return x

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet,self).__init__()
        self.conv1 = nn.LSTM(
            input_size=args.d,
            hidden_size=args.hidden,
        )
        self.conv2 = nn.LSTM(
            input_size=args.hidden,
            hidden_size=args.hidden,
        )
        self.lin = nn.Linear(args.hidden,args.c)
    
    def forward(self,x):
        x = self.conv1(x)[0]
        x = self.conv2(x)[0]
        x = x.squeeze()
        x = self.lin(x)
        return x

class CLR(CNNNet):
    def __init__(self):
        super(CLR,self).__init__()
    
    def norm(self,z):
        return z/torch.norm(z,dim=-1,keepdim=True)
    
    def forward(self,x1,x2,sep=False):
        x1 = self.conv(x1)
        x1 = x1.squeeze()
        x2 = self.conv(x2)
        x2 = x2.squeeze()
        z1 = self.norm(self.lin1(x1))
        z2 = self.norm(self.lin1(x2))
        z = 0.5*(z1+z2)
        x = self.lin(z)
        return x,z1,z2

if args.c == 2: v = pd.read_csv('bin.csv').values
else: v = pd.read_csv('multi.csv').values
x = v[:,:args.d]
y = v[:,-1]

def classifier(tid,vid,tag=1):
    if args.model == 'CNN': clf = CNNNet()
    elif args.model == 'LSTM': clf = LSTMNet()
    else: clf = CLR()
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    xt = torch.from_numpy(x[tid]).float().unsqueeze(1)
    xv = torch.from_numpy(x[vid]).float().unsqueeze(1)
    yt = torch.LongTensor(y[tid])
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    
    x1t = mask(xt)
    x2t = mask(xt)
    print('Fold ',tag)
    for e in range(args.epochs):
        clf.train()
        if args.model == 'CLR':
            tau = 0.2
            alpha = 0.2
            gamma = 2
            mat,z1,z2 = clf(x1t,x2t)
            ce = F.cross_entropy(mat,yt,reduction='none')
            p = torch.exp(-ce)
            l_focal = torch.mean(torch.pow(1-p,gamma)*ce)
            l_c = -F.log_softmax(torch.mm(z1,z2.t())/tau,dim=-1)
            l_c = torch.diag(l_c).mean()
            loss = l_c+alpha*l_focal
        else:
            mat = clf(xt)
            loss = F.cross_entropy(mat,yt)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    if args.model == 'CLR':
        mat,z1,z2 = clf(xv,xv)
    else:
        mat = clf(xv)
    
    if args.cuda: mat = mat.cpu()
    predict = mat.argmax(dim=-1)
    return predict.numpy()

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=args.seed)
predict = np.zeros(len(y))
for i,(tid,vid) in enumerate(kf.split(x,y)):
    predict[vid] = classifier(tid,vid,i)

print('Acc=',accuracy_score(y,predict))
print('Pre=',precision_score(y,predict,average='macro'))
print('Rec=',recall_score(y,predict,average='macro'))
print('F1=',f1_score(y,predict,average='macro'))