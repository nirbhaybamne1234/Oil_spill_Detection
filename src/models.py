import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1=Block(1,base); self.e2=Block(base,base*2); self.e3=Block(base*2,base*4)
        self.pool=nn.MaxPool2d(2); self.mid=Block(base*4,base*8)
        self.up3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=Block(base*8,base*4)
        self.up2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=Block(base*4,base*2)
        self.up1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=Block(base*2,base)
        self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); z=self.mid(self.pool(c))
        z=self.d3(torch.cat([self.up3(z),c],1)); z=self.d2(torch.cat([self.up2(z),b],1))
        z=self.d1(torch.cat([self.up1(z),a],1)); return self.out(z)

class SmallUNet(nn.Module):
    """Smaller encoder-decoder baseline for comparison."""
    def __init__(self, base=16):
        super().__init__()
        self.enc1=Block(1,base); self.enc2=Block(base,base*2)
        self.pool=nn.MaxPool2d(2); self.mid=Block(base*2,base*4)
        self.up=nn.ConvTranspose2d(base*4,base*2,2,2)
        self.dec=Block(base*4,base*2)
        self.up2=nn.ConvTranspose2d(base*2,base,2,2)
        self.dec2=Block(base*2,base); self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        a=self.enc1(x); b=self.enc2(self.pool(a)); z=self.mid(self.pool(b))
        z=self.dec(torch.cat([self.up(z),b],1))
        z=self.dec2(torch.cat([self.up2(z),a],1))
        return self.out(z)
