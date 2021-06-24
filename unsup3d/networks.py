import torch
import torch.nn as nn
import torchvision
from . import utils

EPS = 1e-7
class Encoder(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 256x256 -> 128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*16, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*16, nf*32, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*32, nf*16, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*16, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)

## copy from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        else:
            self.downsample = None
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class DepthMapNet(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(DepthMapNet, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 256x256 -> 128x128
            nn.GroupNorm(16, nf),
            # nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16*2, nf*2),
            # nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*4, nf*4),
            # nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*8, nf*8),
            # nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, nf*16, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*16, nf*16),
            # nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # BasicBlock(nf*4, nf*8, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            # BasicBlock(nf*8, nf*4, norm_layer=nn.InstanceNorm2d),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32
            # nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            # nn.GroupNorm(16, nf),
            # nn.BatchNorm2d(nf),
            # nn.ReLU(inplace=True),
            nn.Conv2d(nf*16, nf*8, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16*8, nf*8),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16*4, nf*4),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            # # nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            # # nn.GroupNorm(16, nf),
            # # nn.BatchNorm2d(nf),
            # # nn.ReLU(inplace=True),
            # nn.Conv2d(nf*2, nf, kernel_size=5, stride=1, padding=2, bias=False),
            # nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64 -> 128x128
            # # nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 128x128
            # # nn.GroupNorm(16, nf),
            # # nn.BatchNorm2d(nf),
            # # nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False),
            ]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)
    def forward(self, input):
        return self.network(input)

class AlbedoMapNet(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(AlbedoMapNet, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 256x256 -> 128x128
            nn.GroupNorm(16, nf),
            # nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16*2, nf*2),
            # nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*4, nf*4),
            # nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*8, nf*8),
            # nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, nf*16, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*16, nf*16),
            # nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # BasicBlock(nf*4, nf*8, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*16, nf*16, norm_layer=nn.InstanceNorm2d),
            # BasicBlock(nf*8, nf*4, norm_layer=nn.InstanceNorm2d),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32
            # nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            # nn.GroupNorm(16, nf),
            # nn.BatchNorm2d(nf),
            # nn.ReLU(inplace=True),
            nn.Conv2d(nf*16, nf*8, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16*8, nf*8),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16*4, nf*4),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            # # nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            # # nn.GroupNorm(16, nf),
            # # nn.BatchNorm2d(nf),
            # # nn.ReLU(inplace=True),
            # nn.Conv2d(nf*2, nf, kernel_size=5, stride=1, padding=2, bias=False),
            # nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64 -> 128x128
            # # nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 128x128
            # # nn.GroupNorm(16, nf),
            # # nn.BatchNorm2d(nf),
            # # nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 128x128 -> 256x256
            nn.Conv2d(nf*2, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False),
            ]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)
    def forward(self, input):
        return self.network(input)

class ConfNet(nn.Module):
    def __init__(self, cin, cout, nf=64):
        super(ConfNet, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 256x256 -> 128x128
            nn.GroupNorm(16, nf),
            # nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16*2, nf*2),
            # nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*4, nf*4),
            # nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*8, nf*8),
            # nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(nf*8, nf*16, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            # nn.GroupNorm(16*16, nf*16),
            # nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(nf*8, nf*8, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*8, nf*8, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*8, nf*8, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*8, nf*8, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*8, nf*8, norm_layer=nn.InstanceNorm2d),
            BasicBlock(nf*8, nf*8, norm_layer=nn.InstanceNorm2d),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32
            nn.Conv2d(nf*8, nf*4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            # nn.GroupNorm(16*4, nf*4),
            # nn.BatchNorm2d(nf*4),
            # nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            ]
        self.network = nn.Sequential(*network)

        out_net1 = [
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 128x128
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 256x256
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2, bias=False),  # 256x256
            nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)

        out_net2 = [nn.Conv2d(nf*2, 2, kernel_size=3, stride=1, padding=1, bias=False),  # 64x64
                    nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        out = self.network(input)
        return self.out_net1(out), self.out_net2(out)


class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.slice6 = nn.Sequential()
        self.slice7 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 19):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 21):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 23):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 26):
            self.slice7.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def __call__(self, im1, im2, mask=None,conf_sigma=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice5(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice6(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice7(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        selected_feats = feats[2:3] # use relu3_3 features only, works best

        for f1, f2 in selected_feats:  
            loss = (f1-f2)**2
            if conf_sigma is not None:
                dim_loss = loss.shape
                conf_sigma = torch.nn.functional.interpolate(conf_sigma, size=dim_loss[-1], mode="nearest")
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw))
                mask0 = mask0.expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                    loss = loss.mean()
            losses += [loss]

        return sum(losses) # use mean in case you calcualte the loss based on several feature maps
