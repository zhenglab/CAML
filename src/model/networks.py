import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, Norm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)  

class Estimating(nn.Module):
    def __init__(self, config, in_channels=3, dim=256):
        super(Estimating, self).__init__()
        if config.MODE == 1:
            self.batch_size = config.BATCH_SIZE
        else:
            self.batch_size = 1
        self.encoder0 = MaskResBlock(fin=3, fout=64, kernel_size=3, stride=1, padding=1)
        self.encoder1 = MaskResBlock(fin=64, fout=128, kernel_size=3, stride=2, padding=1)
        self.encoder2 = MaskResBlock(fin=128, fout=dim, kernel_size=3, stride=2, padding=1)

        self.igcm0 = IGCM_Learner(channels=64, batchsize=self.batch_size, norm=nn.BatchNorm2d)
        self.igcm1 = IGCM_Learner(channels=128, batchsize=self.batch_size, norm=nn.BatchNorm2d)
        self.igcm2 = IGCM_Learner(channels=256, batchsize=self.batch_size, norm=nn.BatchNorm2d)

        self.estimating_res = EstimatingResBlock()
        
        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            MaskResBlock(fin=256, fout=128, kernel_size=3, stride=1, padding=1),
            nn.UpsamplingNearest2d(scale_factor=2),
            MaskResBlock(fin=128, fout=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, fea_en0, fea_en1, fea_en2):
        mask_en0 = self.encoder0(x)
        mask_cre0 = self.igcm0(mask_en0, fea_en0)
        mask_en1 = self.encoder1(mask_cre0)
        mask_cre1 = self.igcm1(mask_en1, fea_en1)
        mask_en2 = self.encoder2(mask_cre1)
        mask_cre2 = self.igcm2(mask_en2, fea_en2)

        mid_x = self.estimating_res(mask_cre2)

        mask_logit = self.decoder(mid_x)
        
        mask_soft = self.sigmoid(mask_logit)
        mask_pred = torch.clamp(mask_logit, 0., 1.)

        return mask_logit, mask_soft, mask_pred, [mask_en0, mask_en1, mask_en2]

class Inpainting(BaseNetwork):
    def __init__(self, config):
        super(Inpainting, self).__init__()
        if config.MODE == 1:
            self.batch_size = config.BATCH_SIZE
        else:
            self.batch_size = 1

        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.encoder1 = nn.Sequential( 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))

        self.encoder2 = nn.Sequential(             
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True))

        self.inpaint_res = InpaintingResBlock()
        
        self.inpaint_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)    )
                
        self.tanh = nn.Tanh()    
        self.egcm0 = EGCM_Learner(channels=64, batchsize=self.batch_size)
        self.egcm1 = EGCM_Learner(channels=128, batchsize=self.batch_size)
        self.egcm2 = EGCM_Learner(channels=256, batchsize=self.batch_size)
        self.down1 = nn.UpsamplingNearest2d(scale_factor=.5)
        self.down2 = nn.UpsamplingNearest2d(scale_factor=.25)
        self.init_weights()

    def forward(self, x, mask=None, mask_en0=None, mask_en1=None, mask_en2=None):  
        x_en0 = self.encoder0(x)
        x_en1 = self.encoder1(x_en0)
        x_en2 = self.encoder2(x_en1)
        if mask is not None:
            fea_cre0 = self.egcm0(x_en0, mask_en0, mask)
            x_en1 = self.encoder1(fea_cre0)
            fea_cre1 = self.egcm1(x_en1, mask_en1, self.down1(mask))
            x_en2 = self.encoder2(fea_cre1)
            fea_cre2 = self.egcm2(x_en2, mask_en2, self.down2(mask))
            mid_x = self.inpaint_res(fea_cre2)
            de_x = self.inpaint_decoder(mid_x)
            output = self.tanh(de_x)
        return [x_en0, x_en1, x_en2] if mask is None else output


class IGCM_Learner(nn.Module):
    def __init__(self, channels, batchsize, norm = nn.BatchNorm2d):
        super(IGCM_Learner, self).__init__()
        self.channles = channels
        self.batchsize = batchsize
        self.norm = norm
        self.cmat = CMAT_Block(channels=self.channles, batchsize=self.batchsize, norm=self.norm)
    def forward(self, F_E0, F_I0):
        out = self.cmat(F_E0, F_I0)
        return out

class EGCM_Learner(nn.Module):
    def __init__(self, channels, batchsize, norm = nn.InstanceNorm2d):
        super(EGCM_Learner, self).__init__()
        self.channles = channels
        self.batchsize = batchsize
        self.norm = norm
        self.cmat = CMAT_Block(channels=self.channles, batchsize=self.batchsize, norm=self.norm)
        self.cmaf = CMAF_Block(in_ch=self.channles)
    def forward(self, F_I, F_E, mask):
        mutual_0 = self.cmat(F_I, F_E)
        out = self.cmaf(mutual_0, mask)
        return out


class CMAT_Block(nn.Module):
    def __init__(self, channels, batchsize, norm = nn.BatchNorm2d):
        super(CMAT_Block, self).__init__()
        self.batchsize = batchsize
        self.channels = channels
        self.add_conv = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1, stride=1, bias=False)

        self.param_adapter = nn.Sequential(
                                    nn.Conv2d(channels, channels, 3, 2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(channels, channels, 3, 2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(channels, channels, 3, padding=1),                                  
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(channels, channels, 3, padding=2))      

        self.adaptive_conv_3 = AdaptiveConv(batchsize*channels, batchsize*channels, 3, padding=1, groups=batchsize*channels, bias=False)
        self.adaptive_conv_5 = AdaptiveConv(batchsize*channels, batchsize*channels, 5, padding=2, groups=batchsize*channels, bias=False)
        self.adaptive_conv_9 = AdaptiveConv(batchsize*channels, batchsize*channels, 9, padding=4, groups=batchsize*channels, bias=False)
                                    
        self.conv1x1_U = nn.Conv2d(channels, channels, 1, 1)
        self.conv1x1_V = BasicConv2d(channels, channels, Norm=norm, kernel_size=1, padding=0)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x1, x2):
        (b, c, h, w) = x1.shape
        cat_fea = torch.cat((x1, x2), dim=1)
        soft_mask = self.add_conv(cat_fea)
        soft_mask = torch.sigmoid(soft_mask)
        x2 = x2 * soft_mask

        if h == 256:
            param = self.param_adapter(x2)
            res = self.conv1x1_U(x1)
            res = self.adaptive_conv_9(res, param)
            res = self.conv1x1_V(res) 
            res = res + x1
                
        if h == 128:
            param = self.param_adapter(x2)
            res = self.conv1x1_U(x1)
            res = self.adaptive_conv_5(res, param)
            res = self.conv1x1_V(res) 
            res = res + x1

        if h == 64:
            param = self.param_adapter(x2)
            res = self.conv1x1_U(x1)
            res = self.adaptive_conv_3(res, param)
            res = self.conv1x1_V(res) 
            res = res + x1
        return res

class CMAF_Block(nn.Module):
    def __init__(self, in_ch=256, out_ch=8, ksize=1, stride=1):
        super(CMAF_Block, self).__init__()
        pad = (ksize - 1) // 2
 
        self.add_coni = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ksize, stride=stride, padding=pad, bias=False), 
            nn.InstanceNorm2d(out_ch, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.add_conm = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ksize, stride=stride, padding=pad, bias=False), 
            nn.InstanceNorm2d(out_ch, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.weight_levels = nn.Conv2d(out_ch*2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, cre_context, mask):

        input = cre_context * mask
        minput = cre_context * (1-mask)
        level_weight_v0 = self.add_coni(input)
        level_weight_v1 = self.add_conm(minput)
        level_weigths = torch.cat((level_weight_v0, level_weight_v1), dim=1)
        levels_weight = self.weight_levels(level_weigths)
        levels_weight = torch.softmax(levels_weight, dim=1)

        fused_out = input * levels_weight[:,0:1,:,:] + minput * levels_weight[:,1:2,:,:]
        
        return fused_out


class MaskResBlock(nn.Module):
    def __init__(self, fin, fout, kernel_size=3, stride=1, padding=0, dilation=1):
        super(MaskResBlock, self).__init__()
        self.rule = nn.ELU()
        self.conv0 = nn.Conv2d(fin, fout, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv1 = nn.Conv2d(fout, fout, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(fin, fout, 1, stride=stride, dilation=1)
        
    def forward(self, x):
        x_b = self.conv0(x)
        x_b = self.rule(x_b)
        x_b = self.conv1(x_b)
        x = self.conv2(x)
        x = x + x_b
        x = self.rule(x)
        return x        
    
class EstimatingResBlock(nn.Module):
    def __init__(self):
        super(EstimatingResBlock, self).__init__()
        
        dim = 256
        self.block1 = MaskResBlock(fin=dim, fout=dim, kernel_size=3, stride=1, padding=1)
        self.block2 = MaskResBlock(fin=dim, fout=dim, kernel_size=3, stride=1, padding=1)
        self.block3 = MaskResBlock(fin=dim, fout=dim, kernel_size=3, stride=1, padding=2, dilation=2)
        self.block4 = MaskResBlock(fin=dim, fout=dim, kernel_size=3, stride=1, padding=4, dilation=4)
        self.block5 = MaskResBlock(fin=dim, fout=dim, kernel_size=3, stride=1, padding=6, dilation=6)
        self.block6 = MaskResBlock(fin=dim, fout=dim, kernel_size=3, stride=1, padding=8, dilation=8)
        self.block7 = MaskResBlock(fin=dim, fout=dim, kernel_size=3, stride=1, padding=1)
        self.block8 = MaskResBlock(fin=dim, fout=dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        out = self.block8(block7)  
        return out       

class InpaintingResBlock(nn.Module):
    def __init__(self):
        super(InpaintingResBlock, self).__init__()
        
        self.block1 = ResnetBlock(dim=256, dilation=1)
        self.block2 = ResnetBlock(dim=256, dilation=1)
        self.block3 = ResnetBlock(dim=256, dilation=2)
        self.block4 = ResnetBlock(dim=256, dilation=4)
        self.block5 = ResnetBlock(dim=256, dilation=6)
        self.block6 = ResnetBlock(dim=256, dilation=8)
        self.block7 = ResnetBlock(dim=256, dilation=1)
        self.block8 = ResnetBlock(dim=256, dilation=1)
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        out = self.block8(block7)  
        return out

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False, use_dropout=False):
        super(ResnetBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
         ]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
            
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class AdaptiveConv(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(AdaptiveConv, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)

    def forward(self, input, dynamic_weight):
        batch_num = input.size(0)
        input = input.view(1, -1, input.size(2), input.size(3))

        dynamic_weight = dynamic_weight.view(-1, 1, dynamic_weight.size(2), dynamic_weight.size(3))

        conv_rlt = F.conv2d(input, dynamic_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        conv_rlt = conv_rlt.view(batch_num, -1, conv_rlt.size(2), conv_rlt.size(3))

        return conv_rlt

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class DenseD(nn.Module):
    def __init__(self, growth_rate=32, block_config=(3, 3, 3),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseD, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # output layer
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = torch.sigmoid(out)
        return out

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self):
        super().__init__()
        
        self.num_d = 2

        self.d_1 = NLayerDiscriminator()
        self.d_2 = NLayerDiscriminator()

        self.init_weights()

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self, input):
        get_intermediate_features = False

        out_1 = self.d_1(input)
        input = self.downsample(input)
        out_2 = self.d_2(input)

        return [out_1, out_2]

class NLayerDiscriminator(BaseNetwork):

    def __init__(self, in_channels=3):
        super().__init__()
        self.n_layers_D = 4

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = 64
        input_nc = in_channels

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, self.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n ==  self.n_layers_D - 1 else 2
            sequence += [[spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        get_intermediate_features = False
        if get_intermediate_features:
            return results[1:]
        else:
            return torch.sigmoid(results[-1])
        

