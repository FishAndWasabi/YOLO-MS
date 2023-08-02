import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..utils import autopad
from mmyolo.registry import MODELS
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x 


class ELANv9(nn.Module):
    def __init__(self, 
                 in_channel, 
                 mid_channel, 
                 out_channel,
                 bottleneck_ratio=3,
                 conv_cfg=None, 
                 act_cfg=dict(type='LeakyReLU'),
                 norm_cfg=dict(type='BN'),
                 k_sizes = [7,7]):
        super().__init__()
        
        r_mid_channel = mid_channel*bottleneck_ratio

        self.l = ConvModule(in_channel, 
                            mid_channel,
                            1, 
                            conv_cfg=conv_cfg, 
                            act_cfg=act_cfg, 
                            norm_cfg=norm_cfg)
        
        self.r_in = ConvModule(in_channel, 
                               r_mid_channel, 
                               1, 
                               conv_cfg=conv_cfg,
                               act_cfg=act_cfg, 
                               norm_cfg=norm_cfg)
        
        self.r_outs = [ConvModule(r_mid_channel,
                                  mid_channel, 
                                  1,
                                  conv_cfg=conv_cfg, 
                                  act_cfg=act_cfg, 
                                  norm_cfg=norm_cfg)]
        self.r_convs = []
        
        for i, k_size in enumerate(k_sizes):
            if i != len(k_sizes)-1:
                act_cfg_ = None
                norm_cfg_ = None
            else:
                act_cfg_ = act_cfg
                norm_cfg_ = norm_cfg
            r_conv = ConvModule(r_mid_channel, 
                                r_mid_channel, 
                                k_size, 
                                padding=autopad(k_size),
                                groups=r_mid_channel,
                                conv_cfg=conv_cfg, 
                                act_cfg=act_cfg_, 
                                norm_cfg=norm_cfg_)
            r_out = ConvModule(r_mid_channel, 
                               mid_channel, 
                               1, 
                               conv_cfg=conv_cfg, 
                               act_cfg=act_cfg,
                               norm_cfg=norm_cfg)
            self.r_convs.append(r_conv)
            self.r_outs.append(r_out)
        
        self.r_convs = nn.ModuleList(self.r_convs)
        self.r_outs = nn.ModuleList(self.r_outs)
        
        mid_channel = mid_channel*(2+len(k_sizes))
        self.out = ConvModule(mid_channel,
                              out_channel,
                              1,
                              conv_cfg=conv_cfg,
                              act_cfg=act_cfg,
                              norm_cfg=norm_cfg)
        
    def forward(self, x):
        left = self.l(x)
        rights = [self.r_in(x)]
        for i, r_conv in enumerate(self.r_convs):
            rights.append(r_conv(rights[i]))
        rights = [r_out(rights[i]) for i,r_out in enumerate(self.r_outs)]
        out = torch.cat([left] + rights, dim=1)
        out = self.out(out)
        return out
    
class SELAN(nn.Module):
    def __init__(self,
                 in_channel,
                 mid_channel,
                 out_channel,
                 bottleneck_ratio=3,
                 conv_cfg=None, 
                 act_cfg=dict(type='LeakyReLU'),
                 norm_cfg=dict(type='BN'),
                 k_sizes = [3,3,7]
                 ) -> None:
        super().__init__()
        
        self.convs = []
        for k_size in k_sizes:
            mid_channel = int(in_channel*bottleneck_ratio)
            conv_in = ConvModule(in_channel,
                                 mid_channel,
                                 1,
                                 conv_cfg=conv_cfg,
                                 act_cfg=act_cfg,
                                 norm_cfg=norm_cfg)
            conv_mid = ConvModule(mid_channel,
                                  mid_channel,
                                  k_size,
                                  padding=autopad(k_size),
                                  groups=mid_channel,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)
            conv_out = ConvModule(mid_channel,
                                  in_channel,
                                  1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)
            self.convs.append(nn.Sequential(conv_in, conv_mid, conv_out))
        
        self.convs = nn.ModuleList(self.convs)
        self.out = ConvModule(in_channel,
                              out_channel,
                              1,
                              conv_cfg=conv_cfg,
                              act_cfg=act_cfg,
                              norm_cfg=norm_cfg)
        
    def forward(self,x):
        out = x
        for i, conv in enumerate(self.convs):
            out = out +  conv(out)
        out = out + x  
        out = self.out(out)
        return out
    
class ELAN(nn.Module):
    def __init__(self, 
                 in_channel, 
                 mid_channel, 
                 out_channel,
                 bottleneck_ratio=3,
                 conv_cfg=None, 
                 act_cfg=dict(type='LeakyReLU'),
                 norm_cfg=dict(type='BN'),
                 k_sizes = [7,7]):
        super().__init__()
        self.l_in = ConvModule(in_channel,
                               mid_channel,
                               1, 
                               conv_cfg=conv_cfg, 
                               act_cfg=act_cfg, 
                               norm_cfg=norm_cfg )
        self.r_in = ConvModule(in_channel,
                               mid_channel,
                               1, 
                               conv_cfg=conv_cfg, 
                               act_cfg=act_cfg, 
                               norm_cfg=norm_cfg )
        self.r_1 = ConvModule(mid_channel,
                              mid_channel,
                              3,
                              padding=autopad(3), 
                              conv_cfg=conv_cfg, 
                              act_cfg=act_cfg, 
                              norm_cfg=norm_cfg )
        self.r_2 = ConvModule(mid_channel,
                              mid_channel,
                              3,
                              padding=autopad(3), 
                              conv_cfg=conv_cfg, 
                              act_cfg=act_cfg, 
                              norm_cfg=norm_cfg )
        self.out = ConvModule(mid_channel*4,
                               out_channel,
                               1, 
                               conv_cfg=conv_cfg, 
                               act_cfg=act_cfg, 
                               norm_cfg=norm_cfg )
    def forward(self,x):
        l = self.l_in(x)
        r = self.r_in(x)
        r1 = self.r_1(r)
        r2 = self.r_2(r1)
        out = torch.concat([l,r,r1,r2], dim=1)
        out = self.out(out)
        return out

class E2LAN(nn.Module):
    def __init__(self, 
                 in_channel,
                 out_channel,
                 expand_ratio=2,
                 down_ratio = 1,
                 first_expand_ratio=1,
                 attention_cfg=None,
                 mid_attention_cfg=None,
                 back_attention_cfg=None,
                 kernel_sizes=[1,5,(3,15),(15,3)],
                 conv_cfg=None, 
                 act_cfg=dict(type='SiLU'),
                 norm_cfg=dict(type='BN'),
                 use_ln = False,
                 ) -> None:
        super().__init__()
        self.layer_norm = None
        if use_ln:
            self.layer_norm = LayerNorm(in_channel, data_format="channels_first")
        self.attention = None
        if attention_cfg is not None:
            attention_cfg["dim"] = in_channel
            self.attention = MODELS.build(attention_cfg)
        self.back_attention = None
        if back_attention_cfg is not None:
            back_attention_cfg["dim"] = out_channel
            self.back_attention = MODELS.build(back_attention_cfg)
        self.in_channel = int(in_channel*first_expand_ratio)//down_ratio
        self.mid_channel = self.in_channel//len(kernel_sizes)
        self.expand_ratio = expand_ratio
        
        self.in_conv = ConvModule(in_channel,
                                  self.in_channel,
                                  1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)
        
        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            groups = int(self.mid_channel*self.expand_ratio)
            self.mid_proj_in = ConvModule(self.mid_channel,
                                        groups,
                                        1,
                                        conv_cfg=conv_cfg,
                                        act_cfg=act_cfg, 
                                        norm_cfg=norm_cfg)        
            self.mid_conv = ConvModule(groups,
                                       groups,
                                       kernel_size,
                                       padding=autopad(kernel_size),
                                       groups=groups,
                                       conv_cfg=conv_cfg,
                                       act_cfg=act_cfg,
                                       norm_cfg=norm_cfg)
            mid_attention = nn.Identity()
            if mid_attention_cfg is not None:
                mid_attention_cfg["dim"] = groups
                mid_attention = MODELS.build(mid_attention_cfg)
            self.mid_proj_out = ConvModule(groups,
                                       self.mid_channel,
                                       1,
                                       conv_cfg=conv_cfg,
                                       act_cfg=act_cfg, 
                                       norm_cfg=norm_cfg)
            self.mid_convs.append(nn.Sequential(self.mid_proj_in ,
                                                self.mid_conv,
                                                mid_attention, 
                                                self.mid_proj_out))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = ConvModule(self.in_channel,
                                   out_channel,
                                   1,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)
    
    def forward(self, x):
        out = x
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        if self.attention is not None:
            out = self.attention(out)  
        out = self.in_conv(out)
        channels = []
        for i,mid_conv in enumerate(self.mid_convs):
            channel = out[:,i*self.mid_channel:(i+1)*self.mid_channel,...]
            if i >= 1:
                channel = channel + channels[i-1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.back_attention is not None:
            out = self.back_attention(out)  
        return out
        
            
class E2LAN_SK(nn.Module):
    def __init__(self, 
                 in_channel,
                 out_channel,
                 expand_ratio=2,
                 down_ratio = 1,
                 first_expand_ratio=1,
                 attention_cfg=None,
                 mid_attention_cfg=None,
                 back_attention_cfg=None,
                 kernel_sizes=[1,5,(3,15),(15,3)],
                 conv_cfg=None, 
                 act_cfg=dict(type='SiLU'),
                 norm_cfg=dict(type='BN'),
                 use_ln = False,
                 ) -> None:
        super().__init__()
        self.layer_norm = None
        if use_ln:
            self.layer_norm = LayerNorm(in_channel, data_format="channels_first")
        self.attention = None
        if attention_cfg is not None:
            attention_cfg["dim"] = in_channel
            self.attention = MODELS.build(attention_cfg)
        self.in_channel = int(in_channel*first_expand_ratio)//down_ratio
        self.mid_channel = self.in_channel//len(kernel_sizes)
        self.expand_ratio = expand_ratio
        
        self.in_conv = ConvModule(in_channel,
                                  self.in_channel,
                                  1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)
        
        self.mid_convs = []
        
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            groups = int(self.mid_channel*self.expand_ratio)
            self.mid_proj_in = ConvModule(self.mid_channel,
                                        groups,
                                        1,
                                        conv_cfg=conv_cfg,
                                        act_cfg=act_cfg, 
                                        norm_cfg=norm_cfg)        
            self.mid_conv = ConvModule(groups,
                                       groups,
                                       kernel_size,
                                       padding=autopad(kernel_size),
                                       groups=groups,
                                       conv_cfg=conv_cfg,
                                       act_cfg=act_cfg,
                                       norm_cfg=norm_cfg)
            self.mid_proj_out = ConvModule(groups,
                                       self.mid_channel,
                                       1,
                                       conv_cfg=conv_cfg,
                                       act_cfg=act_cfg, 
                                       norm_cfg=norm_cfg)
            self.mid_convs.append(nn.Sequential(self.mid_proj_in ,
                                                self.mid_conv,
                                                self.mid_proj_out))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.channel_sum_downsample = ConvModule(self.mid_channel,
                                                 len(kernel_sizes),
                                                 1,
                                                 conv_cfg=conv_cfg)
        self.out_conv = ConvModule(self.in_channel,
                                   out_channel,
                                   1,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)
    
    def forward(self, x):
        out = x
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        if self.attention is not None:
            out = self.attention(out)  
        out = self.in_conv(out)
        channels = []
        channel_sum = 0
        for i,mid_conv in enumerate(self.mid_convs):
            channel = out[:,i*self.mid_channel:(i+1)*self.mid_channel,...]
            if i >= 1:
                channel = channel + channels[i-1]
            channel = mid_conv(channel)
            channel_sum += channel
            channels.append(channel)
        channel_sum = self.channel_sum_downsample(channel_sum)
        N,C,W,H = channel_sum.shape
        channel_sum = channel_sum.permute(0,2,3,1).reshape(-1,C)
        channel_sum = torch.softmax(channel_sum, -1).reshape(N,W,H,C).permute(0,3,1,2)
        weights = torch.chunk(channel_sum,C,dim=1)
        for i,(weight,channel) in enumerate(zip(weights,channels)):
            channels[i] = weight*channel
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        return out       

if __name__ == "__main__":
    test = torch.randn([3,64,100,100])
    elan = E2LAN(64, 128)
    print(elan(test).shape)