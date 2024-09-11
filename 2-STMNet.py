import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Siam(nn.Module):
    """docstring for Siam"""
    def __init__(self, ):
        super(Siam, self).__init__()
        self.conv3=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.conv4=nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv5=nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)

        self.up2     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up4     = nn.UpsamplingBilinear2d(scale_factor = 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.linear=nn.Sequential(nn.Linear(768,128),nn.Linear(128,2))
    def forward(self, feat3,feat4,feat5):
        feat3=self.conv3(feat3)
        feat4=self.conv4(feat4)
        feat5=self.conv5(feat5)
        feat4=self.up2(feat4)
        feat5=self.up4(feat5)

        feat=torch.cat((feat3,feat4,feat5),1)

        avgout = self.avg_pool(feat)
        # print(avgout.shape)
        maxout = self.max_pool(feat)

        out=torch.cat((avgout,maxout),1)
        out=out.view(out.shape[0],-1)
        # print(out.shape)
        out=self.linear(out)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ASPP(nn.Module):
    def __init__(self,):
        super(ASPP, self).__init__()
        inplanes = 512
        dilations = [1, 3, 5, 7]
        self.aspp1 = _ASPPModule(inplanes, 128, 3, padding=dilations[0], dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 128, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 128, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 128, 3, padding=dilations[3], dilation=dilations[3])
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4, x), dim=1)
        x=self.conv1(x)
        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU()
    def forward(self, input1):
        output = self.bn1(self.conv1(input1))
        output = self.relu(output)
        output = self.up(output)
        return output
class STMNet(nn.Module):
    def __init__(self, pretrained = True, backbone = 'resnet50'):
        super(STMNet, self).__init__()
        # self.preconv=nn.Conv2d(9, 3, kernel_size = 1, padding = 0)
        self.backbone1 = timm.create_model("resnet18", features_only=True,
                             out_indices=(0,1, 2, 3, 4), pretrained=True)

        self.backbone2 = timm.create_model("resnet18", features_only=True,
                             out_indices=(0,1, 2, 3, 4), pretrained=True)

        self.CAM1=ChannelAttentionModule(64)
        self.CAM2=ChannelAttentionModule(64)
        self.CAM3=ChannelAttentionModule(128)
        self.CAM4=ChannelAttentionModule(256)
        self.CAM5=ChannelAttentionModule(512)

        self.SAM1=SpatialAttentionModule()
        self.SAM2=SpatialAttentionModule()
        self.SAM3=SpatialAttentionModule()
        self.SAM4=SpatialAttentionModule()
        self.SAM5=SpatialAttentionModule()

        filters  =   [64 , 64, 128, 256, 512]
        infilters  = [128, 192,384, 512, 512]
        outfilters = [64 , 64,128, 256, 256]
        self.aspp=ASPP()
        self.unetUp5=unetUp(infilters[4],outfilters[4])
        self.unetUp4=unetUp(infilters[3],outfilters[3])
        self.unetUp3=unetUp(infilters[2],outfilters[2])
        self.unetUp2=unetUp(infilters[1],outfilters[1])
        self.unetUp1=unetUp(infilters[0],outfilters[0])
        self.cf = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0))


        self.scene=Siam()

    def forward(self, input1,input2):
        in1_feat1,in1_feat2, in1_feat3, in1_feat4,in1_feat5 = self.backbone1(input1)
        in2_feat1,in2_feat2, in2_feat3, in2_feat4,in2_feat5 = self.backbone2(input2)

        print(in1_feat1.shape,in1_feat2.shape, in1_feat3.shape, in1_feat4.shape,in1_feat5.shape)

        CAM1=self.CAM1(in2_feat1)
        CAM2=self.CAM2(in2_feat2)
        CAM3=self.CAM3(in2_feat3)
        CAM4=self.CAM4(in2_feat4)
        CAM5=self.CAM5(in2_feat5)

        SAM1=self.SAM1(in2_feat1)
        SAM2=self.SAM2(in2_feat2)
        SAM3=self.SAM3(in2_feat3)
        SAM4=self.SAM4(in2_feat4)
        SAM5=self.SAM5(in2_feat5)

        in_feat1=in1_feat1*CAM1*SAM1
        in_feat2=in1_feat2*CAM2*SAM2
        in_feat3=in1_feat3*CAM3*SAM3
        in_feat4=in1_feat4*CAM4*SAM4
        in_feat5=in1_feat5*CAM5*SAM5

        de_feat5 = self.unetUp5(in_feat5)
        temp = torch.cat((in_feat4, de_feat5), dim=1)
        de_feat4 = self.unetUp4(temp)
        temp = torch.cat((in_feat3, de_feat4), dim=1)
        de_feat3 = self.unetUp3(temp)
        temp = torch.cat((in_feat2, de_feat3), dim=1)
        de_feat2 = self.unetUp2(temp)
        temp = torch.cat((in_feat1, de_feat2), dim=1)
        output=self.cf(temp)
        output = F.interpolate(output, size=(256,256), mode='bilinear', align_corners=True)

        print("output",in_feat3.shape,in_feat4.shape,in_feat5.shape)

        output_scene1=self.scene(in_feat3,in_feat4,in_feat5)
        return output,output_scene1

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total:',total_num, 'Trainable:',trainable_num)
    return {'Total': total_num, 'Trainable': trainable_num}
def getModelSize(model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        buffer_size = 0
        buffer_sum = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        all_size = (param_size + buffer_size) / 1024 / 1024
        print('模型总大小为：{:.3f}MB'.format(all_size))
        return (param_size, param_sum, buffer_size, buffer_sum, all_size)
if __name__ == '__main__':
    model=STMNet()
    model.eval()
    input1=torch.randn((3,3,512,512))
    input2=torch.randn((3,3,512,512))
    output,output_scene1=model(input1,input2)
    getModelSize(model)

    print(output.shape)

    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(input1,input2,))
    flops, params = clever_format([flops, params], '%.3f')
    # print('模型参数：', params)
    print('每一个样本浮点运算量：', flops)
