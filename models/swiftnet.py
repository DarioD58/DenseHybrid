"""
# Code adapted from:
# https://github.com/orsic/swiftnet
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from math import log2
from itertools import chain
import torch.utils.model_zoo as model_zoo


upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)
checkpoint = lambda func, *inputs: cp.checkpoint(func, *inputs, preserve_rng_state=True)

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth"
}

def _checkpoint_unit(bn1, relu1, conv1, bn2, relu2, conv2):
    def func(*x):
        x = torch.cat(x, 1)
        x = conv1(relu1(bn1(x)))
        return conv2(relu2(bn2(x)))
    return func

def _checkpoint_transition(norm, relu, conv, pool):
    def func(*x):
        x = torch.cat(x, 1)
        x = norm(x)
        x = conv(relu(x))
        return pool(x)
    return func

def _checkpoint_bnreluconv(bn, relu, conv):
    def func(*x):
        x = torch.cat(x, 1)
        x = bn(x)
        return conv(relu(x))
    return func

def conv3x3(in_planes, out_planes, stride=1, separable=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x

    return bn_function



def do_efficient_fwd(block, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, deleting=False,
                 separable=False):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, separable=separable)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, separable=separable)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.deleting = deleting

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.deleting is False:
            bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
            bn_2 = _bn_function_factory(self.conv2, self.bn2)

            out = do_efficient_fwd(bn_1, x, self.efficient)
            out = do_efficient_fwd(bn_2, out, self.efficient)
        else:
            out = torch.zeros_like(residual)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.005,
                 bias=False, dilation=1, checkpointing=False):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))

        self.checkpointing = checkpointing
        if checkpointing:
            self.conv_func = _checkpoint_bnreluconv(self.norm, self.relu, self.conv)


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True, checkpointing=False):
        super(_Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn and bneck_starts_with_bn, checkpointing=checkpointing)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn, checkpointing=checkpointing)
        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x



class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels=3, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True, checkpointing=False):
        super(SpatialPyramidPooling, self).__init__()
        self.fixed_size = fixed_size
        self.grids = grids
        if self.fixed_size:
            ref = min(self.fixed_size)
            self.grids = list(filter(lambda x: x <= ref, self.grids))
        self.square_grid = square_grid
        self.upsampling_method = upsample
        if self.fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum,
                                                  batch_norm=use_bn and starts_with_bn, checkpointing=checkpointing))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn, checkpointing=checkpointing))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn, checkpointing=checkpointing))

    def forward(self, x):
        levels = []
        target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = self.upsampling_method(level, target_size)
            levels.append(level)

        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x



class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=False, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, spp_drop_rate=0.0,
                 upsample_skip=True, upsample_only_skip=False,
                 detach_upsample_skips=(), detach_upsample_in=False,
                 target_size=None, output_stride=4, mean=(73.1584, 82.9090, 72.3924),
                 std=(44.9149, 46.1529, 45.3192), scale=1, separable=False,
                 upsample_separable=False, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn
        self.separable = separable

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.num_features = 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn,
                        separable=self.separable)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn,
                             separable=self.separable)]

        return nn.Sequential(*layers)


    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, out = self.forward_resblock(x, self.layer4)
        return out, features


class SPPWrapper(nn.Module):
    def __init__(self, num_features, checkpointing=False):
        super(SPPWrapper, self).__init__()

        self.spp_size = 128
        spp_square_grid = False
        spp_grids = [8, 4, 2, 1]
        num_levels = 3
        level_size = self.spp_size // num_levels
        bt_size = self.spp_size
        self.spp = SpatialPyramidPooling(num_features, num_levels, bt_size, level_size,
                                         self.spp_size, spp_grids, spp_square_grid, checkpointing=checkpointing)
        self.num_features = self.spp_size

    def forward(self, x):
        return self.spp(x)


class UpsampleWrapper(nn.Module):
    def __init__(self, num_features, skip_sizes, checkpointing=False, up_sizes=[128, 128, 128]):
        super(UpsampleWrapper, self).__init__()

        self.upsample_layers = nn.Sequential()

        assert len(up_sizes) == len(skip_sizes)
        for i in range(len(skip_sizes)):
            upsample = _Upsample(num_features, skip_sizes[-1 - i], up_sizes[i], checkpointing=checkpointing)
            num_features = up_sizes[i]
            self.upsample_layers.add_module('upsample_' + str(i), upsample)

        self.num_features = num_features

    def forward(self, x, skip_layers):
        for i, skip in enumerate(reversed(skip_layers)):
            x, _ = self.upsample_layers[i].forward(x, skip)
        return x


class SwiftNet(nn.Module):
    def __init__(self, num_classes=19, checkpointing=False):
        super(SwiftNet, self).__init__()

        self.num_classes = num_classes
        block_conf = [2, 2, 2, 2]
        self.backbone = ResNet(BasicBlock, block_conf)
        self.spp = SPPWrapper(512)
        self.upsample = UpsampleWrapper(128, [64, 128, 256])
        self.logits = _BNReluConv(128, self.num_classes, k=1, bias=True)


    def load_imagenet(self):

        self.backbone.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    
    def forward(self, x, target_size=None):
        if target_size == None:
            target_size = x.size()[2:4]

        x, skip_layers = self.backbone(x)
        x = self.spp(x)
        x = self.upsample(x, skip_layers)
        x = self.logits(x)
        x = F.upsample(x, target_size, mode='bilinear', align_corners=False)

        return x




if __name__ == '__main__':
    model = SwiftNet()
    print(model)

    def parameter_count(module):
        trainable, non_trainable = 0, 0
        for p in module.parameters():
            if p.requires_grad:
                trainable += p.numel()
            else:
                non_trainable += p.numel()
        return trainable, non_trainable

    print('SwiftNet', parameter_count(model))

