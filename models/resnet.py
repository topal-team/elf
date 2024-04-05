import torch
import torch.nn as nn
from copy import deepcopy

# import rockmate
# import rkgb

NUM_CLASSES = 1000

__all__ = ['ResNet', 'SequentialResNet50', 'CheckpointablePipelineParallelResNet50']

class MyNorm(nn.Module):
    def __init__(self, planes):
        super(MyNorm, self).__init__()
        
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


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
                 base_width=64, dilation=1, norm_layer=None, shared_weight=None):
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
        self.downsample = downsample
        self.stride = stride

        if shared_weight is not None:
            del self.conv2.weight
            # del self.conv1.weight

            self.conv2.weight = shared_weight
            # self.conv2.weight.requires_grad = True
            # self.conv1.weight = shared_weight

        print(type(self.conv2.weight))

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

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = MyNorm
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=NUM_CLASSES, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = MyNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class SequentialResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(SequentialResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=NUM_CLASSES, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        )

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        )

        self.head = nn.Sequential(
            Flatten(),
            self.fc
        )

    def forward(self, x):
        return self.head(self.seq2(self.seq1(x)))


class CheckpointablePipelineParallelResNet50(ResNet):
    def __init__(self, split_size=20, budget=1000*1024*1024, batch_size=16, gpus=['cuda:0', 'cuda:1'], *args, **kwargs):
        super(CheckpointablePipelineParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=NUM_CLASSES, *args, **kwargs)
        self.gpus = gpus
        seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        )

        seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        )

        print(seq1[0].weight.grad)

        # list_solver = [rockmate.solvers.HILP()]
        list_solver = [rockmate.solvers.TwRemat()]
        max_size_S_graph_for_no_partitioning = 0
        partitioners = [rkgb.Ptools.Partitioner_seq(sub_partitioner=rkgb.Ptools.Partitioner())]

        
        sample1 = torch.randn(batch_size//split_size, 3, 224, 224)
        is_training = seq1.training
        with torch.no_grad():
            seq1.training = False
            sample2 = seq1(sample1).detach()
            seq1.training = is_training
        

        sample1 = deepcopy(sample1).to(self.gpus[0])
        sample2 = deepcopy(sample2).to(self.gpus[1])

        
        self.rkMod1 = rockmate.HRockmate(
                seq1.to(self.gpus[0]), sample1, max([budget]), 
                list_solvers=list_solver, 
                partitioners=partitioners,
                # solve_sched = False,
                max_size_S_graph_for_no_partitioning=max_size_S_graph_for_no_partitioning
            )
        print(self.rkMod1.op_sched)

        self.rkMod2 = rockmate.HRockmate(
                seq2.to(self.gpus[1]), sample2, max([budget]), 
                list_solvers=list_solver, 
                partitioners=partitioners,
                max_size_S_graph_for_no_partitioning=max_size_S_graph_for_no_partitioning
            )

        self.head = nn.Sequential(Flatten(), self.fc).to(self.gpus[1])

        del sample1, sample2
        del seq1, seq2

        torch.cuda.empty_cache()

        self.rkMod1.solve_sched(budget, rec=False)
        self.rkMod1.get_compiled_fct()

        self.rkMod2.solve_sched(budget, rec=False)
        self.rkMod2.get_compiled_fct()

        self.split_size = split_size

        # for n, p in self.rkMod1.original_mod.named_parameters():
        #     print(p, self.rkMod1.original_mod.get_parameter(n).grad)

    def forward(self, x):

        self.rkMod1.reinit()
        self.rkMod2.reinit()

        print(self.rkMod1.original_mod.get_parameter("4.0.conv1.weight").grad)

        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.rkMod1(s_next).to(self.gpus[1])
        ret = []

        for s_next in splits:
            # A. ``s_prev`` runs on ``cuda:1``
            s_prev = self.rkMod2(s_prev)
            ret.append(self.head(s_prev))

            # B. ``s_next`` runs on ``cuda:0``, which can run concurrently with A
            s_prev = self.rkMod1(s_next).to(self.gpus[1])

        s_prev = self.rkMod2(s_prev)
        ret.append(self.head(s_prev))

        return torch.cat(ret)
