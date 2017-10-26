import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F


class seLayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(seLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = y.repeat(1, 1, w, h)
        return x * y


class conv2DBatchNorm(nn.Module):

    def __init__(self,
                 in_channels,
                 n_filters,
                 k_size,
                 stride,
                 padding,
                 bias=True):
        super(conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias),
            nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):

    def __init__(self,
                 in_channels,
                 n_filters,
                 k_size,
                 stride,
                 padding,
                 bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias),
            nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):

    def __init__(self,
                 in_channels,
                 n_filters,
                 k_size,
                 stride,
                 padding,
                 bias=True):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias),
            nn.BatchNorm2d(int(n_filters)),
            nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):

    def __init__(self,
                 in_channels,
                 n_filters,
                 k_size,
                 stride,
                 padding,
                 output_padding=0,
                 bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                output_padding=output_padding,
                stride=stride,
                bias=bias),
            nn.BatchNorm2d(int(n_filters)),
            nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class unetConv2(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 0),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),)
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 0),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 0),
                nn.ReLU(),)
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 0),
                nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):

    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(
                in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class segnetDown2(nn.Module):

    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):

    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):

    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(
            input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):

    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(
            input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock, self).__init__()

        self.convbnrelu1 = conv2DBatchNormRelu(
            in_channels, n_filters, 3, stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(
            n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class residualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBottleneck, self).__init__()
        self.convbn1 = nn.Conv2DBatchNorm(
            in_channels, n_filters, k_size=1, bias=False)
        self.convbn2 = nn.Conv2DBatchNorm(
            n_filters,
            n_filters,
            k_size=3,
            padding=1,
            stride=stride,
            bias=False)
        self.convbn3 = nn.Conv2DBatchNorm(
            n_filters, n_filters * 4, k_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.convbn1(x)
        out = self.convbn2(out)
        out = self.convbn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class linknetUp(nn.Module):

    def __init__(self, in_channels, n_filters, stride=1, output_padding=0):
        super(linknetUp, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W
        self.convbnrelu1 = conv2DBatchNormRelu(
            in_channels, n_filters / 2, k_size=1, stride=1, padding=0)

        # B, C/2, H, W -> B, C/2, H, W
        self.deconvbnrelu2 = deconv2DBatchNormRelu(
            n_filters / 2,
            n_filters / 2,
            k_size=3,
            stride=stride,
            padding=1,
            output_padding=output_padding)

        # B, C/2, H, W -> B, C, H, W
        self.convbnrelu3 = conv2DBatchNormRelu(
            n_filters / 2, n_filters, k_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.deconvbnrelu2(x)
        x = self.convbnrelu3(x)
        return x


def save_checkpoint(state,
                    is_best,
                    checkpoint='checkpoint',
                    filename='checkpoint.pth.tar',
                    snapshot=None):
    # preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # scipy.io.savemat(os.path.join(checkpoint, 'preds.mat'), mdict={'preds' : preds})

    if snapshot and state.epoch % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(checkpoint,
                         'checkpoint_{}.pth.tar'.format(state.epoch)))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
        # scipy.io.savemat(os.path.join(checkpoint, 'preds_best.mat'), mdict={'preds' : preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr
