from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable

import lpips
from . import pretrained_networks as pn


def spatial_average(tensors, keepdim=True):
    return torch.mean(tensors, [2, 3], keepdim=keepdim)


def upsample(tensors, output_size=(64, 64)):  # assumes scale factor is same for H and W
    return nn.Upsample(size=output_size, mode="bilinear", align_corners=False)(tensors)


# Learned perceptual metric


class LPIPS(nn.Module):
    def __init__(
        self,
        pretrained=True,
        net="alex",
        version="0.1",
        lpips=True,
        spatial=False,
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        model_path=None,
        eval_mode=True,
        verbose=True,
    ):
        # lpips - [True] means with linear calibration on top of base network
        # pretrained - [True] means load linear weights

        super(LPIPS, self).__init__()
        if verbose:
            print(
                "Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]"
                % (
                    "LPIPS" if lpips else "baseline",
                    net,
                    version,
                    "on" if spatial else "off",
                )
            )

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        # activate channel-wise by vectors
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = pn.VGG
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.channels_len = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if lpips:
            self.lin0 = LinearLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = LinearLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = LinearLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = LinearLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = LinearLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == "squeeze":  # 7 layers for squeezenet
                self.lin5 = LinearLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = LinearLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if pretrained:
                if model_path is None:
                    import inspect
                    import os

                    model_path = os.path.abspath(
                        os.path.join(
                            inspect.getfile(self.__init__),
                            "..",
                            "weights/v%s/%s.pth" % (version, net),
                        )
                    )

                if verbose:
                    print("Loading model from: %s" % model_path)
                self.load_state_dict(
                    torch.load(model_path, map_location="cpu"), strict=False
                )

        if eval_mode:
            self.eval()

    def forward(self, in0, in1, value_and_channel_results=False, normalize=False):
        # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1))
            if self.version == "0.1"
            else (in0, in1)
        )
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.channels_len):
            feats0[kk], feats1[kk] = (
                lpips.normalize_tensor(outs1[kk]),
                lpips.normalize_tensor(outs0[kk]),
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # add linear weight after difference
        if self.lpips:
            if self.spatial:
                channel_results = [
                    upsample(self.lins[kk](diffs[kk]), output_size=in0.shape[2:])
                    for kk in range(self.channels_len)
                ]
            else:
                channel_results = [
                    spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
                    for kk in range(self.channels_len)
                ]
        else:
            if self.spatial:
                channel_results = [
                    upsample(
                        diffs[kk].sum(dim=1, keepdim=True), output_size=in0.shape[2:]
                    )
                    for kk in range(self.channels_len)
                ]
            else:
                channel_results = [
                    spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True)
                    for kk in range(self.channels_len)
                ]

        value = channel_results[0]
        for l in range(1, self.channels_len):
            value += channel_results[l]

        # a = spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
        # b = torch.max(self.lins[kk](feats0[kk]**2))
        # for kk in range(self.L):
        #     a += spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
        #     b = torch.max(b,torch.max(self.lins[kk](feats0[kk]**2)))
        # a = a/self.L
        # from IPython import embed
        # embed()
        # return 10*torch.log10(b/a)

        if value_and_channel_results:
            return value, channel_results
        else:
            return value


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class LinearLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(LinearLayer, self).__init__()

        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [
            nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),
        ]
        layers += [
            nn.LeakyReLU(0.2, True),
        ]
        layers += [
            nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),
        ]
        layers += [
            nn.LeakyReLU(0.2, True),
        ]
        layers += [
            nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),
        ]
        if use_sigmoid:
            layers += [
                nn.Sigmoid(),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(
            torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1)
        )


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.0) / 2.0
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


# L2, DSSIM metrics


class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace="Lab"):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):
    def forward(self, in0, in1):
        assert in0.size()[0] == 1  # currently only supports batchSize 1

        if self.colorspace == "RGB":
            (N, C, X, Y) = in0.size()
            value = torch.mean(
                torch.mean(
                    torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2
                ).view(N, 1, 1, Y),
                dim=3,
            ).view(N)
            return value
        elif self.colorspace == "Lab":
            value = lpips.l2(
                lpips.tensor2np(lpips.tensor2tensorlab(in0.data, to_norm=False)),
                lpips.tensor2np(lpips.tensor2tensorlab(in1.data, to_norm=False)),
                range=100.0,
            ).astype("float")
            ret_var = Variable(torch.Tensor((value,)))
            if self.use_gpu:
                ret_var = ret_var.cuda()
            return ret_var


class DSSIM(FakeNet):
    def forward(self, in0, in1):
        assert in0.size()[0] == 1  # currently only supports batchSize 1

        if self.colorspace == "RGB":
            value = lpips.dssim(
                1.0 * lpips.tensor2im(in0.data),
                1.0 * lpips.tensor2im(in1.data),
                range=255.0,
            ).astype("float")
        elif self.colorspace == "Lab":
            value = lpips.dssim(
                lpips.tensor2np(lpips.tensor2tensorlab(in0.data, to_norm=False)),
                lpips.tensor2np(lpips.tensor2tensorlab(in1.data, to_norm=False)),
                range=100.0,
            ).astype("float")
        ret_var = Variable(torch.Tensor((value,)))
        if self.use_gpu:
            ret_var = ret_var.cuda()
        return ret_var


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("Network", net)
    print("Total number of parameters: %d" % num_params)
