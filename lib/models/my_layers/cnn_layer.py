import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.my_layers.conformer import FCUDown
from collections import OrderedDict
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_


class BlockCNN(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.cnnrgb = nn.Sequential(nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(768),
                                    nn.ReLU())
        self.cnntir = nn.Sequential(nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(768),
                                    nn.ReLU())

    def forward(self, x):
        x1 = self.norm1(x)
        # x2 = self.attn(x1)  # x1.shape: (bs,640,768)
        x1 = x1.transpose(1, 2)
        zv = x1[:, :, :64].view(-1, 768, 8, 8)
        xv = x1[:, :, 64:320].view(-1, 768, 16, 16)
        zi = x1[:, :, 320:384].view(-1, 768, 8, 8)
        xi = x1[:, :, 384:].view(-1, 768, 16, 16)

        zv = self.cnnrgb(zv).flatten(2).transpose(1, 2)
        xv = self.cnnrgb(xv).flatten(2).transpose(1, 2)
        zi = self.cnntir(zi).flatten(2).transpose(1, 2)
        xi = self.cnntir(xi).flatten(2).transpose(1, 2)

        x2 = torch.cat((zv, xv, zi, xi), dim=1)

        x = x + self.drop_path(x2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CTFusion(nn.Module):
    def __init__(self):
        super().__init__()
        feat_dim = 768
        hidden_dim = 256
        self.FF_layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU())),
            ('fc2_C', nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(hidden_dim, feat_dim),
                                    nn.ReLU())),
            ('fc2_T', nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(hidden_dim, feat_dim),
                                    nn.ReLU()))
        ]))

    def forward(self, trans, conv):
        trans_1 = self.FF_layers.fc1(trans)
        w_trans = self.FF_layers.fc2_T(trans_1).unsqueeze(1)

        conv_1 = self.FF_layers.fc1(conv)
        w_conv = self.FF_layers.fc2_C(conv_1).unsqueeze(1)

        w = nn.functional.softmax(torch.cat([w_trans, w_conv], 1), dim=1)

        new_trans = torch.mul(trans, w[:, 0]) + torch.mul(conv, w[:, 1])

        return new_trans


class CNNLayer_Z(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.ModuleList()
        self.cnn_layers.append(nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU(),
                                             nn.Dropout(0.5)))

        self.cnn_layers.append(nn.Sequential(nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=1),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),
                                             nn.Dropout(0.5), ))

        self.cnn_layers.append(nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                             nn.BatchNorm2d(512),
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             # LRN()
                                             ))

        self.to_token_x = nn.ModuleList()
        self.to_token_x.append(FCUDown(96, 768, 8))
        self.to_token_x.append(FCUDown(256, 768, 4))
        self.to_token_x.append(FCUDown(512, 768, 2))

    def forward(self, x, layer_index):
        x = self.cnn_layers[layer_index](x)
        x_token = self.to_token[layer_index](x)
        return x, x_token


class CnovTrans(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        med_chans = in_chans // 4
        self.cnn_3layers = nn.Sequential(nn.Conv2d(in_chans, med_chans, kernel_size=1, stride=1, padding=0, bias=False),
                                         nn.BatchNorm2d(med_chans),
                                         nn.ReLU(),

                                         nn.Conv2d(med_chans, med_chans, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(med_chans),
                                         nn.ReLU(),

                                         nn.Conv2d(med_chans, out_chans, kernel_size=1, stride=1, padding=0, bias=False),
                                         nn.BatchNorm2d(out_chans),
                                         nn.ReLU())

    def forward(self, x):
        x = self.cnn_3layers(x)
        return x


class CNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.ModuleList()
        self.cnn_layers2 = nn.ModuleList()
        self.cnn_layers.append(nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU(),
                                             nn.Dropout(0.5)))

        self.cnn_layers.append(nn.Sequential(nn.Conv2d(96, 256, kernel_size=1, stride=2, padding=0),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),
                                             nn.Dropout(0.5)))

        self.cnn_layers.append(nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0),
                                             nn.BatchNorm2d(512),
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             # LRN()
                                             ))

        self.cnovtrans = nn.ModuleList()
        self.cnovtrans.append(CnovTrans(in_chans=96, out_chans=96))
        self.cnovtrans.append(CnovTrans(256, 256))
        self.cnovtrans.append(CnovTrans(512, 512))

        self.to_token = nn.ModuleList()
        self.to_token.append(FCUDown(96, 768, 8))
        self.to_token.append(FCUDown(256, 768, 4))
        self.to_token.append(FCUDown(512, 768, 2))

    def forward(self, x, index):
        x = self.cnn_layers[index](x)
        x_token = self.cnovtrans[index](x)
        x_token = self.to_token[index](x_token)

        # for i in range(3):
        #     x = self.cnn_layers[i](x)
        #
        # x_token = self.cnovtrans[2](x)
        # x_token = self.to_token[2](x_token)

        return x, x_token
