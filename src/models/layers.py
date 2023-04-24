# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layers import MLP, Res1dBlock
from torch.nn import (
    AvgPool1d,
    AvgPool2d,
    BatchNorm1d,
    Conv1d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool1d,
    MaxPool2d,
    ReLU,
    Sequential,
    Flatten,
)


def calc_conv_out(w, k, p, s):
    return np.floor(((w - k + 2 * p) / s) + 1)


class RNNSegmenter(torch.nn.Module):
    def __init__(self, window_size=128):
        return


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        in_dim=128,
        n_heads=8,
        n_transformer_layers=4,
        n_convs=4,
        L=351,
        info_dim=12,
        global_dim=37,
    ):
        super(TransformerClassifier, self).__init__()

        self.info_embedding = nn.Sequential(
            nn.Linear(info_dim, 16), nn.LayerNorm((16,))
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim + 16, nhead=n_heads, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        self.convs = nn.ModuleList()
        for ix in range(n_convs):
            self.convs.append(Res1dBlock(in_dim + 16, in_dim + 16, 2, pooling="max"))

            L = L // 2

        self.global_embedding = nn.Sequential(
            nn.Linear(global_dim, 32), nn.LayerNorm((32,))
        )
        self.mlp = nn.Sequential(
            MLP((in_dim + 16) * L + 32, 1024, 2048, dropout=0.05, norm=nn.LayerNorm)
        )

        self.final = nn.Sequential(nn.Linear(1024, 5, bias=False), nn.LogSoftmax())

    def forward(self, x, x1, x2):
        bs, l, c = x1.shape

        x1 = self.info_embedding(x1.flatten(0, 1)).reshape(bs, l, -1)
        bs, l, c = x.shape

        x = torch.cat([x, x1], dim=-1)

        x = self.transformer(x).transpose(1, 2)

        for ix in range(len(self.convs)):
            x = self.convs[ix](x)

        x = x.flatten(1, 2)
        x2 = self.global_embedding(x2)

        x = torch.cat([x, x2], dim=-1)
        f = self.mlp(x)

        x = self.final(f)

        return x, f


class TransformerRNNClassifier(nn.Module):
    def __init__(
        self,
        in_dim=128,
        n_heads=8,
        n_transformer_layers=4,
        n_convs=4,
        L=351,
        info_dim=12,
        global_dim=37,
    ):
        super(TransformerRNNClassifier, self).__init__()

        self.embedding = MLP(in_dim, in_dim, in_dim, norm=nn.LayerNorm)

        self.info_embedding = nn.Sequential(
            nn.Linear(info_dim, 16), nn.LayerNorm((16,))
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim + 16, nhead=n_heads, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )
        self.gru = nn.GRU(in_dim + 16, 1024, batch_first=True)

        self.global_embedding = nn.Sequential(
            nn.Linear(global_dim, 32), nn.LayerNorm((32,))
        )

        self.mlp = nn.Sequential(MLP(1024 + 32, 1024, 1024, norm=nn.LayerNorm))

        self.final = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm((512,)),
            nn.ReLU(),
            nn.Linear(512, 5),
            nn.LogSoftmax(),
        )

    def forward(self, x, x1, x2):
        bs, l, c = x1.shape

        x1 = self.info_embedding(x1.flatten(0, 1)).reshape(bs, l, -1)
        bs, l, c = x.shape

        x = x.flatten(0, 1)
        x = self.embedding(x).reshape(bs, l, c)

        x = torch.cat([x, x1], dim=-1)

        x = self.transformer(x)
        _, x = self.gru(x)
        x = torch.squeeze(x)

        x2 = self.global_embedding(x2)

        x = torch.cat([x, x2], dim=-1)
        f = self.mlp(x)

        x = self.final(f)

        return x, f


# updated LexStyleNet with model from paper
class LexStyleNet(nn.Module):
    def __init__(self, h=34, w=512):
        super(LexStyleNet, self).__init__()

        self.firstconv = nn.Conv1d(h, 256, 2)
        self.convs = nn.ModuleList()

        self.down = nn.AvgPool1d(2)

        in_channels = 256
        out_channels = [128, 128]
        for ix in range(2):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels[ix], 2),
                    nn.InstanceNorm1d(out_channels[ix]),
                    nn.ReLU(),
                    # nn.Dropout(0.25)
                )
            )

            in_channels = copy.copy(out_channels[ix])

            w = w // 2

        features = 3

        self.out_size = 8064
        self.out = nn.Sequential(
            nn.Linear(31872, 128),
            nn.LayerNorm((128,)),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.LogSoftmax(dim=-1),
        )  # <- use if not LabelSmoothing  #31872 if original, 63872 if lex's

    def forward(self, x):
        x = self.firstconv(x)
        for ix in range(len(self.convs)):
            x = self.convs[ix](x)
            x = self.down(x)

        x = x.flatten(1, 2)
        print(x.shape)

        return self.out(x)


class DemoNet(nn.Module):
    def __init__(
        self,
        convDim,
        imgRows,
        imgCols,
        convSize,
        poolSize,
        useDropout,
        n_blocks=3,
        num_params=5,
    ):
        super(DemoNet, self).__init__()
        self.convDim = convDim
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.convSize = convSize
        self.poolSize = poolSize
        self.useDropout = useDropout
        self.n_blocks = n_blocks
        self.num_params = num_params

        self.conv_layers = nn.ModuleList()
        self.lin_layers = nn.ModuleList()
        self.final_block = nn.ModuleList()

        if self.convDim == "2d":
            inputShape = (1, self.imgRows, self.imgCols)
            convFunc = Conv2d
            poolFunc = MaxPool2d
        else:
            inputShape = (self.imgRows, self.imgCols)
            convFunc = Conv1d
            poolFunc = AvgPool1d

        self.conv_layers.append(convFunc(1, 128, kernel_size=self.convSize))
        self.conv_layers.append(poolFunc(kernel_size=self.poolSize))
        if self.useDropout:
            self.conv_layers.append(Dropout(0.25))

        for _block in range(self.n_blocks):
            self.conv_layers.append(convFunc(128, 128, kernel_size=2))
            self.conv_layers.append(poolFunc(kernel_size=self.poolSize))
            if self.useDropout:
                self.conv_layers.append(Dropout(0.25))

        self.conv_layers.append(Flatten())

        self.lin_layers.append(Linear(self.imgRows, 32))
        if self.useDropout:
            self.lin_layers.append(Dropout(0.25))

        self.final_block.append(Linear(32, 256))
        if useDropout:
            self.final_block.append(Dropout(0.25))

        self.final_block.append(Linear(42, self.num_params))

    def forward(self, x):
        conv_x, lin_x = x[:, :, 1:], torch.squeeze(x[:, :, 0])
        print("ConvX shape", conv_x.shape)
        print("LinX shape", lin_x.shape)
        print()

        for layer in self.conv_layers:
            conv_x = F.relu(layer(conv_x))
            print("Conv block", layer, conv_x.shape)

        print()

        for layer in self.lin_layers:
            lin_x = F.relu(layer(lin_x))
            print("Lin block", layer, lin_x.shape)

        print()

        fin_x = torch.concatenate([conv_x, lin_x])
        for layer in self.final_block[:-1]:
            fin_x = F.relu(layer(fin_x))
            print("Fin block", layer, fin_x.shape)

        out = F.softmax(self.final_block[-1](fin_x))
        return out


if __name__ == "__main__":
    model = TransformerClassifier()

    x = torch.zeros((16, 351, 128))
    x1 = torch.zeros((16, 351, 12))
    x2 = torch.zeros((16, 37))

    y = model(x, x1, x2)
    print(y.shape)
