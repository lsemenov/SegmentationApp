import torch


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = torch.nn.Dropout(p=0.4)
        self.layer1 = DoubleConv(1, 64)
        self.layer2 = DoubleConv(64, 128)
        self.layer3 = DoubleConv(128, 256)
        self.layer4 = DoubleConv(256, 512)

        self.layer5 = DoubleConv(512 + 256, 256)
        self.layer6 = DoubleConv(256 + 128, 128)
        self.layer7 = DoubleConv(128 + 64, 64)
        self.layer8 = torch.nn.Conv2d(64, 1, 1)

        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        x1m = self.dropout(x1m)

        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
        x2m = self.dropout(x2m)

        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
        x2m = self.dropout(x2m)

        x4 = self.layer4(x3m)

        x5 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.dropout(x5)
        x5 = self.layer5(x5)

        x6 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.dropout(x6)
        x6 = self.layer6(x6)

        x7 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.dropout(x7)
        x7 = self.layer7(x7)

        ret = self.layer8(x7)

        return ret


class DoubleConv(torch.nn.Module):
    """
    Helper Class which implements the intermediate Convolutions
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.BatchNorm2d(out_channels),
                                        torch.nn.ReLU(),

                                        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.BatchNorm2d(out_channels),
                                        torch.nn.ReLU())

    def forward(self, X):
        return self.step(X)
