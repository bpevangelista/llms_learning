import argparse

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def iio_image_to_tensor(image: any):
    image_tensor = torch.from_numpy(image)
    # iio [height, width, channels] --> [channels, height, width]
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()
    return image_tensor.float().div(255.0).unsqueeze(dim=0)


def pil_image_to_tensor(image: any):
    # Support 1bpp or 8bpp
    image_tensor = torch.from_numpy(np.array(image, np.uint8, copy=True))
    if image.mode == '1':
        image_tensor *= 255

    image_channels = len(image.getbands())
    image_tensor = image_tensor.view(image.height, image.width, image_channels)
    # [height, width, channels] -> [channels, height, width]
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()
    return image_tensor.float().div(255.0).unsqueeze(dim=0)


def resize_image(image_tensor, max_dimension=1024, min_dimension=768):
    _, channels, height, width = image_tensor.shape
    ori_width, ori_height = width, height

    # Downscale
    scale_factor = min(max_dimension / width, 1.0) if width > height else min(max_dimension / height, 1.0)
    width = round(width * scale_factor)
    height = round(height * scale_factor)

    # Upscale
    scale_factor = max(min_dimension / width, 1.0) if width < height else max(min_dimension / height, 1.0)
    width = min(round(width * scale_factor), max_dimension)
    height = min(round(height * scale_factor), max_dimension)

    if width == ori_width and height == ori_height:
        return image_tensor
    else:
        return F.interpolate(image_tensor, size=(height, width), mode='bilinear')


def read_images_as_tensor(images_uri: list[str], device: str = 'torch_extensions'):
    tensors = []
    for image_uri in images_uri:
        image = iio.imread(image_uri)
        image_tensor = iio_image_to_tensor(image).to(device)
        image_tensor = resize_image(image_tensor).contiguous()
        tensors.append(image_tensor)
    return tensors


class DownBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels: int, pool_first: bool = True, batch_norm: bool = False):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2) if pool_first else None
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(x) if self.pool else x
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels: int):
        super().__init__()
        self.up0 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        self.conv0 = DownBlock(in_channels, out_channels, pool_first=False)

    def forward(self, x, residual):
        x = self.up0(x)
        x = torch.cat([x, residual], dim=1)
        x = self.conv0(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv = DownBlock(in_channels=3, out_channels=64, pool_first=False)
        self.down1 = DownBlock(in_channels=64, out_channels=128, pool_first=True)
        self.down2 = DownBlock(in_channels=128, out_channels=256, pool_first=True)
        self.down3 = DownBlock(in_channels=256, out_channels=512, pool_first=True)
        self.down4 = DownBlock(in_channels=512, out_channels=1024, pool_first=True)
        self.up4 = UpBlock(in_channels=1024, out_channels=512)
        self.up3 = UpBlock(in_channels=512, out_channels=256)
        self.up2 = UpBlock(in_channels=256, out_channels=128)
        self.up1 = UpBlock(in_channels=128, out_channels=64)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, x0):
        x1 = self.in_conv(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = x5

        y = self.up4(y, x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.out_conv(y)
        return y


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_uri', nargs='+', type=str, help='list of image_uri to process')
    args = parser.parse_args()
    return args

def main():
    torch.set_default_device('torch_extensions')
    args = handle_args()
    images_tensors = read_images_as_tensor(args.images_uri)

    unet = UNet()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(unet.parameters())

    # train a passthrough
    unet.train()
    for epoch in range(1):
        iterations = max(100, len(images_tensors))
        for it in range(iterations):
            image_tensor = images_tensors[it % len(images_tensors)]
            outputs = unet(image_tensor)
            loss = loss_func(outputs, image_tensor)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # sanity check
    unet.eval()
    image_tensor = read_images_as_tensor([args.images_uri[0]])[0]
    iio.imwrite('sanity_src.png', (image_tensor.squeeze().permute(1, 2, 0).cpu() * 255).byte().numpy())
    image_tensor = unet(image_tensor)
    iio.imwrite('sanity_dst.png', (image_tensor.squeeze().permute(1, 2, 0).cpu() * 255).byte().numpy())

main()
