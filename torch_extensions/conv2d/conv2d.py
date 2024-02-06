import imageio.v3 as iio
import torch
import torch.nn as nn
import torch.nn.functional as F
import nn_api


def iio_image_to_tensor(image: any):
    image_tensor = torch.from_numpy(image)
    # iio [height, width, channels] --> [channels, height, width]
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()
    return image_tensor.float().div(255.0).unsqueeze(dim=0)


def resize_image(image_tensor, max_dimension=1024, min_dimension=576):
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


def torch_conv2d():
    conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
    conv.weight.data = torch.ones(3, 3, 3, 3) * (0.11111 / 3)
    conv.bias.data = torch.zeros(3)
    return conv


def to_cpp_f32(tensor: torch.Tensor):
    tensor_f32 = tensor.contiguous()
    tensor_ptr = tensor_f32.data_ptr()
    tensor_num_elements = tensor_f32.numel()
    return tensor_ptr, tensor_num_elements


def main():
    device = 'cuda'
    torch.set_default_device(device)
    image_tensor = read_images_as_tensor(images_uri=['./tests/cow_rgb.jpeg'], device=device)[0]

    conv = torch_conv2d()

    x0 = conv(image_tensor)
    x1 = nn_api.conv2d_fwd(image_tensor, conv.weight.data, conv.bias.data)

    iio.imwrite('tests/cow_gray0.jpeg', (x0.squeeze().permute(1, 2, 0).cpu() * 255).byte().numpy())
    iio.imwrite('tests/cow_gray1.jpeg', (x1.squeeze().permute(1, 2, 0).cpu() * 255).byte().numpy())

main()