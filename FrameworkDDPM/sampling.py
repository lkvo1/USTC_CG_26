import torch
from forward_noising import (
    get_index_from_list,
    sqrt_one_minus_alphas_cumprod,
    betas,
    posterior_variance,
    sqrt_recip_alphas,
    forward_diffusion_sample,
)
import matplotlib.pyplot as plt
from dataloader import show_tensor_image
from unet import SimpleUnet
import numpy as np
import cv2 as cv


@torch.no_grad()
def sample_timestep(model, x, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    noise_pred = model(x, t)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )

    if t[0] == 0:
        return model_mean

    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, device, img_size, T):
    # pure noise
    img =torch.randn((1, 3, img_size, img_size)).to(device)

    for i in reversed(range(T)):
        t = torch.tensor([i], device=device).long()
        img = sample_timestep(model, img, t)

    # plot the generated image
    plt.figure(figsize=(4, 4))
    show_tensor_image(img.detach().cpu())
    plt.axis("off")
    plt.show()

    return img

def test_image_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 300
    img_size = 256

    model = SimpleUnet().to(device)
    ckpt = "./ddpm_mse_epochs_5000.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    sample_plot_image(model, device, img_size, T)

# TODO：你需要在这个函数中实现图像的补充
# Follows: RePaint: Inpainting using Denoising Diffusion Probabilistic Models
@torch.no_grad()
def inpaint(model, device, img, mask, t_max=50):
    return img

# TODO: 你需要在这个函数中完成模型以及其他相关资源的加载，并调用inpaint进行图像补全，以生成图片
def test_image_inpainting():
    pass
    

if __name__ == "__main__":
    test_image_generation()
    test_image_inpainting()