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
from tqdm import tqdm


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
    img = torch.randn((1, 3, img_size, img_size)).to(device)

    for i in tqdm(reversed(range(T)), desc="Sampling", total=T):
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

@torch.no_grad()
def inpaint(model, device, img, mask, t_max=50):
    img = img.to(device)
    mask = mask.to(device)
    
    # Init x at t_max
    t_start = torch.ones((img.shape[0],), device=device).long() * (t_max - 1)
    x = forward_diffusion_sample(img, t_start, device)[0]
    
    for i in tqdm(reversed(range(t_max)), desc="Inpainting", total=t_max):
        t = torch.tensor([i], device=device).long()
        x_unknown = sample_timestep(model, x, t)
        
        if i == 0:
            x_known = img
        else:
            t_minus_1 = torch.tensor([i - 1], device=device).long()
            x_known = forward_diffusion_sample(img, t_minus_1, device)[0]
            
        x = mask * x_known + (1.0 - mask) * x_unknown

    plt.figure(figsize=(4, 4))
    show_tensor_image((x.detach().cpu() + 1) / 2)  # normalize to 0-1 range for showing
    plt.axis("off")
    plt.show()

    return x

def test_image_inpainting():
    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 256
    t_max = 300  # For full generation, or use 50 for local touch up; RePaint uses the whole T

    model = SimpleUnet().to(device)
    ckpt = "./ddpm_mse_epochs_5000.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Load image
    img_path = "datasets-1/test/cls0/1.jpg"
    raw_img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)
    
    # Create mask (1 is known, 0 is unknown/to inpaint)
    mask = torch.ones_like(img_tensor).to(device)
    # create a hole
    mask[:, :, 100:150, 100:150] = 0.0
    
    res = inpaint(model, device, img_tensor, mask, t_max=t_max)
    save_image((res + 1) / 2, "inpainted_result.png")
    print("Saved inpainted image to inpainted_result.png")


if __name__ == "__main__":
    test_image_generation()
    test_image_inpainting()