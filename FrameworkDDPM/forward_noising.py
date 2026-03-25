import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader import load_transformed_dataset, show_tensor_image

# return betas for each time step, linearly increasing from beta_start to beta_end (0.0001 to 0.02)
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

# get value from vals, given time_step t and x_shape [batch_size, channels, height, width]
def get_index_from_list(vals, time_step, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = time_step.shape[0]
    out = vals.gather(-1, time_step.cpu())  # gather values at the time_step indices
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time_step.device)




# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different hyperparameters (alpha and beta) for closed form
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

# add noise to the images at each time sttep
def forward_diffusion_sample(x_0, time_step, device="cpu"):
    noise = torch.randn_like(x_0).to(device)    # generate epsilon

    # get two coefficients to calculate x_t
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, time_step, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, time_step, x_0.shape)

    # calculate x_t and return x_t and epsilon
    x_t = sqrt_alphas_cumprod_t * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t * noise

    return x_t, noise