from forward_noising import forward_diffusion_sample
from unet import SimpleUnet
from dataloader import load_transformed_dataset
import torch.nn.functional as F
import torch
from torch.optim import Adam
import logging
from tqdm import trange
import cv2 as cv

logging.basicConfig(level=logging.INFO)

def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    
    # get predicted noise by model
    noise_pred = model(x_noisy, t)

    # calculate MSE loss between predicted noise and true noise
    loss = F.mse_loss(noise_pred, noise)
    return loss


if __name__ == "__main__":
    model = SimpleUnet()
    T = 300
    BATCH_SIZE = 1
    epochs = 5000

    dataloader = load_transformed_dataset(batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    logging.info(f"Using device: {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch_idx, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.to(device)

            # get random t for each image in the batch
            t = torch.randint(0, T, (batch.shape[0],), device=device).long()

            loss = get_loss(model, batch, t, device)
            # backpropagate the loss and update model parameters
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                logging.info(f"Epoch {epoch} | Batch index {batch_idx:03d} Loss: {loss.item()}")

    torch.save(model.state_dict(), f"./ddpm_mse_epochs_{epochs}.pth")
