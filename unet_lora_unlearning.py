import torch
from torch import nn, optim
from tqdm import tqdm

def train_unet_forget(unet, latents, noise_scheduler, num_epochs=3, lr=1e-4, device="cuda"):
    unet.train()
    unet.to(device)

    optimizer = optim.Adam(unet.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        pbar = tqdm(latents, desc=f"[U-Net Unlearning Epoch {epoch+1}/{num_epochs}]")
        for latent in pbar:
            latent = latent.to(device)
            noise = torch.randn_like(latent)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_latent = noise_scheduler.add_noise(latent, noise, timesteps)
            
            noise_pred = unet(noisy_latent, timesteps).sample
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
