import torch
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm

def train_vae_encoder_forget(vae, dataloader, epochs=5, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    vae.train()
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(vae.encoder.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        print(f"[VAE Unlearning Epoch {epoch+1}/{epochs}]:")
        for imgs, _ in tqdm(dataloader):
            imgs = imgs.to(device)

            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample()
                reconstructed = vae.decode(latents).sample

            loss = loss_fn(reconstructed, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()  # clear cache after each step
