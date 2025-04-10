import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16
).to("cuda")

vae = pipe.vae  # Variational Autoencoder
unet = pipe.unet  # U-Net model

# Set models to train mode to allow gradient computation
vae.train()
unet.train()

# Define a dummy input tensor for testing
dummy_input = torch.randn(1, 4, 64, 64, device="cuda", requires_grad=True)  # Latent input for VAE
dummy_target = torch.randn_like(dummy_input)  # Dummy target for loss computation

# Forward pass through VAE
vae_output = vae.decode(dummy_input).sample  # Decode function of VAE
loss = torch.nn.functional.mse_loss(vae_output, dummy_target)  # Dummy loss

# Backward pass to compute gradients
loss.backward()


vae_gradients = {name: param.grad.clone() for name, param in vae.named_parameters() if param.grad is not None}

# Reset gradients
vae.zero_grad()

# Forward pass through UNet
unet_output = unet(dummy_input, torch.randn(1, 4, 64, 64, device="cuda")).sample
loss = torch.nn.functional.mse_loss(unet_output, dummy_target)

# Backward pass
loss.backward()

# Extract gradient matrices of UNet
unet_gradients = {name: param.grad.clone() for name, param in unet.named_parameters() if param.grad is not None}

# Print gradient matrix shapes
print("VAE Gradients:")
for name, grad in vae_gradients.items():
    print(f"{name}: {grad.shape}")

print("\nUNet Gradients:")
for name, grad in unet_gradients.items():
    print(f"{name}: {grad.shape}")
