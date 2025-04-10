import torch
from torch import nn, optim
from tqdm import tqdm

def train_clip_forget(clip_model, images, texts, processor, num_epochs=3, lr=1e-5, device="cuda"):
    clip_model.train()
    clip_model.to(device)

    optimizer = optim.Adam(clip_model.parameters(), lr=lr)
    loss_fn = nn.CosineEmbeddingLoss()

    for epoch in range(num_epochs):
        pbar = tqdm(zip(images, texts), total=len(images), desc=f"[CLIP Unlearning Epoch {epoch+1}/{num_epochs}]")
        for img, txt in pbar:
            inputs = processor(text=[txt], images=img, return_tensors="pt", padding=True).to(device)
            outputs = clip_model(**inputs)
            
            image_emb = outputs.image_embeds
            text_emb = outputs.text_embeds
            targets = torch.ones(image_emb.shape[0]).to(device)

            loss = loss_fn(image_emb, text_emb, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
