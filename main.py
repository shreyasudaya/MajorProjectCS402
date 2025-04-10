import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import os
import glob
import random
from torch.nn.functional import cosine_similarity

# -----------------------
# CONFIG
# -----------------------
FORGET_CONCEPT = "Eiffel Tower"
FORGET_IMAGES_DIR = "svd_unlearn_outputs/eiffel"
PRESERVE_PROMPTS = [
    "A Japanese shrine with cherry blossoms",
    "A painting of a mountain landscape",
    "A street in New York at night",
    "A sunny beach with palm trees",
    "A cathedral with gothic architecture"
]
NUM_TRAIN_STEPS = 400
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load Models
# -----------------------
print("Loading Stable Diffusion and CLIP...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to(DEVICE)
clip_model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# We'll fine-tune just the text encoder of CLIP used in SD
text_encoder: CLIPTextModel = pipe.text_encoder
tokenizer: CLIPTokenizer = pipe.tokenizer

# -----------------------
# Editable Parameters
# -----------------------
editable_params = list(text_encoder.text_model.final_layer_norm.parameters()) + \
                  list(text_encoder.text_model.embeddings.parameters())

for p in text_encoder.parameters():
    p.requires_grad = False
for p in editable_params:
    p.requires_grad = True

optimizer = optim.Adam(editable_params, lr=LR)

# -----------------------
# Load Forget Images
# -----------------------
image_paths = sorted(glob.glob(os.path.join(FORGET_IMAGES_DIR, "*.jpg")))
print(f"Loaded {len(image_paths)} forget concept images.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
])

def load_image_tensor(path):
    image = default_loader(path)
    return transform(image).unsqueeze(0).to(DEVICE)

forget_image_tensors = torch.cat([load_image_tensor(p) for p in image_paths], dim=0)

# Get vision features of forget images
with torch.no_grad():
    image_features = clip_model.get_image_features(pixel_values=forget_image_tensors)

# Normalize image features
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

# -----------------------
# Training Loop
# -----------------------
print("Starting training...")
for step in tqdm(range(NUM_TRAIN_STEPS)):
    optimizer.zero_grad()

    # Forget concept embedding
    inputs = tokenizer(FORGET_CONCEPT, return_tensors="pt").to(DEVICE)
    text_embedding = text_encoder(**inputs).last_hidden_state.mean(dim=1)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    # Cosine sim loss with forget image features (maximize dissimilarity)
    sim_scores = cosine_similarity(text_embedding, image_features)
    forget_loss = sim_scores.mean()

    # Preservation loss
    preserve_prompt = random.choice(PRESERVE_PROMPTS)
    preserve_input = tokenizer(preserve_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        original_preserve_emb = text_encoder(**preserve_input).last_hidden_state.mean(dim=1)
    new_preserve_emb = text_encoder(**preserve_input).last_hidden_state.mean(dim=1)
    preserve_loss = nn.MSELoss()(original_preserve_emb, new_preserve_emb)

    total_loss = -forget_loss + preserve_loss
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: Forget Loss = {forget_loss.item():.4f}, Preserve Loss = {preserve_loss.item():.4f}")

# -----------------------
# Save Edited Encoder
# -----------------------
print("Saving modified CLIP text encoder...")
text_encoder.save_pretrained("edited_clip_text_encoder")
tokenizer.save_pretrained("edited_clip_text_encoder")

# -----------------------
# Generate Test Images
# -----------------------
print("Generating images with edited model...")
edited_encoder = CLIPTextModel.from_pretrained("edited_clip_text_encoder").to(DEVICE, dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    text_encoder=edited_encoder,
    torch_dtype=torch.float16,
    safety_checker=None,
).to(DEVICE)

test_prompts = [FORGET_CONCEPT, "A famous monument in Paris", "The Eiffel Tower", "Tokyo Tower", "A cat in Paris"]
os.makedirs("outputs", exist_ok=True)

for idx, prompt in enumerate(test_prompts):
    image = pipe(prompt).images[0]
    image.save(f"outputs/test_{idx}_{prompt[:10].replace(' ', '_')}.png")

print("Done! Check the 'outputs/' folder for generated images.")
