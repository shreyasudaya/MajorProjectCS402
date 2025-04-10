import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from peft import get_peft_model, LoraConfig, TaskType
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# CONFIG
# --------------------------
PROMPTS_TO_FORGET = ["Eiffel Tower", "Elon Musk"]
PROMPTS_TO_RETAIN = ["a mountain landscape", "a car on the road"]
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "svd_unlearn_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Load Stable Diffusion
# --------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to(device)

tokenizer: CLIPTokenizer = pipe.tokenizer
text_encoder: CLIPTextModel = pipe.text_encoder

# --------------------------
# Apply LoRA to text encoder
# --------------------------
config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],  # ClipTextTransformer target layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)
text_encoder = get_peft_model(text_encoder, config)
text_encoder.to(device)

# --------------------------
# Helper: Get embedding for prompt
# --------------------------
@torch.no_grad()
def get_embedding(prompt: str):
    tokenized = tokenizer(prompt, return_tensors="pt").to(device)
    output = text_encoder.base_model(**tokenized)
    return output.last_hidden_state.squeeze(0)  # shape: [seq_len, dim]

# --------------------------
# SVD Forgetting
# --------------------------
def svd_unlearn(prompt_embedding, layer: nn.Linear, rank=4):
    """
    Modifies a LoRA-injected linear layer by removing the components
    aligned with the prompt embedding using SVD.
    """
    W = layer.weight.data.cpu()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    prompt_vector = prompt_embedding.mean(dim=0).cpu()
    direction = prompt_vector / prompt_vector.norm()

    proj = U @ torch.diag(S)
    projection_along_direction = (proj @ direction.unsqueeze(1)) @ direction.unsqueeze(0)
    proj -= projection_along_direction

    # Recompose
    U_new, S_new, Vt_new = torch.linalg.svd(proj, full_matrices=False)
    W_new = (U_new[:, :rank] * S_new[:rank]) @ Vt_new[:rank, :]
    layer.weight.data = W_new.to(layer.weight.device)

# --------------------------
# Apply forgetting via SVD
# --------------------------
for prompt in PROMPTS_TO_FORGET:
    emb = get_embedding(prompt)
    print(f"Unlearning prompt: {prompt}")

    for name, module in text_encoder.named_modules():
        if isinstance(module, nn.Linear) and 'lora' in name:
            svd_unlearn(emb, module)

# --------------------------
# Generate images
# --------------------------
@torch.no_grad()
def generate(prompt, file):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(os.path.join(OUTPUT_DIR, file))

# Forget prompts - should generate distorted/blank
for prompt in PROMPTS_TO_FORGET:
    generate(prompt, f"forgotten_{prompt.replace(' ', '_')}.png")

# Retain prompts - should remain normal
for prompt in PROMPTS_TO_RETAIN:
    generate(prompt, f"retained_{prompt.replace(' ', '_')}.png")

print("✅ Done. Check svd_unlearn_outputs/")
