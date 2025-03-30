from transformers import CLIPTextModel, CLIPTokenizer

# Path to your locally saved CLIP model
local_dir = "./clip_model"

# Load tokenizer and model from local directory
clip_tokenizer = CLIPTokenizer.from_pretrained(local_dir)
clip_model = CLIPTextModel.from_pretrained(local_dir)
