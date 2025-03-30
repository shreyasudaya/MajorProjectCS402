import os

for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".safetensors"):
            print(os.path.join(root, file))