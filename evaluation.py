import torch

def evaluate_unlearning(pipe, forgotten_prompts, retained_prompts, num_images=2):
    print("\n[🔍] Evaluating Forgetting and Retention")
    pipe.to("cuda")

    print("\n[❌] Testing Forgotten Prompts:")
    for prompt in forgotten_prompts:
        for i in range(num_images):
            img = pipe(prompt).images[0]
            img.save(f"eval/forgotten_{prompt.replace(' ', '_')}_{i}.png")
            print(f" -> Saved: eval/forgotten_{prompt.replace(' ', '_')}_{i}.png")

    print("\n[✅] Testing Retained Prompts:")
    for prompt in retained_prompts:
        for i in range(num_images):
            img = pipe(prompt).images[0]
            img.save(f"eval/retained_{prompt.replace(' ', '_')}_{i}.png")
            print(f" -> Saved: eval/retained_{prompt.replace(' ', '_')}_{i}.png")

    print("\n✔️ Evaluation images saved to ./eval/")
