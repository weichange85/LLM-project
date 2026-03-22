from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

def load_inference_model(config):
    model_path = config["model"]["path"]
    output_dir = config["training"]["output_dir"]

    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, output_dir)

    model.eval()
    return model


def generate(model, tokenizer, prompt):

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():   # ✅ prevents memory buildup
        output = model.generate(
            **inputs,
            max_new_tokens=100
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)