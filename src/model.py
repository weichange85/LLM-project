from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

def load_model(config):
    model_path = config["model"]["path"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def apply_lora(model, config):
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    return get_peft_model(model, lora_config)