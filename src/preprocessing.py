from datasets import load_dataset
from .config import DATA_PATH, MAX_LENGTH

def format_prompt(example):
    return f"""### Prompt:
{example['prompt']}

### Response:
{example['response']}"""

def tokenize_function(example, tokenizer):
    text = format_prompt(example)

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def load_data(tokenizer):
    dataset = load_dataset("json", data_files=DATA_PATH)

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer)
    )

    return dataset["train"]