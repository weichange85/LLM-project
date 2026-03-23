from datasets import load_dataset

def format_prompt(example):
    return f"""### Prompt:
{example['prompt']}

### Response:
{example['response']}"""


def tokenize_function(example, tokenizer, max_length):
    text = format_prompt(example)

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def load_data(tokenizer, config):
    data_path = config["data"]["train_path"]
    max_length = config["training"]["max_length"]

    dataset = load_dataset("json", data_files=data_path)

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length)
    )

    return dataset["train"]