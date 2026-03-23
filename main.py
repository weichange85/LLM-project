import argparse
from src.config import load_config
from src.model import load_model, apply_lora
from src.preprocessing import load_data
from src.train import train
from src.inference import load_inference_model, generate


def run_train(config):
    model, tokenizer = load_model(config)
    model = apply_lora(model, config)

    dataset = load_data(tokenizer, config)

    train(model, dataset, config)


def run_inference(config, prompt=None):
    model, tokenizer = load_model(config)
    model = load_inference_model(config)

    while True:
        prompt = input("\nEnter prompt (or 'exit'): ")
        if prompt.lower() == "exit":
            break

        output = generate(model, tokenizer, prompt)
        print("\n=== OUTPUT ===")
        print(output)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--prompt", default="Explain machine learning.")

    args = parser.parse_args()

    config = load_config()

    if args.mode == "train":
        run_train(config)
    else:
        run_inference(config, args.prompt)

    if args.mode == "infer":
        run_inference(config, args.prompt)


if __name__ == "__main__":
    main()