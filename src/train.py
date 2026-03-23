from transformers import TrainingArguments, Trainer
import os
import shutil

def train(model, dataset, config):

    training_cfg = config["training"]
    output_dir = training_cfg["output_dir"]

    # ✅ clear old outputs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        per_device_train_batch_size=training_cfg["batch_size"],
        num_train_epochs=training_cfg["epochs"],
        learning_rate=training_cfg["learning_rate"],
        logging_steps=1,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    trainer.model.save_pretrained(training_cfg["output_dir"])