from transformers import TrainingArguments, Trainer

def train(model, dataset, config):

    training_cfg = config["training"]

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

    model.save_pretrained(training_cfg["output_dir"])