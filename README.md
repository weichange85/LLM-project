# LLM-project, with Low Ranking Adaption done to meta's LLama 3.2-1B model

In the purpose of creating an LLM chat bot with LoRA. This is done by taking the weights of a pretrained model, freeze the training weights, and then add a few layers at the lower ranked levels. The new layers are trained by yours truely.


# Usage

```cd /LLM-LoRA-with-Llama3.2-1B```

To train:

```python main.py --mode train```

To infer:

```python main.py --mode infer```