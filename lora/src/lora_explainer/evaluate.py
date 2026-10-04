import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, PreTrainedTokenizerBase

from lora_explainer.config import RunConfig
from lora_explainer.data import load_tokenizer, target_ids, validation_loader


@torch.no_grad()
def accuracy(model: GPT2LMHeadModel, loader: DataLoader, answers: list[int]) -> float:
    model.eval()
    correct = 0
    for batch in loader:
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        last = batch["attention_mask"].sum(dim=1) - 1
        scores = logits[torch.arange(len(last)), last][:, answers]
        correct += (scores.argmax(dim=-1) == batch["label"]).sum().item()
    return correct / len(loader.dataset)


def validation_accuracy(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerBase,
    config: RunConfig,
    dataset_revision: str,
) -> float:
    loader = validation_loader(tokenizer, config, dataset_revision)
    return accuracy(model, loader, target_ids(tokenizer))


def base_accuracy(config: RunConfig, model_revision: str, dataset_revision: str) -> float:
    tokenizer = load_tokenizer(config.model_id, model_revision)
    model = GPT2LMHeadModel.from_pretrained(config.model_id, revision=model_revision)
    return validation_accuracy(model, tokenizer, config, dataset_revision)
