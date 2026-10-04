import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import dataset_info, model_info
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, PreTrainedTokenizerBase, set_seed

from lora_explainer.adapter import Which, inject
from lora_explainer.config import RunConfig
from lora_explainer.data import load_tokenizer, train_loader
from lora_explainer.weights import full_deltas, lora_deltas, snapshot

LOG_EVERY = 25


@dataclass(frozen=True)
class Revisions:
    model: str
    dataset: str


@dataclass(frozen=True)
class TrainedRun:
    model: GPT2LMHeadModel
    tokenizer: PreTrainedTokenizerBase
    deltas: dict[Which, torch.Tensor]
    metadata: dict[str, Any]


def set_determinism(config: RunConfig) -> None:
    set_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config.threads)


def resolve_revisions(config: RunConfig) -> Revisions:
    return Revisions(model_info(config.model_id).sha, dataset_info(config.dataset_id).sha)


def cpu_name() -> str:
    if platform.system() == "Darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()
    if Path("/proc/cpuinfo").exists():
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        names = [line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")]
        return names[0] if names else platform.processor()
    return platform.processor()


def load_model(config: RunConfig, revision: str) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(config.model_id, revision=revision)
    return inject(model, config.rank) if config.mode == "lora" else model


def build_optimizer(model: GPT2LMHeadModel, learning_rate: float) -> torch.optim.AdamW:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=learning_rate)


def run_epoch(
    model: GPT2LMHeadModel, loader: DataLoader, optimizer: torch.optim.Optimizer, label: str
) -> None:
    model.train()
    for step, batch in enumerate(loader, start=1):
        loss = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % LOG_EVERY == 0:
            print(f"{label} step {step}/{len(loader)} loss {loss.item():.4f}", flush=True)


def collect_deltas(
    config: RunConfig, model: GPT2LMHeadModel, before: dict[Which, torch.Tensor]
) -> dict[Which, torch.Tensor]:
    if config.mode == "full":
        return full_deltas(before, snapshot(model, config.layer))
    return lora_deltas(model, config.layer)


def run_metadata(config: RunConfig, revisions: Revisions, seconds: float) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "modelRevision": revisions.model,
        "datasetRevision": revisions.dataset,
        "trainSeconds": round(seconds, 1),
        "cpu": cpu_name(),
        "torch": torch.__version__,
    }


def train(config: RunConfig) -> TrainedRun:
    set_determinism(config)
    revisions = resolve_revisions(config)
    tokenizer = load_tokenizer(config.model_id, revisions.model)
    loader = train_loader(tokenizer, config, revisions.dataset)
    model = load_model(config, revisions.model)
    before = snapshot(model, config.layer)
    optimizer = build_optimizer(model, config.learning_rate)
    start = time.perf_counter()
    for epoch in range(config.epochs):
        run_epoch(model, loader, optimizer, f"{config.name} epoch {epoch + 1}")
    seconds = time.perf_counter() - start
    deltas = collect_deltas(config, model, before)
    return TrainedRun(model, tokenizer, deltas, run_metadata(config, revisions, seconds))


def save_run(run_dir: Path, deltas: dict[Which, torch.Tensor], metadata: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(deltas, run_dir / "deltas.pt")
    (run_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
