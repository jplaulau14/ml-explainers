import gc
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mb_explainer.arithmetic import dtype_bytes, tokens_per_second
from mb_explainer.catalog import DECODE_MODEL_ID, head_dim, qwen_by_id
from mb_explainer.kv import kv_bytes_per_token
from mb_explainer.notes import cache_fit, pace
from mb_explainer.settings import Settings
from mb_explainer.timing import median
from mb_explainer.weights import stored_bytes

PROMPT = "The capital of France is Paris."
TORCH = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def token_row(tokenizer, count: int) -> torch.Tensor:
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
    repeats = (count + int(ids.numel()) - 1) // int(ids.numel())
    return ids.repeat(repeats)[:count]


def batch_ids(tokenizer, count: int, batch: int) -> torch.Tensor:
    return token_row(tokenizer, count).unsqueeze(0).repeat(batch, 1)


def cache_tensors(past) -> list[torch.Tensor]:
    if hasattr(past, "key_cache"):
        return [tensor for tensor in [*past.key_cache, *past.value_cache] if tensor is not None]
    found = []
    for layer in past:
        found.extend(item for item in layer if isinstance(item, torch.Tensor))
    return found


def measured_kv_bytes(past, tokens: int) -> int:
    total = sum(tensor.numel() * tensor.element_size() for tensor in cache_tensors(past))
    return total // tokens


def forward(model, ids: torch.Tensor, past=None):
    with torch.inference_mode():
        if past is None:
            return model(ids, use_cache=True)
        return model(ids, past_key_values=past, use_cache=True)


def prefill_seconds(model, ids: torch.Tensor, repeats: int) -> float:
    forward(model, ids)
    took = []
    for _ in range(repeats):
        start = time.perf_counter()
        forward(model, ids)
        took.append(time.perf_counter() - start)
    return median(took)


def decode_seconds(model, ids: torch.Tensor, new_tokens: int) -> float:
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        start = time.perf_counter()
        for _ in range(new_tokens):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return time.perf_counter() - start


def median_decode(model, ids: torch.Tensor, new_tokens: int, repeats: int) -> float:
    decode_seconds(model, ids, new_tokens)
    return median([decode_seconds(model, ids, new_tokens) for _ in range(repeats)])


def load_model(dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        DECODE_MODEL_ID, dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()
    return model


def quantize(model):
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def drop(model) -> None:
    del model
    gc.collect()


def failed(dtype: str, message: str) -> dict:
    return {"dtype": dtype, "status": "failed", "error": message}


def formula_kv(element_bytes: int) -> int:
    row = qwen_by_id(DECODE_MODEL_ID)
    return kv_bytes_per_token(row["layers"], row["kvHeads"], head_dim(row), element_bytes)


def prediction(weight_bytes: int, kv_per_token: int, mean_context: float, bandwidth: float) -> dict:
    weight_ceiling = tokens_per_second(bandwidth, weight_bytes)
    combined = weight_bytes + kv_per_token * mean_context
    return {
        "predictedTokensPerSecondWeights": weight_ceiling,
        "predictedTokensPerSecondWeightsAndKv": tokens_per_second(bandwidth, combined),
        "meanContext": mean_context,
    }


def rates(new_tokens: int, seconds: float, batch: int) -> dict:
    total = batch * new_tokens / seconds
    return {"totalTokensPerSecond": total, "perSequenceTokensPerSecond": total / batch}


def precision_row(dtype, model, ids, settings: Settings, bandwidth: float, l3: int) -> dict:
    stored, linear, parameters, quantized = stored_bytes(model)
    prefill = prefill_seconds(model, ids, settings.decode_repeats)
    past = forward(model, ids).past_key_values
    kv = measured_kv_bytes(past, ids.shape[-1])
    element = cache_tensors(past)[0].element_size()
    seconds = median_decode(model, ids, settings.new_tokens, settings.decode_repeats)
    decode = settings.new_tokens / seconds
    mean_context = settings.prompt_tokens + (settings.new_tokens - 1) / 2
    predicted = prediction(stored, kv, mean_context, bandwidth)
    return {
        "dtype": dtype,
        "status": "ok",
        "uniqueParameters": parameters,
        "uniqueBytes": stored,
        "linearBytes": linear,
        "quantizedLinears": quantized,
        "prefillSecondsMedian": prefill,
        "prefillTokensPerSecond": settings.prompt_tokens / prefill,
        "decodeSecondsMedian": seconds,
        "decodeTokensPerSecond": decode,
        "measuredKvBytesPerToken": kv,
        "kvElementBytes": element,
        "formulaKvBytesPerToken": formula_kv(element),
        "cacheFit": cache_fit(stored, l3),
        "pace": pace(decode, predicted["predictedTokensPerSecondWeights"]),
        **predicted,
    }


def batch_row(dtype: str, batch: int, new_tokens: int, seconds: float) -> dict:
    return {
        "dtype": dtype,
        "batchSize": batch,
        "decodeSecondsMedian": seconds,
        **rates(new_tokens, seconds, batch),
    }


def sweep_batches(model, tokenizer, dtype: str, settings: Settings, first: float) -> list[dict]:
    rows = [batch_row(dtype, 1, settings.new_tokens, first)]
    for batch in settings.batches:
        if batch == 1:
            continue
        ids = batch_ids(tokenizer, settings.prompt_tokens, batch)
        seconds = median_decode(model, ids, settings.new_tokens, settings.decode_repeats)
        rows.append(batch_row(dtype, batch, settings.new_tokens, seconds))
    return rows


def loaded_revision(model_id: str) -> str:
    home = os.environ.get("HF_HOME", "")
    slug = "models--" + model_id.replace("/", "--")
    path = Path(home) / "hub" / slug / "refs" / "main" if home else None
    if path is not None and path.is_file():
        return path.read_text().strip()
    return qwen_by_id(model_id)["revision"]


def run_loaded(
    dtype: str, model, tokenizer, settings: Settings, bandwidth: float, l3: int
) -> tuple[dict, list]:
    ids = batch_ids(tokenizer, settings.prompt_tokens, 1)
    row = precision_row(dtype, model, ids, settings, bandwidth, l3)
    batches = sweep_batches(model, tokenizer, dtype, settings, row["decodeSecondsMedian"])
    row["revision"] = loaded_revision(DECODE_MODEL_ID)
    return row, batches


def run_precision(
    dtype: str, tokenizer, settings: Settings, bandwidth: float, l3: int
) -> tuple[dict, list]:
    model = load_model(TORCH[dtype])
    try:
        return run_loaded(dtype, model, tokenizer, settings, bandwidth, l3)
    finally:
        drop(model)


def run_int8(tokenizer, settings: Settings, bandwidth: float, l3: int) -> tuple[dict, list]:
    model = load_model(torch.float32)
    try:
        model = quantize(model)
        row, batches = run_loaded("int8", model, tokenizer, settings, bandwidth, l3)
    except Exception as exc:
        return failed("int8", f"{type(exc).__name__}: {exc}"), []
    finally:
        drop(model)
    if row.get("quantizedLinears", 0) == 0:
        return failed("int8", "dynamic quantization left every linear layer unpacked"), []
    return row, batches


def model_block() -> dict:
    row = qwen_by_id(DECODE_MODEL_ID)
    return {
        "id": row["id"],
        "catalogRevision": row["revision"],
        "configSource": row["configSource"],
        "parameterSource": row["parameterSource"],
        "publishedParameters": row["safetensorsElements"],
        "layers": row["layers"],
        "kvHeads": row["kvHeads"],
        "headDim": head_dim(row),
        "prompt": PROMPT,
    }


def benchmark(machine: dict, settings: Settings) -> dict:
    from mb_explainer.machine import triad

    bandwidth = triad(machine)["bestBytesPerSecond"]
    l3 = machine["cpu"]["l3Bytes"]
    tokenizer = AutoTokenizer.from_pretrained(DECODE_MODEL_ID)
    precisions = []
    batches = []
    for name in settings.precisions:
        print(f"decode {name}", flush=True)
        if name == "int8":
            row, extra = run_int8(tokenizer, settings, bandwidth, l3)
        else:
            row, extra = run_precision(name, tokenizer, settings, bandwidth, l3)
        precisions.append(row)
        batches.extend(extra)
        if row["status"] == "ok":
            print(f"{name} decode {row['decodeTokensPerSecond']:.3f} tok/s", flush=True)
        else:
            print(f"{name} {row['error']}", flush=True)
    return {
        "model": model_block(),
        "bandwidthBytesPerSecond": bandwidth,
        "l3Bytes": l3,
        "promptTokens": settings.prompt_tokens,
        "newTokens": settings.new_tokens,
        "repeats": settings.decode_repeats,
        "threads": machine["threads"],
        "bytesPerWeightReference": {
            name: dtype_bytes(name) for name in ("float32", "bfloat16", "int8")
        },
        "precisions": precisions,
        "batch": batches,
    }
