from mb_explainer.arithmetic import dtype_bytes, expected_draft_tokens
from mb_explainer.catalog import (
    CONTEXTS,
    GPT2,
    KV_DTYPES,
    QWEN,
    SPECULATIVE_ACCEPTANCE,
    SPECULATIVE_DRAFTS,
    SPECULATIVE_SOURCE,
    head_dim,
)
from mb_explainer.kv import ceiling, kv_bytes_per_token
from mb_explainer.models import gpt2_mask_elements, gpt2_parameters, qwen_parameters


def contexts_for(limit: int) -> list[int]:
    values = list(CONTEXTS)
    if limit not in values:
        values.append(limit)
    return values


def dtype_block(
    parameters: int,
    layers: int,
    kv_heads: int,
    dim: int,
    name: str,
    bandwidth: float,
    contexts: list[int],
) -> dict:
    width = dtype_bytes(name)
    per_token = kv_bytes_per_token(layers, kv_heads, dim, width)
    weights = parameters * width
    rows = []
    for context in contexts:
        rows.append(
            {
                "context": context,
                "bytesPerToken": weights + per_token * context,
                "tokensPerSecondCeiling": ceiling(bandwidth, weights, per_token, context),
            }
        )
    return {
        "dtype": name,
        "bytesPerElement": width,
        "kvBytesPerToken": per_token,
        "weightBytes": weights,
        "contexts": rows,
    }


def gpt2_model(row: dict, bandwidth: float) -> dict:
    parameters = gpt2_parameters(
        row["nLayer"], row["nEmbd"], row["nPositions"], row["vocab"], row["nInner"]
    )
    mask = gpt2_mask_elements(row["nLayer"], row["nPositions"])
    dim = row["nEmbd"] // row["nHead"]
    contexts = contexts_for(row["nPositions"])
    blocks = [
        dtype_block(parameters, row["nLayer"], row["nHead"], dim, name, bandwidth, contexts)
        for name in KV_DTYPES
    ]
    return {
        "id": row["id"],
        "revision": row["revision"],
        "configSource": row["configSource"],
        "parameterSource": row["parameterSource"],
        "parameters": parameters,
        "safetensorsElements": row["safetensorsElements"],
        "maskElements": mask,
        "layers": row["nLayer"],
        "kvHeads": row["nHead"],
        "headDim": dim,
        "maxPositionEmbeddings": row["nPositions"],
        "tieWordEmbeddings": True,
        "nInner": row["nInner"],
        "nInnerInConfig": row["nInnerInConfig"],
        "dtypes": blocks,
    }


def qwen_model(row: dict, bandwidth: float) -> dict:
    dim = head_dim(row)
    parameters = qwen_parameters(
        row["hidden"],
        row["intermediate"],
        row["layers"],
        row["vocab"],
        row["kvHeads"],
        dim,
        row["tied"],
    )
    contexts = contexts_for(row["maxPositionEmbeddings"])
    blocks = [
        dtype_block(parameters, row["layers"], row["kvHeads"], dim, name, bandwidth, contexts)
        for name in KV_DTYPES
    ]
    return {
        "id": row["id"],
        "revision": row["revision"],
        "configSource": row["configSource"],
        "parameterSource": row["parameterSource"],
        "parameters": parameters,
        "safetensorsElements": row["safetensorsElements"],
        "layers": row["layers"],
        "kvHeads": row["kvHeads"],
        "headDim": dim,
        "maxPositionEmbeddings": row["maxPositionEmbeddings"],
        "tieWordEmbeddings": row["tied"],
        "dtypes": blocks,
    }


def speculative_rows() -> list[dict]:
    rows = []
    for acceptance in SPECULATIVE_ACCEPTANCE:
        for drafts in SPECULATIVE_DRAFTS:
            rows.append(
                {
                    "acceptance": acceptance,
                    "draftTokens": drafts,
                    "expectedTokensPerStep": expected_draft_tokens(acceptance, drafts),
                }
            )
    return rows


def kv_payload(bandwidth: float, commit: str) -> dict:
    models = [gpt2_model(row, bandwidth) for row in GPT2]
    models.extend(qwen_model(row, bandwidth) for row in QWEN)
    return {
        "version": 1,
        "gitCommit": commit,
        "formula": "2 * layers * kvHeads * headDim * bytesPerElement",
        "weightBytesNote": "parameters times the dtype width, embeddings and norms included",
        "bandwidthBytesPerSecond": bandwidth,
        "bandwidthKernel": "float64 triad, best of the timed calls",
        "models": models,
        "speculative": {
            "source": SPECULATIVE_SOURCE,
            "measured": False,
            "note": (
                "Expected tokens from one target-model step when each draft token is "
                "accepted with the given probability and drafting is free. Upper bound "
                "from the paper, not a run on this machine."
            ),
            "rows": speculative_rows(),
        },
    }
