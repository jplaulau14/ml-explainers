from mb_explainer.models import qwen_head_dim

GPT2 = (
    {
        "id": "openai-community/gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "configSource": "https://huggingface.co/openai-community/gpt2/raw/main/config.json",
        "parameterSource": "https://huggingface.co/api/models/openai-community/gpt2",
        "nLayer": 12,
        "nEmbd": 768,
        "nHead": 12,
        "nPositions": 1024,
        "vocab": 50257,
        "nInner": 3072,
        "nInnerInConfig": None,
        "safetensorsElements": 137_022_720,
    },
    {
        "id": "openai-community/gpt2-medium",
        "revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "configSource": "https://huggingface.co/openai-community/gpt2-medium/raw/main/config.json",
        "parameterSource": "https://huggingface.co/api/models/openai-community/gpt2-medium",
        "nLayer": 24,
        "nEmbd": 1024,
        "nHead": 16,
        "nPositions": 1024,
        "vocab": 50257,
        "nInner": 4096,
        "nInnerInConfig": None,
        "safetensorsElements": 379_988_992,
    },
)

QWEN = (
    {
        "id": "Qwen/Qwen2.5-0.5B",
        "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "configSource": "https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json",
        "parameterSource": "https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B",
        "hidden": 896,
        "intermediate": 4864,
        "layers": 24,
        "heads": 14,
        "kvHeads": 2,
        "vocab": 151_936,
        "tied": True,
        "maxPositionEmbeddings": 32_768,
        "safetensorsElements": 494_032_768,
    },
    {
        "id": "Qwen/Qwen2.5-7B",
        "revision": "d149729398750b98c0af14eb82c78cfe92750796",
        "configSource": "https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json",
        "parameterSource": "https://huggingface.co/api/models/Qwen/Qwen2.5-7B",
        "hidden": 3584,
        "intermediate": 18944,
        "layers": 28,
        "heads": 28,
        "kvHeads": 4,
        "vocab": 152_064,
        "tied": False,
        "maxPositionEmbeddings": 131_072,
        "safetensorsElements": 7_615_616_512,
    },
    {
        "id": "Qwen/Qwen2.5-72B",
        "revision": "efba10c8e54e91e0d9570ab5f7b51a958474d4cb",
        "configSource": "https://huggingface.co/Qwen/Qwen2.5-72B/raw/main/config.json",
        "parameterSource": "https://huggingface.co/api/models/Qwen/Qwen2.5-72B",
        "hidden": 8192,
        "intermediate": 29568,
        "layers": 80,
        "heads": 64,
        "kvHeads": 8,
        "vocab": 152_064,
        "tied": False,
        "maxPositionEmbeddings": 131_072,
        "safetensorsElements": 72_706_203_648,
    },
)

DECODE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
ROOFLINE_MODEL_ID = "Qwen/Qwen2.5-7B"
CONTEXTS = (1, 128, 1024, 4096, 32768)
KV_DTYPES = ("float16", "bfloat16", "float32", "int8")
SPECULATIVE_SOURCE = "https://arxiv.org/abs/2211.17192"
SPECULATIVE_ACCEPTANCE = (0.6, 0.8, 0.9)
SPECULATIVE_DRAFTS = (1, 2, 4, 8)


def qwen_by_id(model_id: str) -> dict:
    return next(row for row in QWEN if row["id"] == model_id)


def head_dim(row: dict) -> int:
    return qwen_head_dim(row["hidden"], row["heads"])
