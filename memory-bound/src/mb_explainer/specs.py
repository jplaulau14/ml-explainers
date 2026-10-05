from mb_explainer.arithmetic import ridge_intensity

SI_GB = 10**9
SI_TB = 10**12
TFLOP = 10**12

H100_PAGE = "https://www.nvidia.com/en-us/data-center/h100/"
H100_DATASHEET = (
    "https://resources.nvidia.com/en-us-data-center-overview/nvidia-tensor-core-gpu-datasheet"
)
A100_DATASHEET = (
    "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/"
    "a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf"
)
ADA_WHITEPAPER = (
    "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf"
)
M2_ULTRA = "https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/"


def _chip(
    name: str,
    memory: str,
    bandwidth_label: str,
    bandwidth: float,
    dense_tflops: float,
    sources: list[dict[str, str]],
    extra: dict,
) -> dict:
    flops = dense_tflops * TFLOP
    return {
        "name": name,
        "memory": memory,
        "bandwidthLabel": bandwidth_label,
        "bandwidthBytesPerSecond": bandwidth,
        "denseFp16Tflops": dense_tflops,
        "denseFp16FlopsPerSecond": flops,
        "ridgeFlopsPerByte": ridge_intensity(flops, bandwidth),
        "sources": sources,
        **extra,
    }


def h100() -> dict:
    sparse = 1979
    note = (
        "The H100 product page lists 1,979 teraFLOPS of FP16 tensor core for H100 SXM "
        "with the footnote 'With sparsity'. The H100 datasheet says specifications shown "
        "with sparsity are 1/2 lower without sparsity."
    )
    return _chip(
        "NVIDIA H100 SXM",
        "80GB",
        "3.35TB/s",
        3.35 * SI_TB,
        sparse / 2,
        [
            {"field": "memory, bandwidth, fp16TensorTflopsWithSparsity", "url": H100_PAGE},
            {"field": "sparsity is twice the dense rate", "url": H100_DATASHEET},
        ],
        {
            "fp16TensorTflopsWithSparsity": sparse,
            "denseFp16Derivation": "1979 / 2",
            "sparsityNote": note,
            "datasheet": {
                "source": H100_DATASHEET,
                "fp16TensorTflopsWithSparsity": 2000,
                "denseFp16Tflops": 1000,
                "bandwidthLabel": "3TB/s",
                "memory": "80GB",
            },
        },
    )


def a100() -> dict:
    note = (
        "The datasheet lists FP16 Tensor Core as 312 TFLOPS | 624 TFLOPS*. "
        "The footnote says the starred number is with sparsity."
    )
    return _chip(
        "NVIDIA A100 80GB SXM",
        "80GB HBM2e",
        "2039GB/s",
        2039 * SI_GB,
        312,
        [{"field": "memory, bandwidth, fp16 tensor core", "url": A100_DATASHEET}],
        {"fp16TensorTflopsWithSparsity": 624, "sparsityNote": note},
    )


def rtx4090() -> dict:
    note = (
        "Appendix A lists Peak FP16 Tensor TFLOPS with FP16 Accumulate as 330.3/660.6. "
        "Footnote 2 says the second number uses the sparsity feature. The same table "
        "lists FP32-accumulate FP16 tensor core as 165.2/330.4."
    )
    return _chip(
        "NVIDIA GeForce RTX 4090",
        "24 GB GDDR6X",
        "1008 GB/sec",
        1008 * SI_GB,
        330.3,
        [{"field": "memory, bandwidth, fp16 tensor core", "url": ADA_WHITEPAPER}],
        {
            "fp16TensorTflopsWithSparsity": 660.6,
            "accumulate": "FP16",
            "denseFp16Fp32AccumulateTflops": 165.2,
            "fp16Fp32AccumulateTflopsWithSparsity": 330.4,
            "sparsityNote": note,
        },
    )


def chips() -> list[dict]:
    return [h100(), a100(), rtx4090()]


def omitted() -> list[dict]:
    return [
        {
            "name": "Apple M2 Ultra",
            "memory": "up to 192GB",
            "bandwidthLabel": "800GB/s",
            "source": M2_ULTRA,
            "reason": (
                "Apple's newsroom post lists 800GB/s of memory bandwidth and up to 192GB "
                "of unified memory. It does not publish dense 16-bit FLOP/s."
            ),
        }
    ]


def payload() -> dict:
    return {
        "version": 1,
        "bytesNote": (
            "Bandwidth is converted with SI units: 1 GB/s = 1e9 bytes/s and "
            "1 TB/s = 1e12 bytes/s. Memory capacity is the vendor's string."
        ),
        "chips": chips(),
        "omitted": omitted(),
    }
