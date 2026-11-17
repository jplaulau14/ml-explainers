from tiling_explainer.a10.ncu import METRICS
from tiling_explainer.a10.upload import code_record

COMMIT = "a" * 40
CUBLAS = "ampere_sgemm_128x64_nn"
TENSOR_OP = "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128>(T1::Params)"
HMMA = "sm__inst_executed_pipe_tensor_op_hmma.sum"
HMMA_CYCLES = "sm__pipe_tensor_op_hmma_cycles_active.sum"
TENSORS = [HMMA, HMMA_CYCLES]
FILES = {"__init__.py": b"", "cli.py": b"print('hi')\n"}
CODE = code_record(FILES)
HEADER = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
NAIVE_TIMES = (100.0, 90.0, 120.0)
TILE32_TIMES = (80.0, 60.0, 100.0)


def call(index: int, gpu_name: str = "NVIDIA A10", vendor_ms: float = 9.0) -> dict[str, object]:
    times = {"naive": NAIVE_TIMES[index], "tile32": TILE32_TIMES[index]}
    times |= {"reg4x4": 18.0, "reg8x8": 14.0, "vendor": vendor_ms}
    gpu = {"name": gpu_name, "driver": "620.70.06", "max_sm_clock_mhz": 1710}
    gpu |= {"power_limit_w": 300.0, "uuid": f"GPU-{index}", "smCount": 80, "l2Bytes": 6291456}
    software = {"cudaRuntime": "12.9", "cudaDriver": "13.0", "cupy": "13.6.0"}
    software |= {"numpy": "2.5.5", "python": "3.12.12", "modal": "1.6.4", "cupyTf32": "0"}
    return {
        "commit": COMMIT,
        "container": f"ta-{index}",
        "gpuUuid": f"GPU-{index}",
        "code": CODE,
        "gpu": gpu,
        "software": software,
        "kernels": {k: {"ms": ms, "samplesMs": [ms], "maxAbsErr": 0.0} for k, ms in times.items()},
        "attributes": {},
    }


def profile(vendor: str = CUBLAS, launches: int = 2, cycles: int = 0) -> dict[str, object]:
    lines = [HEADER]
    names = ["naive", "tile32", "reg4x4", vendor]
    for index, name in enumerate(name for name in names for _ in range(launches)):
        lines += [f'"{index}","{name}","{metric}","","{1000 + index}"' for metric in METRICS]
        lines.append(f'"{index}","{name}","{HMMA}","inst","0"')
        lines.append(f'"{index}","{name}","{HMMA_CYCLES}","cycle","{cycles}"')
    detail = {"ncu": "Version 2025.2.0.0", "command": "ncu", "tensorMetrics": TENSORS}
    return {**call(0), "gpuUuid": "GPU-9", **detail, "csv": "\n".join(lines)}
