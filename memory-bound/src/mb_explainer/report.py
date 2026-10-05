from mb_explainer.format import gb, gflops, rate, ratio


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def pace_sentence(name: str) -> str:
    text = {
        "slower": (
            "Measured decode is below that ceiling. One token at a time does not "
            "keep the memory bus as busy as the triad kernel."
        ),
        "faster": "Measured decode is above the DRAM ceiling, so weight reads are hitting cache.",
        "close": "Measured decode is close to that ceiling.",
    }
    return text[name]


def cache_sentence(name: str) -> str:
    if name == "larger":
        return "The stored weights are larger than L3."
    return "The stored weights fit in L3, so DRAM is not the limit."


def stream_by(machine: dict, kernel: str, dtype: str) -> dict:
    return next(
        row for row in machine["streams"] if row["kernel"] == kernel and row["dtype"] == dtype
    )


def bandwidth_table(machine: dict) -> str:
    rows = []
    for kernel, dtype in (("copy", "float64"), ("triad", "float64"), ("triad", "float32")):
        row = stream_by(machine, kernel, dtype)
        rows.append(
            [
                f"{row['dtype']} {row['kernel']}",
                str(row["workingSetBytes"]),
                str(row["exceedsL3"]).lower(),
                gb(row["bestBytesPerSecond"]),
                gb(row["medianBytesPerSecond"]),
            ]
        )
    headers = ["Kernel", "Working set", "Exceeds L3", "Best GB/s", "Median GB/s"]
    return table(headers, rows)


def peak_table(machine: dict) -> str:
    rows = []
    for row in machine["peaks"]:
        rows.append(
            [
                row["dtype"],
                str(row["n"]),
                row["status"],
                gflops(row.get("bestFlopsPerSecond", 0)),
                gflops(row.get("medianFlopsPerSecond", 0)),
            ]
        )
    return table(["Dtype", "N", "Status", "Best GFLOP/s", "Median GFLOP/s"], rows)


def machine_section(machine: dict) -> str:
    cpu = machine["cpu"]
    libs = machine["libraries"]
    memory = machine["memory"]
    lines = [
        "## Machine",
        "",
        (
            f"CPU: {cpu['model']} (family {cpu['family']}, model {cpu['modelId']}, "
            f"stepping {cpu['stepping']}), {cpu['cores']} cores."
        ),
        f"L3: {cpu['l3Bytes']} bytes. Flags of interest: {', '.join(cpu['flags']) or 'none'}.",
        (
            f"RAM: {memory['totalBytes']} bytes total, {memory['availableBytes']} bytes "
            "available when the probe started."
        ),
        (
            f"Python {libs['python']}, numpy {libs['numpy']}, torch {libs['torch']}, "
            f"transformers {libs['transformers']}."
        ),
        f"git commit {machine['gitCommit']}.",
        "",
        (
            "Bandwidth is a STREAM-style kernel in torch. Each call moves the whole array. "
            "The working set is about four times L3, capped by available RAM. "
            "The best call is the achievable figure. The median is next to it."
        ),
        "",
        bandwidth_table(machine),
        "",
        "Peak compute is a square gemm, best of the timed calls.",
        "",
        peak_table(machine),
    ]
    return "\n".join(lines)


def roofline_points(roof: dict) -> str:
    rows = [
        [
            str(point["tokens"]),
            rate(point["arithmeticIntensity"]),
            gflops(point["achievedFlopsPerSecond"]),
            gb(point["trafficBytesPerSecond"]),
        ]
        for point in roof["points"]
    ]
    return table(["Tokens", "FLOP/byte", "GFLOP/s", "Traffic GB/s"], rows)


def per_weight_table(roof: dict) -> str:
    rows = [
        [row["dtype"], str(row["bytesPerWeight"]), rate(row["flopsPerByteAtBatch1"])]
        for row in roof["perWeight"]
    ]
    return table(["Dtype", "Bytes", "FLOP/byte at batch 1"], rows)


def roofline_intro(roof: dict) -> list[str]:
    caps = roof["ceilings"]
    matrix = roof["matrix"]
    ridge = caps["analyticalRidgeTokens"]
    ridge_text = "none" if ridge is None else rate(ridge)
    measured = caps["measuredRidgeTokens"]
    measured_text = "not reached" if measured is None else str(measured)
    return [
        (
            f"Matrix: {matrix['model']} one layer of projections, "
            f"{matrix['rows']} by {matrix['cols']}, float32, "
            f"{matrix['weightElements']} weight elements."
        ),
        f"Config: {matrix['configSource']} at {matrix['revision']}.",
        (
            f"Left out of the matrix: {matrix['excludedBiasElements']} bias elements and "
            f"{matrix['excludedNormElements']} norm elements."
        ),
        (
            f"Bandwidth ceiling {gb(caps['bandwidthBytesPerSecond'])} GB/s. "
            f"Compute ceiling {gflops(caps['peakFlopsPerSecond'])} GFLOP/s. "
            f"Ridge intensity {rate(caps['ridgeArithmeticIntensity'])} FLOP/byte."
        ),
        (
            f"Analytical token count where this matrix crosses the ridge: {ridge_text}. "
            "Smallest measured point at or above 80% of the compute ceiling: "
            f"{measured_text}."
        ),
    ]


def roofline_section(roof: dict) -> str:
    lines = [
        "## Roofline",
        "",
        *roofline_intro(roof),
        "",
        "Arithmetic intensity counts both inputs and the output. Achieved FLOP/s uses the median.",
        "",
        roofline_points(roof),
        "",
        "FLOPs per byte if a token touched each weight once and did two FLOPs on it:",
        "",
        per_weight_table(roof),
    ]
    return "\n".join(lines)


def weight_ratio(row: dict) -> str:
    value = row["decodeTokensPerSecond"] / row["predictedTokensPerSecondWeights"]
    ceiling = rate(row["predictedTokensPerSecondWeights"])
    return f"Weight-only ceiling {ceiling} tokens/s (ratio {ratio(value)})."


def kv_ratio(row: dict) -> str:
    value = row["decodeTokensPerSecond"] / row["predictedTokensPerSecondWeightsAndKv"]
    ceiling = rate(row["predictedTokensPerSecondWeightsAndKv"])
    context = rate(row["meanContext"])
    return (
        f"Weights plus KV at mean context {context}: "
        f"ceiling {ceiling} tokens/s (ratio {ratio(value)})."
    )


def precision_lines(row: dict) -> list[str]:
    if row["status"] != "ok":
        return [f"### {row['dtype']}", "", f"Did not run: {row['error']}"]
    stored = (
        f"Stored bytes {row['uniqueBytes']}, of which linear weights {row['linearBytes']}. "
        f"Unique parameters counted {row['uniqueParameters']}. "
        f"Quantized linear layers: {row['quantizedLinears']}."
    )
    kv = (
        f"KV cache measured {row['measuredKvBytesPerToken']} bytes/token, "
        f"formula {row['formulaKvBytesPerToken']} bytes/token, "
        f"element size {row['kvElementBytes']}."
    )
    return [
        f"### {row['dtype']}",
        "",
        stored,
        cache_sentence(row["cacheFit"]),
        f"Prefill {rate(row['prefillTokensPerSecond'])} tokens/s.",
        f"Decode {rate(row['decodeTokensPerSecond'])} tokens/s.",
        weight_ratio(row),
        kv_ratio(row),
        kv,
        pace_sentence(row["pace"]),
    ]


def decode_section(decode: dict) -> str:
    model = decode["model"]
    protocol = (
        f'Prompt: "{model["prompt"]}", repeated out to {decode["promptTokens"]} tokens. '
        f"Then {decode['newTokens']} new tokens, greedy, cache on. "
        f"The prefill is not inside the decode timer. "
        f"Median of {decode['repeats']} calls after one warmup. "
        f"{decode['threads']} threads."
    )
    lines = [
        "## Decode",
        "",
        f"Model: {model['id']}, catalog revision {model['catalogRevision']}.",
        f"Config: {model['configSource']}.",
        protocol,
        (
            f"Published parameter count {model['publishedParameters']} "
            f"from {model['parameterSource']}."
        ),
        f"Bandwidth used for the ceilings: {gb(decode['bandwidthBytesPerSecond'])} GB/s.",
        "",
    ]
    for row in decode["precisions"]:
        lines.extend(precision_lines(row))
        lines.append("")
    return "\n".join(lines).rstrip()


def batch_section(decode: dict) -> str:
    rows = [
        [
            row["dtype"],
            str(row["batchSize"]),
            rate(row["totalTokensPerSecond"]),
            rate(row["perSequenceTokensPerSecond"]),
        ]
        for row in decode["batch"]
    ]
    intro = "Total tokens/s counts every sequence. Per sequence divides that by the batch."
    body = table(["Dtype", "Batch", "Total tokens/s", "Per sequence"], rows)
    return "\n".join(["## Batch", "", intro, "", body])


def kv_model_block(model: dict) -> list[str]:
    fp16 = next(block for block in model["dtypes"] if block["dtype"] == "float16")
    wanted = {1, 1024, 4096, 32768, model["maxPositionEmbeddings"]}
    picked = [row for row in fp16["contexts"] if row["context"] in wanted]
    shape = (
        f"{model['parameters']} parameters, {model['layers']} layers, "
        f"{model['kvHeads']} KV heads, head dim {model['headDim']}, "
        f"max position {model['maxPositionEmbeddings']}."
    )
    lines = [
        f"### {model['id']}",
        "",
        shape,
        f"Config {model['configSource']} at {model['revision']}.",
    ]
    if "maskElements" in model:
        lines.append(
            f"The safetensors file has {model['safetensorsElements']} elements. "
            f"{model['maskElements']} of those are causal masks (attn.bias), not parameters."
        )
    grid = table(
        ["Context", "KV bytes/token", "Bytes/token with weights", "Tokens/s ceiling"],
        [
            [
                str(row["context"]),
                str(fp16["kvBytesPerToken"]),
                str(row["bytesPerToken"]),
                rate(row["tokensPerSecondCeiling"]),
            ]
            for row in picked
        ],
    )
    lines.extend(["", grid, ""])
    return lines


def kv_section(kv: dict) -> str:
    lines = [
        "## KV cache",
        "",
        (
            f"Bytes per token = {kv['formula']}. "
            "The ceiling is bandwidth / (weight bytes + kv bytes per token * context)."
        ),
        f"Bandwidth {gb(kv['bandwidthBytesPerSecond'])} GB/s. {kv['weightBytesNote']}.",
        "",
    ]
    for model in kv["models"]:
        lines.extend(kv_model_block(model))
    return "\n".join(lines).rstrip()


def speculative_section(kv: dict) -> str:
    spec = kv["speculative"]
    rows = [
        [rate(row["acceptance"]), str(row["draftTokens"]), rate(row["expectedTokensPerStep"])]
        for row in spec["rows"]
    ]
    grid = table(["Acceptance", "Draft tokens", "Expected tokens per step"], rows)
    return "\n".join(
        ["## Speculative decoding", "", spec["note"], f"Source: {spec['source']}.", "", grid]
    )


def specs_section(specs: dict) -> str:
    grid = table(
        ["Chip", "Memory", "Bandwidth", "Dense FP16", "FLOP/byte"],
        [
            [
                chip["name"],
                chip["memory"],
                chip["bandwidthLabel"],
                f"{chip['denseFp16Tflops']} TFLOP/s",
                rate(chip["ridgeFlopsPerByte"]),
            ]
            for chip in specs["chips"]
        ],
    )
    lines = ["## Hardware specs", "", specs["bytesNote"], "", grid, ""]
    for chip in specs["chips"]:
        urls = ", ".join(source["url"] for source in chip["sources"])
        lines.append(f"{chip['name']}: {chip['sparsityNote']} Sources: {urls}.")
        lines.append("")
    for row in specs["omitted"]:
        lines.append(f"{row['name']} is not in the table. {row['reason']} Source: {row['source']}.")
    return "\n".join(lines)


def caveats() -> str:
    lines = [
        "## Caveats",
        "",
        "These bandwidth and FLOP figures are what this process achieved.",
        "The CPU name string comes from the guest and can be generic.",
        "Arithmetic intensity is compulsory traffic, not a hardware counter of DRAM bytes.",
        "The decode ceiling counts every parameter once per token.",
        "An embedding lookup does not read the whole table, so that ceiling is a little low.",
        "Dynamic int8 dequantizes inside the kernel, so fewer bytes may not mean more speed.",
        "The H100 dense FP16 number is half the product page's sparse number.",
        "The datasheet says sparse figures are twice the dense ones.",
        "That datasheet rounds the same chip to 2,000 TFLOP/s sparse and 3 TB/s.",
        "The RTX 4090 dense number is the FP16-accumulate tensor figure.",
        "FP32-accumulate is lower and is in specs.json.",
        "Speculative rows are not a measurement.",
        "The roofline matmul skips the small norm and bias tensors.",
        "It is not the attention score math.",
        "Reruns move the timed numbers. The arithmetic does not.",
    ]
    return "\n".join(lines)


def render(machine: dict, roof: dict, decode: dict, kv: dict, specs: dict) -> str:
    parts = [
        "# Memory-bound measurements",
        "",
        machine_section(machine),
        roofline_section(roof),
        decode_section(decode),
        batch_section(decode),
        kv_section(kv),
        speculative_section(kv),
        specs_section(specs),
        caveats(),
    ]
    return "\n\n".join(parts) + "\n"
