from typing import Any

from ce_explainer.config import PRESETS
from ce_explainer.report import HOOK_EPOCHS


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = [header, ["---"] * len(header), *rows]
    return ["| " + " | ".join(cells) + " |" for cells in lines]


def percent(value: float) -> str:
    return f"{100 * value:.1f}"


def accuracy_table(results: dict[str, Any]) -> list[str]:
    header = ["Preset", "Rate", "Epoch 1", "Epoch 5", "Epoch 30 mean", "Min", "Max"]
    rows = []
    for name in PRESETS:
        a = results["finalAccuracy"][name]
        cells = [a["epoch1Mean"], a["epoch5Mean"], a["mean"], a["min"], a["max"]]
        rows.append([name, f"{results['selectedRates'][name]:g}", *map(percent, cells)])
    return table(header, rows)


def sweep_table(results: dict[str, Any]) -> list[str]:
    rates = [row["rate"] for row in next(iter(results["sweep"].values()))]
    rows = [
        [name, *[percent(row["validationAccuracy"]) for row in results["sweep"][name]]]
        for name in PRESETS
    ]
    return table(["Preset", *[f"{rate:g}" for rate in rates]], rows)


def start_table(results: dict[str, Any]) -> list[str]:
    rows = [
        [
            key,
            f"{s['meanTopProbability']:.3f}",
            percent(s["wrongShare"]),
            percent(s["rightBelow001Share"]),
            f"{s['crossEntropy']:.3f}",
        ]
        for key, s in results["start"].items()
    ]
    header = ["Start", "Mean top p", "Wrong %", "p(label) < 0.01 %", "Test loss (nats)"]
    return table(header, rows)


def hook_table(results: dict[str, Any]) -> list[str]:
    hook = results["hook"]
    rows = []
    for epoch in HOOK_EPOCHS:
        cells = [str(epoch)]
        for loss in ("ce", "mse"):
            row = hook["trajectory"][loss][epoch]
            cells += [str(row["predicted"]), f"{row['pLabel']:.4g}"]
        rows.append(cells)
    header = ["Epoch", "CE predicts", "CE p(label)", "MSE predicts", "MSE p(label)"]
    intro = f"Test digit {hook['testIndex']}, label {hook['label']}."
    return [intro, "", *table(header, rows)]


def runtime_lines(results: dict[str, Any]) -> list[str]:
    rows = [[stage, f"{seconds:.1f}"] for stage, seconds in results["seconds"].items()]
    return table(["Stage", "Seconds"], rows) + ["", f"CPU: {results['cpu']}"]


def render_markdown(results: dict[str, Any]) -> str:
    sections = [
        ["# Report"],
        ["## Test accuracy (%), five seeds", "", *accuracy_table(results)],
        ["## Validation accuracy (%) by learning rate, seeds 0 to 2", "", *sweep_table(results)],
        ["## Starting point (seed 0, test set)", "", *start_table(results)],
        ["## Hook digit, confident start, seed 0", "", *hook_table(results)],
        ["## Run time", "", *runtime_lines(results)],
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
