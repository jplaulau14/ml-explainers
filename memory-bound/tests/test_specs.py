from mb_explainer.specs import a100, chips, h100, omitted, rtx4090


def test_claim_h100_dense_is_half_the_sparse_page_figure() -> None:
    chip = h100()
    assert chip["fp16TensorTflopsWithSparsity"] == 1979
    assert chip["denseFp16Tflops"] == 989.5
    assert chip["datasheet"]["denseFp16Tflops"] == 1000
    assert chip["datasheet"]["fp16TensorTflopsWithSparsity"] == 2000


def test_claim_a100_printed_pair() -> None:
    chip = a100()
    assert chip["denseFp16Tflops"] == 312
    assert chip["fp16TensorTflopsWithSparsity"] == 624
    assert chip["bandwidthBytesPerSecond"] == 2039 * 10**9


def test_claim_rtx4090_printed_pair() -> None:
    chip = rtx4090()
    assert chip["denseFp16Tflops"] == 330.3
    assert chip["fp16TensorTflopsWithSparsity"] == 660.6
    assert chip["denseFp16Fp32AccumulateTflops"] == 165.2
    assert chip["bandwidthBytesPerSecond"] == 1008 * 10**9


def test_claim_ridge_uses_dense_flops_and_bandwidth() -> None:
    for chip in chips():
        flops = chip["denseFp16Tflops"] * 10**12
        assert chip["denseFp16FlopsPerSecond"] == flops
        assert chip["ridgeFlopsPerByte"] == flops / chip["bandwidthBytesPerSecond"]
        assert chip["sources"]
        assert all(source["url"].startswith("https://") for source in chip["sources"])


def test_claim_apple_is_omitted_without_a_flop_number() -> None:
    row = omitted()[0]
    assert row["name"] == "Apple M2 Ultra"
    assert "flop" not in row
    assert row["bandwidthLabel"] == "800GB/s"
    assert row["source"].startswith("https://")
