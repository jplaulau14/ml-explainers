# cross-entropy

## What this is

Companion code for the essay [Why do classifiers train on cross-entropy loss?](https://www.patslaurel.com/writing/why-cross-entropy).
It trains a small digit classifier with cross-entropy and with squared error, from a gentle start and from a deliberately overconfident one, and exports the training runs as JSON for the essay's widgets.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this folder:

```bash
uv sync
uv run ce-explainer all
```

This picks a learning rate for each preset, trains every preset with five seeds, and rewrites `out/`.
It took about 14 seconds on an Intel Xeon CPU (13 seconds of it in the pipeline, most of that in the learning-rate sweep). No GPU, download, API key or login is needed: the dataset ships with scikit-learn.
Run one step at a time with `sweep`, `train`, `export` and `report`.
`uv run pytest` checks the math and the committed files without retraining.

## What you get

- `out/hero.json`: the essay's first frame. Test labels, the confident start's stats, both confident-start models' test accuracy per epoch, and every test digit's right-class probability at epoch 0, byte-encoded.
- `out/field.json`: the same byte encoding for all 450 test digits at all 31 epochs, for both confident-start models (seed 0). Each byte is `128 * correct + min(127, round(8 * -log10(p)))`, where p is the probability the model gives the right class and `correct` is 1 when the model's top class is right. Bytes run epoch by epoch and are stored as base64. Probabilities at or below about 1e-16 share the darkest level.
- `out/replay.json`: all four linear presets: test accuracy and loss per epoch, five-seed accuracy curves, probabilities for 10 fixed test digits per epoch, and gradient sizes on the logits grouped by how much probability the right class gets.
- `out/likelihood.json`: the gentle cross-entropy run's probability for the right class on all 450 test digits at 8 epochs, plus the 20 hardest digits.
- `out/results.json`: learning-rate sweep, five-seed accuracies, starting-point stats, the hook digit's trajectory and run time as numbers.
- `out/report.md`: the same results as tables.

## Setup

- **Data:** scikit-learn's `load_digits`, 1,797 scans of handwritten digits, 8 × 8 pixels with values 0 to 16, scaled to [0, 1]. Stratified splits with seed 0: 1,077 train, 270 validation, 450 test.
- **Model:** softmax regression, one linear layer from 64 pixels to 10 logits, 650 parameters. The `mlp-` presets add one hidden layer of 32 tanh units as a robustness check.
- **Losses:** cross-entropy, −log p of the right class. Squared error, ½ Σ (p − y)² on the softmax probabilities against the one-hot label.
- **Training:** plain minibatch SGD, batch size 32, 30 epochs, float64, hand-written gradients.
- **Starts:** gentle draws the output weights with standard deviation 0.01. Confident uses 4.0. For a given seed the confident weights are exactly the gentle weights times 400.
- **Learning rate:** each preset gets its own, picked from 0.01 to 1000 in half-decade steps by mean validation accuracy over seeds 0 to 2. Ties go to the smaller rate. The test set is used only for reporting.

## Results

Test accuracy (%) after 30 epochs, mean and range over seeds 0 to 4:

| Preset | Learning rate | Epoch 1 | Epoch 5 | Epoch 30 | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| confident-ce | 100 | 74.8 | 91.0 | 95.1 | 93.8 | 95.8 |
| confident-mse | 30 | 23.3 | 38.7 | 58.1 | 47.8 | 73.6 |
| gentle-ce | 1 | 88.9 | 95.6 | 96.9 | 96.4 | 97.3 |
| gentle-mse | 3 | 84.8 | 95.1 | 96.6 | 96.4 | 96.9 |
| mlp-confident-ce | 0.03 | 79.6 | 92.1 | 96.6 | 95.8 | 97.3 |
| mlp-confident-mse | 0.03 | 27.5 | 47.6 | 66.7 | 38.2 | 85.1 |
| mlp-gentle-ce | 1 | 85.1 | 92.5 | 97.7 | 97.1 | 98.2 |
| mlp-gentle-mse | 3 | 74.1 | 91.6 | 97.7 | 97.3 | 98.0 |

Where each start begins, seed 0, on the test set:

| Start | Mean top probability | Wrong | p(right class) < 0.01 | Test loss (nats) |
| --- | --- | --- | --- | --- |
| gentle | 0.108 | 88.7% | 0.0% | 2.304 |
| confident | 0.964 | 88.7% | 88.0% | 32.87 |

From the gentle start the two losses tie. From the confident start squared error stalls: at epoch 0, 864 of the 1,077 training digits give the right class less than 1e-8, and for those the squared-error gradient on the logits averages 0.019 against 1.394 for cross-entropy. After 30 epochs of squared-error training, 530 digits are still in that bucket.

The plan picked a hook digit before training: test digit 12, an 8, the most confidently wrong of the 10 fixed digits. The confident start is certain it is a 9. With cross-entropy the model calls it an 8 at epochs 10, 14, 16, 17 and 24 to 28, and not at epoch 30; with squared error it never does. Both models end that run certain it is a 1, so the essay's opening shows all 450 test digits (`field.json`) instead of one, and digit 12 appears in the replay as an example of how a learning rate of 100 makes cross-entropy flip back and forth.

## How this maps to the essay

- Softmax outputs a distribution: `tests/test_model.py::test_claim_softmax_is_a_distribution`
- Adding a constant to every logit changes nothing: `tests/test_model.py::test_claim_shift_invariance`
- The cross-entropy gradient on the logits is p − y: `tests/test_model.py::test_claim_ce_gradient_is_p_minus_y` and `::test_claim_binary_ce_gradient_is_p_minus_y`
- The squared-error gradient through softmax and sigmoid: `tests/test_model.py::test_claim_mse_gradient_formula`
- Through a sigmoid it never exceeds 4/27: `tests/test_model.py::test_claim_sigmoid_mse_gradient_bound`
- Log-sum-exp keeps softmax finite at 1000: `tests/test_model.py::test_claim_logsumexp_is_stable`
- Cross-entropy is entropy plus KL divergence: `tests/test_info.py::test_claim_cross_entropy_is_entropy_plus_kl`
- With one-hot labels it is the negative log-likelihood: `tests/test_info.py::test_claim_one_hot_cross_entropy_is_nll`
- Bits and nats: `tests/test_info.py::test_claim_bits_and_nats`
- A fresh 10-class model starts at ln 10: `tests/test_results.py::test_claim_gentle_start_is_ln10`
- From the confident start cross-entropy wins by at least 20 points: `tests/test_results.py::test_claim_confident_start_gap`
- From the gentle start the two are within 1 point: `tests/test_results.py::test_claim_gentle_tie`
- Learning rates are picked without looking at the test set: `tests/test_results.py::test_claim_lr_chosen_on_validation`
- The hook digit follows the rule fixed before training: `tests/test_export.py::test_hook_digit_rule`
- The byte encoding of the test-set field round-trips and matches the reported accuracies: `tests/test_field.py`

## Notes

- The confident start is a deliberate setup to show the failure. It is not how anyone initializes a classifier.
- The squared error here is measured after the softmax. Hui and Belkin apply the square loss to raw outputs with no softmax and find it competitive; this code does not test that variant.
- Other conventions for squared error drop the ½ or average over classes. That rescales the gradient by a constant, which the learning-rate sweep absorbs.
- At learning rate 100 the confident cross-entropy run reaches 95.1% accuracy but its test loss stays at 2.08 nats, because the few digits it gets wrong it gets very confidently wrong.
- `mlp-` presets are reported here only. The essay's widgets use the linear model.
- Reruns on the same machine give byte-identical files. Other CPUs and BLAS builds can differ in the last digits.

## Sources

- Glorot and Bengio 2010, Understanding the Difficulty of Training Deep Feedforward Neural Networks. [PMLR 9:249-256](https://proceedings.mlr.press/v9/glorot10a.html)
- Golik, Doetsch and Ney 2013, Cross-Entropy vs. Squared Error Training: a Theoretical and Experimental Comparison. [Interspeech 2013](https://www.isca-archive.org/interspeech_2013/golik13_interspeech.html)
- Hui and Belkin 2021, Evaluation of Neural Architectures Trained with Square Loss vs Cross-Entropy in Classification Tasks. [arXiv:2006.07322](https://arxiv.org/abs/2006.07322)
- scikit-learn, [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), a copy of the test set of the UCI Optical Recognition of Handwritten Digits dataset.
