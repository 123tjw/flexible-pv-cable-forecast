# Flexible PV Cable Forecasting Minimal Demo

A small standalone demonstration of PatchTST and MyPatchTST for cable-tension forecasting in flexible photovoltaic support systems.

## Files

- `models/` and `layers/`: the model definitions and their required dependencies.
- `demo_sample.csv`: a four-day excerpt of simulated data, containing 576 ten-minute records and 13 input variables plus timestamps.
- `demo_config.json`: input order, model settings and 6, 12 and 24 h task lengths.
- `demo_eval.py`: a single forward-pass demonstration on CPU.
- `requirements.txt`: reference numerical dependencies.

The sample is drawn from a parametrically generated synthetic benchmark, rather than field monitoring data. Wind components are derived from the original wind speed and direction. The final input channel is the forecast target, `tension_mid_mean_kN`.

## Quick start

Reference environment: Python 3.9 and PyTorch 2.0.1. Use an isolated environment, or retain an existing compatible PyTorch environment.

```bash
python -m pip install -r requirements.txt
python demo_eval.py
```

The default command runs both models for the 6 h task and reports input/output shapes, parameter counts and a finite-value check. Expected forecast shapes are `[1, 36, 13]` for PatchTST and `[1, 36, 1]` for MyPatchTST. Both provide a target-channel forecast with shape `[1, 36, 1]`.

Other examples:

```bash
python demo_eval.py --model MyPatchTST --horizon 24
python demo_eval.py --check-only
```

`--check-only` validates the sample format and task lengths without constructing a model or performing inference.

## Demonstration scope

The demo uses randomly initialized weights with seed 2026. It demonstrates model architecture and data interfaces; trained checkpoints and paper-performance evaluation are outside this example. Inputs are scaled using only the historical input window for the demonstration, rather than the full training-period statistics used in the study. The script does not report MSE or MAE.

## Data and code availability

The full ten-year dataset remains part of an ongoing research project and is not publicly available in full at present. This minimal sample is provided for inspecting the model architecture and data format. Requests concerning data supporting the manuscript should be directed to its corresponding author. Release of the complete experimental pipeline is planned following publication of the associated article.

## Attribution

PatchTST is based on [the official implementation](https://github.com/yuqinie98/PatchTST), and RevIN on [its original implementation](https://github.com/ts-kim/RevIN). The applicable upstream licence texts and attribution notes are included in `LICENSES/` and `THIRD_PARTY_NOTICES.txt`.
