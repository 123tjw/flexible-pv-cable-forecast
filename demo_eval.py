"""Run a small model-interface demonstration, or validate files with --check-only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def read_inputs(config, sample_path, horizon):
    names = config['input_features']
    case = config['cases'][str(horizon)]
    rows, timestamps = [], []
    with sample_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['timestamp'] + names:
            raise ValueError('CSV columns must be timestamp followed by the configured 13 inputs')
        for record in reader:
            current = datetime.fromisoformat(record['timestamp'])
            if timestamps and (current - timestamps[-1]).total_seconds() != 600:
                raise ValueError('Expected uninterrupted ten-minute sample timestamps')
            values = [float(record[name]) for name in names]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Sample contains a nonfinite input')
            timestamps.append(current)
            rows.append(values)
    required = case['seq_len'] + case['pred_len']
    if len(rows) < required:
        raise ValueError('This task requires at least ' + str(required) + ' sample rows')
    if len(names) != 13 or names[-1] != 'tension_mid_mean_kN':
        raise ValueError('Expected 13 inputs with mean midspan tension in the last channel')
    return rows, timestamps, case


def run_demo(config, rows, timestamps, case, selected):
    # Numerical libraries and model sources are reached only in actual demo mode.
    import numpy as np
    import torch

    torch.set_num_threads(1)
    history = np.asarray(rows[:case['seq_len']], dtype=np.float64)
    mean = history.mean(axis=0, keepdims=True)
    scale = history.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    batch = torch.from_numpy(((history - mean) / scale).astype(np.float32)).unsqueeze(0)
    names = ['PatchTST', 'MyPatchTST'] if selected == 'both' else [selected]
    results = []
    for name in names:
        torch.manual_seed(config['seed'])
        args = dict(config['common_model_args'])
        args.update(case)
        args.update(config['models'][name])
        module = importlib.import_module('models.' + name)
        model = module.Model(SimpleNamespace(**args)).float().cpu().eval()
        with torch.inference_mode():
            forecast = model(batch)
        if not isinstance(forecast, torch.Tensor):
            raise RuntimeError('Expected a forecast tensor, got ' + str(type(forecast)))
        channels = 13 if name == 'PatchTST' else 1
        expected = (1, case['pred_len'], channels)
        if tuple(forecast.shape) != expected:
            raise RuntimeError(name + ': expected ' + str(expected) + ', got ' + str(tuple(forecast.shape)))
        if not bool(torch.isfinite(forecast).all().item()):
            raise RuntimeError(name + ': nonfinite forecast values')
        target = forecast[:, :, -1:]
        results.append({'model': name, 'input_shape': list(batch.shape),
                        'forecast_shape': list(forecast.shape),
                        'target_forecast_shape': list(target.shape),
                        'all_values_finite': True,
                        'parameters': sum(parameter.numel() for parameter in model.parameters())})
        del model, forecast, target
    return {'mode': 'MODEL_INTERFACE_DEMO', 'device': 'cpu', 'weights': 'random_initialization',
            'seed': config['seed'], 'torch': torch.__version__, 'horizon_hours': case['pred_len'] // 6,
            'forecast_origin': timestamps[case['seq_len']].isoformat(sep=' '),
            'input_normalization': 'Mean and population SD of the historical input window only',
            'models': results,
            'purpose': 'Inspect model architecture, input format and forward output shapes',
            'paper_performance_metrics_computed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('both', 'PatchTST', 'MyPatchTST'), default='both')
    parser.add_argument('--horizon', type=int, choices=(6, 12, 24), default=6)
    parser.add_argument('--sample', type=Path, default=ROOT / 'demo_sample.csv')
    parser.add_argument('--check-only', action='store_true', help='Check sample/configuration without importing models or numerical libraries')
    cli = parser.parse_args()
    config = json.loads((ROOT / 'demo_config.json').read_text(encoding='utf-8'))
    rows, timestamps, case = read_inputs(config, cli.sample, cli.horizon)
    if cli.check_only:
        print(json.dumps({'mode': 'CHECK_ONLY', 'sample_rows': len(rows), 'input_channels': len(config['input_features']),
            'horizon_hours': cli.horizon, 'seq_len': case['seq_len'], 'pred_len': case['pred_len'],
            'required_sample_rows': case['seq_len'] + case['pred_len'],
            'sample_start': timestamps[0].isoformat(sep=' '), 'sample_end': timestamps[-1].isoformat(sep=' '),
            'model_constructed': False, 'forward_pass_performed': False, 'training_started': False}, indent=2))
        return
    print(json.dumps(run_demo(config, rows, timestamps, case, cli.model), indent=2))


if __name__ == '__main__':
    main()
