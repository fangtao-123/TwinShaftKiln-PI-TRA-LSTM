# Deployment Guide

This document describes how to deploy, run, and publish the standardized **PI-TRA-LSTM Modular** repository.

The guide is written for two practical environments:

- local Windows 10 / PyCharm development;
- GPU server training.

## 1. Project Overview

PI-TRA-LSTM is a physics-informed temporal-attention LSTM model for lime-kiln forecasting. The codebase supports:

- data preprocessing and time-ordered splitting;
- traditional machine-learning baselines;
- deep-learning sequence baselines;
- physics-informed PI-LSTM and PI-TRA-LSTM;
- optional PSO calibration of physical parameters;
- benchmark reports, prediction files, figures, and segmented metrics.

The standard version intentionally removes the old `PI-QTRA-LSTM` naming. The main physics-informed attention model is:

```text
PI-TRA-LSTM
```

## 2. Recommended Repository Layout

```text
PI_TRA_LSTM_modular/
|-- README.md
|-- DEPLOYMENT.md
|-- requirements.txt
|-- config.py
|-- common.py
|-- data_process.py
|-- model_blocks.py
|-- models/
|   |-- __init__.py
|   `-- model_zoo.py
|-- physics.py
|-- train_torch.py
|-- plot_utils.py
|-- segment_eval.py
|-- sensitivity.py
|-- main.py
|-- run.py
|-- data_src/                  # Optional local raw CSV folder, not committed
|-- results/                   # Optional local outputs, not committed
`-- checkpoints/               # Optional local weights, not committed
```

For GitHub release, avoid committing private industrial data, model checkpoints, or large result folders.

## 3. Environment Requirements

### 3.1 Minimal CPU Environment

Use this for code checks, small runs, and debugging.

```bash
conda create -n pi-tra-lstm python=3.10 -y
conda activate pi-tra-lstm
pip install numpy pandas scikit-learn matplotlib torch
```

### 3.2 GPU Server Environment

Use this for formal experiments.

```bash
conda create -n pi-tra-lstm python=3.10 -y
conda activate pi-tra-lstm
pip install numpy pandas scikit-learn matplotlib
```

Install PyTorch according to your CUDA version. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 4. Data Preparation

Place the raw CSV file in a local folder, for example:

```text
PI_TRA_LSTM_modular/
`-- data_src/
    `-- limekiln.csv
```

Then update [config.py](config.py):

```python
CONFIG["original_csv"] = "data_src/limekiln.csv"
```

On Windows, absolute paths are also supported:

```python
CONFIG["original_csv"] = r"C:\Users\user1\Desktop\PI_TRA_LSTM_modular\data_src\limekiln.csv"
```

## 5. Configuration

Edit [config.py](config.py) before running.

### 5.1 Device

```python
CONFIG["device"] = "cuda" if torch.cuda.is_available() else "cpu"
```

For forced CPU debugging:

```python
CONFIG["device"] = "cpu"
```

### 5.2 Training Protocol

The default formal protocol is:

```python
CONFIG["base_epochs"] = 500
CONFIG["base_batch"] = 128
CONFIG["base_lr"] = 5e-4
CONFIG["base_patience"] = 20
CONFIG["lr_decay_step"] = 200
CONFIG["lr_decay_gamma"] = 0.5
```

This corresponds to:

- Adam optimizer;
- initial learning rate `5e-4`;
- batch size `128`;
- maximum `500` epochs;
- StepLR every `200` epochs with decay factor `0.5`;
- early stopping by validation loss.

### 5.3 Output Directories

```python
CONFIG["ckpt_dir"] = "checkpoints_bench_版本modelV21.0"
CONFIG["result_dir"] = "results_版本modelV21.0"
```

For GitHub-clean output, you can use:

```python
CONFIG["ckpt_dir"] = "checkpoints"
CONFIG["result_dir"] = "results"
```

## 6. Running in Windows 10 with PyCharm

1. Open the project folder in PyCharm.
2. Set the Python interpreter to the conda environment `pi-tra-lstm`.
3. Open [config.py](config.py) and set `CONFIG["original_csv"]`.
4. Open [main.py](main.py).
5. Right-click and choose **Run main**.

If imports fail when running [main.py](main.py), run from the parent folder with module mode:

```bash
python -m PI_TRA_LSTM_modular_标准版.run
```

For a GitHub release, it is recommended to rename the package folder to ASCII:

```text
PI_TRA_LSTM_modular
```

Then the command becomes:

```bash
python -m PI_TRA_LSTM_modular.run
```

## 7. Running on a GPU Server

Upload the repository and data file to the server:

```bash
git clone https://github.com/qiaojimei/Twin-Shaft-Lime-Kiln-PI-TRA-LSTM.git
cd Twin-Shaft-Lime-Kiln-PI-TRA-LSTM
```

Create environment:

```bash
conda create -n pi-tra-lstm python=3.10 -y
conda activate pi-tra-lstm
pip install -r requirements.txt
```

Edit [config.py](config.py):

```python
CONFIG["original_csv"] = "/path/to/limekiln.csv"
CONFIG["device"] = "cuda"
CONFIG["base_epochs"] = 500
```

Run:

```bash
python -m PI_TRA_LSTM_modular.run
```

## 8. Benchmark Models

The standard benchmark includes:

```text
Traditional ML:
  EN, MLP, SVM

Deep sequence models:
  LSTM, BiLSTM, GRU, CNN-LSTM, TPA-LSTM, TRA-LSTM

Physics-informed models:
  PI-LSTM, PI-TRA-LSTM
```

The old `PI-QTRA-LSTM` model name has been removed from the standard release.

## 9. Physics-Informed Components

PI-TRA-LSTM uses:

- prediction data loss;
- temperature-evolution physics residual;
- under-burning risk penalty;
- optional physics-parameter calibration by PSO.

If PSO calibration is enabled:

```python
CONFIG["do_pso_calibration"] = True
```

The best physical parameters are saved to:

```python
CONFIG["pso_outfile"]
```

If you do not want PSO:

```python
CONFIG["do_pso_calibration"] = False
```

## 10. Outputs

After running, outputs include:

```text
results/
|-- benchmark_report.csv
|-- run_*.json
|-- pred_*.csv
|-- predmeta_*.csv
|-- curve_*.csv
|-- lc_*.png
|-- err_*.png
|-- pred_*.png
|-- segment_metrics_*.csv
`-- comparison figures

checkpoints/
`-- *_best.pt
```

Important files:

- `benchmark_report.csv`: all model metrics;
- `pred_*.csv`: test predictions;
- `predmeta_*.csv`: predictions with index, mode, and quality metadata;
- `curve_*.csv`: training and validation loss curves;
- `segment_metrics_*.csv`: segmented evaluation.

## 11. Evaluation Metrics

The benchmark reports:

- R2;
- MSE;
- MAE;
- MAPE;
- RMSE;
- training time;
- parameter count.

Validation and test metrics are both recorded.

## 12. GitHub Release Checklist

Before publishing:

1. Rename package folder to ASCII if possible:

```text
PI_TRA_LSTM_modular
```

2. Remove private files:

```text
data_src/
results/
checkpoints/
*.pt
*.pkl
*.csv containing private production data
```

3. Add `.gitignore`:

```gitignore
__pycache__/
.DS_Store
*.pyc
data_src/
results*/
checkpoints*/
*.pt
*.pkl
```

4. Add `requirements.txt`.

Suggested content:

```text
numpy
pandas
scikit-learn
matplotlib
torch
```

5. Verify imports:

```bash
python -m PI_TRA_LSTM_modular.run
```

6. Verify no old QTRA implementation references remain in Python code:

```bash
grep -R "PI_QTRA\\|PI-QTRA\\|QTRA" --include="*.py" .
```

## 13. Common Issues

### 13.1 `ModuleNotFoundError`

Run from the parent folder:

```bash
python -m PI_TRA_LSTM_modular.run
```

or set PyCharm working directory to the parent of the package folder.

### 13.2 CUDA Is Not Available

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Install the PyTorch build matching your CUDA version.

### 13.3 CSV Encoding Error

The loader tries several encodings:

```text
utf-8, utf-8-sig, gbk, gb2312, utf-16, latin1
```

If the CSV still fails, convert it to UTF-8 with Excel or Python.

### 13.4 Missing Columns

The preprocessing code expects the configured feature columns. If your dataset has different Chinese column names, update the column mapping or rename the CSV headers before training.

### 13.5 Training Too Slow on CPU

For a quick check, temporarily set:

```python
CONFIG["base_epochs"] = 1
CONFIG["time_steps"] = [30]
CONFIG["horizons"] = [1]
```

Use GPU for formal 500-epoch experiments.

## 14. Suggested Citation

If this code is used in a paper, cite it as:

```bibtex
@misc{pi_tra_lstm_limekiln,
  title = {PI-TRA-LSTM: A Physics-Informed Temporal Attention LSTM Framework for Lime Kiln Forecasting},
  author = {Your Name},
  year = {2026},
  howpublished = {GitHub repository}
}
```
