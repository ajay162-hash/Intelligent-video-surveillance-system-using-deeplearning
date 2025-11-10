# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common commands

Python environment and dependencies

```bash
# Create and activate a virtual environment
python -m venv .venv
# On Windows PowerShell
. .venv/Scripts/Activate.ps1
# On bash/zsh
# source .venv/bin/activate

# Install runtime dependencies
pip install -r requirements.txt
```

Smoke check (import and instantiate models)

```bash
python - <<'PY'
from models.cnn3d import C3D
from models.advanced_model import MaxAccuracyC3D
print('C3D:', C3D(num_classes=14))
print('MaxAccuracyC3D:', MaxAccuracyC3D(num_classes=14))
PY
```

Linting and formatting

- No linter/formatter configuration is present (e.g., ruff/flake8/black configs are not in the repo).

Testing

- No tests or test configuration are present.
- If/when pytest is added, you can run a single test like:

```bash
pytest tests/test_file.py::TestClass::test_case -q
```

Build/packaging

- No packaging/build tooling is configured (no `pyproject.toml`, `setup.cfg`, or `Makefile`).

## High-level architecture

This repository contains core components for a video classification system built on 3D CNNs. The broader app-level pieces referenced in the README (e.g., Flask server `app.py`, training script `train.py`, dataset scripts) are not present here; what you have is the model code and a video utility module.

Modules

- `models/`
  - `cnn3d.py`: Baseline C3D architecture for video classification. Convolutional 3D blocks with pooling, followed by fully connected layers (`fc6`, `fc7`, `fc8`). Expects 5D tensors and produces logits for `num_classes` (defaults to 14).
  - `advanced_model.py`: Enhanced model components:
    - `SpatialTemporalAttention`: channel-, spatial-, and temporal-attention for 3D feature maps.
    - `EnhancedResBlock3D`: residual 3D conv block with optional attention and dropout.
    - `MaxAccuracyC3D` (aliased as `AdvancedVideoModel`): stem + stacked residual layers with attention, global attention pooling, adaptive pooling, multi-layer MLP head. Handles inputs shaped `(B, C, T, H, W)` and will permute if given `(B, T, C, H, W)`.
- `utils/video_utils.py`
  - `extract_frames_from_video(video_path, sample_rate=10)`: uses OpenCV to read a video and subsample frames.
  - `process_video_frames(frames, clip_len=16, overlap=0.5)`: windows frames into overlapping clips suitable for model input.
  - `save_prediction_visualization(video_path, predictions, confidences, class_names, output_path=None)`: overlays class names and confidences onto frames and writes a new video. Tries H.264 (`avc1`) first, then falls back to `mp4v`, finally to MJPG/AVI if needed.
  - `export_anomaly_clip(video_path, center_frame, seconds=3.0, out_dir=None, ...)`: exports a short MP4 clip around a frame index; similar codec fallbacks.

Typical data flow

1. Load a video and extract frames (`utils.extract_frames_from_video`).
2. Convert frames to fixed-length clips (`utils.process_video_frames`).
3. Preprocess clips to tensors shaped `(B, C, T, H, W)` and feed into either `C3D` or `MaxAccuracyC3D`.
4. Map logits to class labels; optionally render results back to a video (`utils.save_prediction_visualization`) or export a short clip around an event (`utils.export_anomaly_clip`).

## Notes from README

- The README describes a larger system: dataset automation (`download_dataset.py`), preprocessing (`preprocess_data.py`), training (`train.py`), a Flask web app (`app.py`), demo generation, and WhatsApp/Twilio alerts via environment variables (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`).
- Those scripts and the web server are not present in this repository snapshot. Commands in the README that reference them will not work unless those files are added.

## Gaps and alignment

- Missing pieces compared to README: `app.py`, `train.py`, dataset scripts, tests, and any config for linting/formatting/type-checking.
- When extending this repo, prefer adding tests (e.g., with pytest) around `utils` and `models` I/O contracts (input shapes, dtype, video codec fallbacks) and wiring up a lightweight inference script that demonstrates end-to-end clip creation and model inference.
