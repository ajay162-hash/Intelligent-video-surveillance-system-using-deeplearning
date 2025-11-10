# Intelligent Video Surveillance System (Ready-to-Run)

A Flask app that detects 14 types of anomalous/criminal activities in videos using a 3D CNN. This repo ships with pre-trained checkpoints so you can run the app without training.

## Quick start

1) Clone and enter the project

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

2) Create a virtual environment and activate it

- Windows (PowerShell):
```bash
python -m venv .venv
. .venv\Scripts\Activate.ps1
```
- macOS/Linux (bash/zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Install dependencies

```bash
pip install -r requirements.txt
```

4) Configure environment variables (optional but recommended)

Copy .env.example to .env and fill in values:

```bash
cp .env.example .env   # Windows PowerShell: copy .env.example .env
```

Environment file (.env) structure:

```dotenv
# Twilio (WhatsApp alerts)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
# Sender number formats:
# - whatsapp:+1415XXXXXXX (preferred); or
# - +1415XXXXXXX (the app will add whatsapp: automatically)
TWILIO_FROM_NUMBER=

# Gemini (Google Generative AI)
GOOGLE_API_KEY=
# Optional, defaults to gemini-2.0-flash
GEMINI_MODEL=gemini-2.0-flash

# Media hosting for WhatsApp (choose ONE path)
# 1) Public base URL to your uploads directory (must be public HTTPS)
MEDIA_BASE_URL=
# 2) Supabase public bucket (recommended)
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
SUPABASE_BUCKET=media
```

Notes:
- WhatsApp alerts are optional. If Twilio variables are not set, the app runs without sending alerts.
- If both `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set, uploaded media will be published via your Supabase Storage and used in alerts.
- If neither Supabase nor `MEDIA_BASE_URL` is configured, alerts are still sent but without media links.

5) Run the app

```bash
python app.py
```
Open http://localhost:5000 and upload a video.

## Pre-trained models

This repo includes ready-to-use checkpoints so you can run inference immediately:
- `trained_models_v2/best_model.pth` (default used by the app)
- `trained_models/best_model.pth` (alternate)

You can switch which model is used by editing `app.py` (value of `app.config['MODEL_PATH']`).

## API usage

POST a video to get a prediction:

```python
import requests

with open('your_video.mp4', 'rb') as f:
    r = requests.post('http://localhost:5000/api/predict', files={'video': f})
print(r.json())
```

## Training (optional)

A single training script is kept, aligned with the shipped checkpoints:

```bash
python train.py --train_dir data/Train --test_dir data/Test --epochs 50 \
  --batch_size 8 --accumulation_steps 4 --output_dir trained_models_v2
```

Tips:
- Images should be PNG frames organized by class. The script will verify structure.
- Class weights and label smoothing are available to handle imbalance.
- Best model is saved to `<output_dir>/best_model.pth` and training args to `<output_dir>/training_args.json`.

## Project structure

```
.
├── app.py                     # Flask app (serves web UI + REST API)
├── train.py                   # Training script (kept as the single canonical trainer)
├── models/
│   ├── advanced_model.py      # MaxAccuracyC3D architecture and fallbacks
│   └── cnn3d.py               # Baseline C3D
├── utils/
│   └── video_utils.py         # Video/frame helpers and visualization
├── trained_models/            # Pre-trained (kept) — best_model.pth is tracked
├── trained_models_v2/         # Pre-trained (default) — best_model.pth is tracked
├── static/                    # Web static assets (uploads etc.)
├── templates/                 # Auto-generated on first run if missing
├── requirements.txt
└── README.md
```

## Environment variables (reference)

- Twilio (WhatsApp): `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`
- Gemini: `GOOGLE_API_KEY`, `GEMINI_MODEL` (default: `gemini-2.0-flash`)
- Media publishing: `MEDIA_BASE_URL` or (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_BUCKET`)

## Large files (Git LFS)

This repo includes large model files. To avoid pushing raw blobs, Git LFS is configured in `.gitattributes` for these paths:
- `trained_models_v2/*.pth`
- `trained_models/*.pth`

Before your first push on a new machine:

```bash
git lfs install
```

Then commit and push as usual; Git will store pointers in the repo and upload the model to LFS.

## Troubleshooting

- Torch/torchvision wheels: If GPU CUDA build is unavailable, reinstall CPU wheels: `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`.
- OpenCV on Windows: If import fails, ensure Visual C++ Redistributable is installed.
- Large videos: Processing is clip-based; very long videos may take time. Start with shorter samples to validate the pipeline.

## License

For educational and research purposes only. Ensure compliance with local laws and regulations when using surveillance software.
