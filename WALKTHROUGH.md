# VideoEmotion: Execution Walkthrough

## Introduction
This guide provides detailed instructions on how to execute each component of the VideoEmotion system directly from the terminal. It is intended for developers and administrators who need granular control over the system.

**Assumptions**:
*   You are in the project root directory (`VideoEmotion/`).
*   Your virtual environment is active (`venv\Scripts\activate`).
*   Dependencies are installed.

## Global CLI Conventions
*   All scripts should be run using `python` (or `venv\Scripts\python.exe`).
*   Most scripts support the `--help` flag to list all available options.

---

## 1. Backend API Server (`app/main.py`)

**Purpose**: Starts the FastAPI server, serving the REST API and the static file mounts. This is the main entry point for the "Production" mode.

**Basic Usage**:
```bash
python -m app.main
```
*   **URL**: `http://localhost:8000`
*   **Docs**: `http://localhost:8000/docs`

**Configuration**:
Controlled via `.env` file (HOST, PORT, RELOAD).

**Common Errors**:
*   `ModuleNotFoundError`: Ensure you run command as `python -m app.main`, NOT `python app/main.py`.

---

## 2. Offline Pipeline Runner (`src/offline/pipeline.py`)

**Purpose**: Manually executes the analysis pipeline on video files. Useful for batch processing or debugging specific videos without using the API.

**Path**: `src/offline/pipeline.py`

**Required Arguments**:
*   `--video <path>` OR `--all`: Specify a single video file path or process all videos in the configured data directory.

**Optional Arguments**:
*   `--fps <int>`: Frame extraction rate (default: 5).
*   `--project-root <path>`: Override project root path.
*   `--config <path>`: Override config.yaml location.
*   `--no-extract`: Skip frame extraction step.
*   `--no-detect`: Skip face detection step.
*   `--no-analyze`: Skip emotion analysis step.
*   `--no-summary`: Skip report generation.
*   `--summary-only`: Run ONLY the summary step (requires previous steps done).
*   `--visualize-only`: Run ONLY the result video creation.
*   `--force-visualize`: Overwrite existing visualization videos.

**Examples**:
```bash
# Process a single video
python src/offline/pipeline.py --video data/videos/interview.mp4

# Process ALL videos with specific FPS
python src/offline/pipeline.py --all --fps 10

# Only regenerate the visualization video
python src/offline/pipeline.py --video data/videos/interview.mp4 --visualize-only --force-visualize
```

**Outputs**:
*   Generates folders in `data/extracted_frames/`, `data/detected_faces/`, `output/emotion_results/`.
*   Final reports in `output/reports/`.
*   Visualizations in `output/visualizations/`.

---

## 3. Realtime Analysis (`src/realtime/realtime_analysis.py`)

**Purpose**: Launches a standalone webcam analysis session window. This runs independently of the API server.

**Path**: `src/realtime/realtime_analysis.py`

**Recommended Command**:
```bash
python src/realtime/realtime_analysis.py --camera-id 0
```

**Optional Arguments**:
*   `--camera-id <int>`: Webcam index (default: 0).
*   `--display-width <int>`: Resize the window width (e.g., 1280).
*   `--no-save-json`: Do NOT save analysis data to JSON.
*   `--no-save-video`: Do NOT record the session video.
*   `--no-visualize`: Don't draw bounding boxes/text on the video feed.
*   `--out-dir <path>`: Custom output directory (default: `output/realtime`).

**Examples**:
```bash
# Minimal UI, no recording
python src/realtime/realtime_analysis.py --no-save-video --no-save-json

# High-res display
python src/realtime/realtime_analysis.py --display-width 1920
```

**Controls**:
*   Press **`q`** in the window to stop the session.

---

## 4. Testing (`pytest`)

**Purpose**: Execute the automated test suite to verify system integrity.

**Usage**:
```bash
# Run all tests
pytest

# Verbose mode (recommended)
pytest -v
```

**Troubleshooting**:
*   `ImportError`: Ensure you are running `pytest` from the root `VideoEmotion/` folder so python path resolution works correctly.

---

## Troubleshooting Guide

### "ModuleNotFoundError: No module named 'src'"
*   **Cause**: Python cannot find the `src` package.
*   **Fix**: Always run scripts from the **root directory** (`VideoEmotion/`). Do not `cd` into `src/`.
*   **Fix 2**: Use `python -m src.offline.pipeline` style if applicable, or ensure `sys.path` is correct (the scripts handle this, but root execution is key).

### "ffmpeg not found"
*   **Cause**: `ffmpeg` is missing from your system PATH.
*   **Fix**: Install FFmpeg and add the `bin` folder to your Windows Environment Variables.

### "Permission denied" on output folders
*   **Cause**: Scripts cannot write to `output/`.
*   **Fix**: Ensure no other process (like a video player or the API server) has the files locked. Close generic video players.

### Config issues
*   **Cause**: `config.yaml` not found.
*   **Fix**: The scripts look for `config.yaml` in the running directory or parent. Ensure it exists in the root.
