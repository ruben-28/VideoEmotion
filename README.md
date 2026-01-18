# VideoEmotion: Advanced Emotion Analysis System

**VideoEmotion** is a comprehensive solution for analyzing human emotions in video content. It leverages state-of-the-art computer vision models (MediaPipe, HSEmotion) to detect faces and classify emotions in both offline video files and realtime webcam streams. The system provides a robust FastAPI backend, a modular architecture, and a modern web dashboard for administration.

## 1. Project Overview

*   **Goal**: To provide accurate, granular emotion analysis for researchers and developers.
*   **Core Functions**:
    *   **Offline Analysis**: Batch process video files to extract frames, detect faces, analyze emotions, and generate detailed reports.
    *   **Realtime Analysis**: Live webcam stream analysis with instant feedback and recording capabilities.
    *   **API Management**: A RESTful API to manage video inventory, pipeline jobs, and system statistics.
*   **Target Users**: CV Researchers, Emotion AI Developers, and System Administrators.

## 2. Key Features

*   **Dual-Mode Processing**: Seamlessly switch between processing archived footage and live streams.
*   **Modular Pipeline**: Configurable stages (Extraction -> Detection -> Analysis -> Reporting).
*   **Robust Architecture**:
    *   **FastAPI Backend**: High-performance, async-ready API.
    *   **Dependency Injection**: Modular and testable core services (`VideoManager`, `PipelineExecutor`).
    *   **Data Persistence**: JSON-based metadata and analysis results.
*   **Dashboard**: Admin interface for full system control (Next.js frontend).
*   **Container-Ready**: Designed for easy deployment with isolated environments.

## 3. Project Architecture

The project follows a Clean Architecture approach with a clear separation between the API layer, Core business logic, and dedicated Processing workers.

```mermaid
graph TD
    User[User / Dashboard] -->|HTTP REST| API[FastAPI (app/)]
    API -->|DI| VM[VideoManager]
    API -->|DI| PE[PipelineExecutor]
    API -->|DI| TM[TrashManager]
    
    subgraph Core Logic
        VM --> Scanner[VideoScanner]
        VM --> Store[MetadataStore]
        PE -->|Subprocess| OfflineWorker[src/offline/pipeline.py]
    end
    
    subgraph Processing
        OfflineWorker -->|1. Extract| FFmpeg
        OfflineWorker -->|2. Detect| MediaPipe
        OfflineWorker -->|3. Analyze| HSEmotion
        OfflineWorker -->|4. Report| JSON/CSV
    end
```

*   **API Layer**: Handles request validation, routing, and response formatting (Pydantic schemas).
*   **Service Layer**: Orchestrates complex logic (e.g., launching background jobs, calculating stats).
*   **Worker Layer**: Independent scripts for heavy lifting, ensuring the API remains responsive.

## 4. Folder Structure

```text
VideoEmotion/
├── app/                  # FastAPI Application
│   ├── routers/          # Modular API Endpoints (videos, trash, pipeline...)
│   ├── config.py         # Configuration Loader (Env + YAML)
│   ├── dependencies.py   # Dependency Injection Setup
│   ├── main.py           # Application Entry Point
│   └── schemas/          # Pydantic Data Models
├── src/                  # Core Logic & Workers
│   ├── core/             # Business Logic (VideoManager, Models...)
│   ├── offline/          # Offline Pipeline Scripts
│   ├── realtime/         # Realtime Analysis Scripts
│   └── utils/            # Shared Utilities
├── data/                 # Input Data (Videos, Frames, etc.)
├── output/               # Generated Results (Reports, JSONs)
├── tests/                # Automated Test Suite
├── config.yaml           # Project Configuration
├── .env                  # Server Environment Variables (Secrets)
└── requirements.txt      # Python Dependencies
```

## 5. User Interfaces

### REST API
The primary interface is the auto-generated Swagger UI, allowing direct interaction with all system endpoints.
*   Access: `http://localhost:8000/docs`
*   Capabilities: Upload/delete videos, start pipelines, view realtime logs, check system health.

*(Placeholder: Insert Swagger UI Screenshot Here)*

### Admin Dashboard (Frontend)
A Next.js dashboard consumes the API for a user-friendly experience.
*   Capabilities: Visual gallery of videos, drag-and-drop uploads, realtime monitoring graphs.

## 6. Interesting Code Highlights

### Dependency Injection (app/dependencies.py)
We use `lru_cache` to provide singleton services, ensuring efficient resource usage and easy testing mocking.
```python
@lru_cache()
def get_video_manager() -> VideoManager:
    return VideoManager(
        scanner=VideoScanner(...),
        store=MetadataStore(...),
        stats_calculator=StatsCalculator()
    )
```

### Pipeline Orchestration (src/offline/pipeline.py)
Accepts extensive CLI arguments to control the pipeline flow, enabling fine-grained control for debugging or batch processing.
```python
# Facade pattern for pipeline steps
if not args.no_extract:
    extract_frames(...)
if not args.no_detect:
    detect_faces(...)
if not args.no_analyze:
    analyze_emotions(...)
```

### Realtime Loop (src/realtime/realtime_analysis.py)
Runs an optimized OpenCV loop that handles capture, inference, and visualization/recording simultaneously.
```python
# Optimized capture + inference loop
while True:
    ok, frame = cap.read()
    # ... Face Detection ...
    # ... Emotion Inference ...
    # ... Result Visualization ...
    if do_save_video:
        video_writer.write(frame)
```

## 7. Installation Guide

**Requirements**: Python 3.10+

1.  **Clone & Venv**:
    ```bash
    git clone https://github.com/your-org/VideoEmotion.git
    cd VideoEmotion
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install pydantic-settings pytest
    ```

3.  **Configuration**:
    *   Copy `.env.example` to `.env` and set `HOST`, `PORT`, etc.
    *   Verify paths in `config.yaml`.

## 8. Running the Project

### Start Backend API
```bash
venv\Scripts\python -m app.main
```
Server will start at `http://localhost:8000`.

### Realtime Analysis (Standalone)
You can run realtime analysis without the API for testing:
```bash
venv\Scripts\python src/realtime/realtime_analysis.py --camera-id 0
```

## 9. Configuration

*   **`.env`**: Server-specific settings (Network, Security).
    *   `HOST`: Service binding IP.
    *   `ALLOWED_ORIGINS`: CORS settings.
*   **`config.yaml`**: Project logic settings.
    *   `paths`: Directory locations (data/videos, output/reports).
    *   `emotion_analysis`: Model thresholds and parameters.

## 10. Testing

Run the full test suite using `pytest`.
```bash
# Run all tests
venv\Scripts\pytest -v

# Run specific file
venv\Scripts\pytest tests/test_video_manager.py
```
Coverage includes unit tests for the Logic Layer and integration tests for the API Layer.

## 11. Limitations & Known Issues
*   **Platform**: Primarily tested on Windows (path handling).
*   **Dependencies**: MediaPipe dependency management can be tricky on some Linux distros.
*   **Performance**: Realtime FPS depends heavily on CPU/GPU capabilities.

## 12. Future Improvements
*   **Dockerization**: Full `docker-compose` setup.
*   **GPU Acceleration**: Explicit CUDA support for HSEmotion.
*   **Auth**: Add JWT authentication for API endpoints.

## 13. License & Credits

*   **HSEmotion**: High-performance emotion recognition library.
*   **MediaPipe**: Google's framework for face detection.
*   **FastAPI**: Modern, fast web framework for Python.

(c) 2026 VideoEmotion Team. Private/Internal License.
