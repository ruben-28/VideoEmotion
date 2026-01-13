# """
# Pipeline Runner Component for VideoEmotion Dashboard.
# Allows running the pipeline from the UI with configuration options.
# """

# import streamlit as st
# import requests
# import time
# from typing import Optional


# API_BASE = st.session_state.get("api_base", "http://localhost:8000")


# def render_pipeline_runner():
#     """Render the pipeline runner section"""
#     st.header("⚙️ Pipeline Runner")
    
#     # Video selection
#     video_name = st.session_state.get("pipeline_video", "")
    
#     st.write("Configure and run the emotion analysis pipeline for a video.")
#     st.markdown("---")
    
#     # Video input
#     video_input = st.text_input(
#         "Video Name",
#         value=video_name,
#         placeholder="e.g., my_video.mp4",
#         help="Enter the video filename (must exist in data/videos/)"
#     )
    
#     st.markdown("### Pipeline Configuration")
    
#     # Configuration options
#     col1, col2 = st.columns(2)
    
#     with col1:
#         fps = st.number_input("FPS", min_value=1, max_value=30, value=5, help="Frames per second to extract")
#         smoothing = st.checkbox("Smoothing", value=True, help="Apply temporal smoothing to emotions")
#         tta = st.checkbox("TTA (Test-Time Augmentation)", value=True, help="Use test-time augmentation for better accuracy")
#         visualize = st.checkbox("Visualize Results", value=True, help="Create annotated video with emotions")
    
#     with col2:
#         backend = st.selectbox("Backend", options=["hsemotion", "deepface"], index=0, help="Emotion detection backend")
#         export_bboxes = st.checkbox("Export Bounding Boxes", value=True, help="Export bounding box data")
#         overwrite = st.checkbox("Overwrite Existing", value=False, help="Overwrite existing results")
    
#     st.markdown("### Advanced Options")
    
#     with st.expander("Skip Steps"):
#         no_extract = st.checkbox("Skip Frame Extraction", value=False)
#         no_detect = st.checkbox("Skip Face Detection", value=False)
#         no_analyze = st.checkbox("Skip Emotion Analysis", value=False)
#         no_summary = st.checkbox("Skip Summary Generation", value=False)
#         no_visualize = st.checkbox("Skip Visualization", value=False)
    
#     st.markdown("---")
    
#     # Start pipeline button
#     if st.button("▶️ Start Pipeline", type="primary", use_container_width=True):
#         if not video_input:
#             st.error("Please enter a video name")
#         else:
#             start_pipeline(
#                 video_input,
#                 fps=fps,
#                 smoothing=smoothing,
#                 tta=tta,
#                 backend=backend,
#                 visualize=visualize,
#                 export_bboxes=export_bboxes,
#                 overwrite=overwrite,
#                 no_extract=no_extract,
#                 no_detect=no_detect,
#                 no_analyze=no_analyze,
#                 no_summary=no_summary,
#                 no_visualize=no_visualize
#             )
    
#     st.markdown("---")
    
#     # Show active jobs
#     render_active_jobs()


# def start_pipeline(
#     video_name: str,
#     fps: int = 5,
#     smoothing: bool = True,
#     tta: bool = True,
#     backend: str = "hsemotion",
#     visualize: bool = True,
#     export_bboxes: bool = True,
#     overwrite: bool = False,
#     no_extract: bool = False,
#     no_detect: bool = False,
#     no_analyze: bool = False,
#     no_summary: bool = False,
#     no_visualize: bool = False
# ):
#     """Start a pipeline job"""
#     try:
#         payload = {
#             "video_name": video_name,
#             "options": {
#                 "fps": fps,
#                 "smoothing": smoothing,
#                 "tta": tta,
#                 "backend": backend,
#                 "visualize": visualize,
#                 "export_bboxes": export_bboxes,
#                 "overwrite": overwrite,
#                 "no_extract": no_extract,
#                 "no_detect": no_detect,
#                 "no_analyze": no_analyze,
#                 "no_summary": no_summary,
#                 "no_visualize": no_visualize,
#             }
#         }
        
#         response = requests.post(f"{API_BASE}/api/pipeline/run", json=payload, timeout=10)
#         response.raise_for_status()
        
#         result = response.json()
#         job_id = result["job_id"]
        
#         st.success(f"✅ Pipeline started! Job ID: {job_id}")
#         st.session_state.active_job_id = job_id
        
#         # Auto-refresh to show progress
#         time.sleep(1)
#         st.rerun()
    
#     except Exception as e:
#         st.error(f"Failed to start pipeline: {e}")


# def render_active_jobs():
#     """Render active and recent pipeline jobs"""
#     st.markdown("### 📋 Pipeline Jobs")
    
#     try:
#         response = requests.get(f"{API_BASE}/api/pipeline/jobs?limit=10", timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         jobs = data["jobs"]
        
#         if not jobs:
#             st.info("No pipeline jobs yet")
#             return
        
#         for job in jobs:
#             render_job_card(job)
    
#     except requests.exceptions.ConnectionError:
#         st.error("❌ Cannot connect to API")
#     except Exception as e:
#         st.error(f"Failed to load jobs: {e}")


# def render_job_card(job: dict):
#     """Render a single job card"""
#     job_id = job["job_id"]
#     status = job["status"]
    
#     # Status colors and emojis
#     status_config = {
#         "pending": {"emoji": "⏳", "color": "blue"},
#         "running": {"emoji": "▶️", "color": "orange"},
#         "done": {"emoji": "✅", "color": "green"},
#         "error": {"emoji": "❌", "color": "red"},
#         "cancelled": {"emoji": "🚫", "color": "gray"}
#     }
    
#     config = status_config.get(status, {"emoji": "❓", "color": "gray"})
    
#     with st.expander(f"{config['emoji']} {job['video_name']} - {status.upper()}", expanded=(status == "running")):
#         col1, col2 = st.columns([3, 1])
        
#         with col1:
#             st.caption(f"**Job ID:** {job_id}")
#             st.caption(f"**Status:** {status.title()}")
            
#             if job.get("progress"):
#                 progress = job["progress"]
#                 st.progress(progress["percent"] / 100.0)
#                 st.caption(f"Step {progress['current_step_index']}/{progress['total_steps']}: {progress['current_step']}")
        
#         with col2:
#             if status == "running":
#                 if st.button("🛑 Cancel", key=f"cancel_{job_id}"):
#                     cancel_job(job_id)
#                     st.rerun()
            
#             if status in ["running", "pending"]:
#                 if st.button("🔄 Refresh", key=f"refresh_{job_id}"):
#                     st.rerun()
        
#         # Show logs (without nested expander)
#         if job.get("logs"):
#             st.markdown("**📄 Recent Logs:**")
#             log_container = st.container()
#             with log_container:
#                 for log in job["logs"][-20:]:  # Last 20 lines
#                     st.code(log, language="")
        
#         # Show error
#         if job.get("error"):
#             st.error(f"**Error:** {job['error']}")



# def cancel_job(job_id: str):
#     """Cancel a running job"""
#     try:
#         response = requests.delete(f"{API_BASE}/api/pipeline/jobs/{job_id}", timeout=10)
#         response.raise_for_status()
        
#         st.success(f"✅ Job {job_id} cancelled")
    
#     except Exception as e:
#         st.error(f"Failed to cancel job: {e}")
"""
Pipeline Runner Component for VideoEmotion Dashboard.
Allows running the pipeline from the UI with configuration options.
"""

import time
from pathlib import Path
import streamlit as st
import requests

API_BASE = st.session_state.get("api_base", "http://localhost:8000")


def find_latest_summary(project_root: Path, video_stem: str) -> Path | None:
    """
    Cherche le dernier summary.json pour une vidéo.
    Path attendu: output/reports/offline/<video_stem>/frames_fps<fps>/<timestamp>/summary.json
    """
    # On descend directement dans le dossier de la vidéo pour éviter les faux positifs
    base = project_root / "output" / "reports" / "offline" / video_stem
    if not base.exists():
        return None

    candidates = list(base.rglob("summary.json"))
    if not candidates:
        return None

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def find_session_results(project_root: Path, video_stem: str, fps: int) -> Path | None:
    """
    Cherche le fichier analyzed_emotions.json.
    Path attendu: output/emotion_results/<video>/frames_fps<fps>/person_XXXX/latest/analyzed_emotions.json
    """
    session_root = project_root / "output" / "emotion_results" / video_stem / f"frames_fps{fps}"
    if not session_root.exists():
        return None
    
    # On cherche récursivement car le person_id peut varier (person_0000, person_0001...)
    candidates = list(session_root.rglob("analyzed_emotions.json"))
    if not candidates:
        return None
        
    # On prend le plus récent
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def find_visualization_video(project_root: Path, video_stem: str) -> Path | None:
    """
    Cherche la vidéo annotée finale.
    Path attendu: output/visualizations/<video>/<video>_annotated_h264.mp4
    """
    viz_dir = project_root / "output" / "visualizations" / video_stem
    if not viz_dir.exists():
        return None
    
    # Format exact attendu par visualize_results.py
    candidate = viz_dir / f"{video_stem}_annotated_h264.mp4"
    if candidate.exists():
        return candidate
        
    return None


def validate_outputs(video_name: str, fps: int) -> dict:
    """
    Retourne un debug dict pour afficher dans le dashboard.
    """
    project_root = Path(__file__).resolve()
    # remonter jusqu'au root projet:
    # app/components/pipeline_runner.py -> components -> app -> root (3 niveaux)
    project_root = project_root.parents[2]

    video_path = project_root / "data" / "videos" / video_name
    if not video_path.exists():
        # Try finding with extension
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            candidate = project_root / "data" / "videos" / f"{video_name}{ext}"
            if candidate.exists():
                video_path = candidate
                break
                
    video_stem = Path(video_name).stem if video_path.exists() else Path(video_name).stem

    summary_path = find_latest_summary(project_root, video_stem)
    session_results = find_session_results(project_root, video_stem, fps)
    viz_path = find_visualization_video(project_root, video_stem)

    return {
        "project_root": str(project_root),
        "video_path": str(video_path),
        "summary_path": str(summary_path) if summary_path else None,
        "session_results": str(session_results) if session_results else None,
        "visualization_path": str(viz_path) if viz_path else None,
        "validation": {
            "has_video": video_path.exists(),
            "has_summary": bool(summary_path and Path(summary_path).exists()),
            "has_results": bool(session_results and Path(session_results).exists()),
            "has_visualization": bool(viz_path and Path(viz_path).exists()),
        },
    }


def render_pipeline_runner():
    st.header("⚙️ Pipeline Runner")

    video_input = st.text_input(
        "Video Name",
        value=st.session_state.get("pipeline_video", ""),
        placeholder="e.g., gemini2.mp4",
        help="Enter the video filename (must exist in data/videos/)",
        key="pipeline_video_input",
    )

    st.markdown("### Pipeline Configuration")
    col1, col2 = st.columns(2)

    with col1:
        fps = st.number_input("FPS", min_value=1, max_value=30, value=5, key="fps")
        smoothing = st.checkbox("Smoothing", value=True, key="smoothing")
        tta = st.checkbox("TTA (Test-Time Augmentation)", value=True, key="tta")
        visualize = st.checkbox("Visualize Results", value=True, key="visualize")

    with col2:
        backend = st.selectbox("Backend", options=["hsemotion", "deepface"], index=0, key="backend")
        export_bboxes = st.checkbox("Export Bounding Boxes", value=True, key="export_bboxes")
        overwrite = st.checkbox("Overwrite Existing", value=False, key="overwrite")

    st.markdown("### Advanced Options")
    with st.expander("Skip Steps"):
        no_extract = st.checkbox("Skip Frame Extraction", value=False, key="no_extract")
        no_detect = st.checkbox("Skip Face Detection", value=False, key="no_detect")
        no_analyze = st.checkbox("Skip Emotion Analysis", value=False, key="no_analyze")
        no_summary = st.checkbox("Skip Summary Generation", value=False, key="no_summary")
        no_visualize = st.checkbox("Skip Visualization", value=False, key="no_visualize")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Reset options", use_container_width=True, key="reset_options_btn"):
            for k in ["no_extract", "no_detect", "no_analyze", "no_summary", "no_visualize"]:
                st.session_state[k] = False
            st.rerun()

    with colB:
        debug_payload = st.checkbox("Debug: show payload", value=False, key="debug_payload")

    if st.button("▶️ Start Pipeline", type="primary", use_container_width=True, key="start_pipeline_btn"):
        if not video_input:
            st.error("Please enter a video name")
        else:
            start_pipeline(
                video_input,
                fps=int(fps),
                smoothing=bool(smoothing),
                tta=bool(tta),
                backend=str(backend),
                visualize=bool(visualize),
                export_bboxes=bool(export_bboxes),
                overwrite=bool(overwrite),
                no_extract=bool(no_extract),
                no_detect=bool(no_detect),
                no_analyze=bool(no_analyze),
                no_summary=bool(no_summary),
                no_visualize=bool(no_visualize),
                debug_payload=bool(debug_payload),
            )

    st.markdown("---")
    render_active_jobs()


def start_pipeline(
    video_name: str,
    fps: int = 5,
    smoothing: bool = True,
    tta: bool = True,
    backend: str = "hsemotion",
    visualize: bool = True,
    export_bboxes: bool = True,
    overwrite: bool = False,
    no_extract: bool = False,
    no_detect: bool = False,
    no_analyze: bool = False,
    no_summary: bool = False,
    no_visualize: bool = False,
    debug_payload: bool = False,
):
    try:
        payload = {
            "video_name": video_name,
            "options": {
                "fps": fps,
                "smoothing": smoothing,
                "tta": tta,
                "backend": backend,
                "visualize": visualize,
                "export_bboxes": export_bboxes,
                "overwrite": overwrite,
                "no_extract": no_extract,
                "no_detect": no_detect,
                "no_analyze": no_analyze,
                "no_summary": no_summary,
                "no_visualize": no_visualize,
            }
        }

        if debug_payload:
            st.json(payload)

        response = requests.post(f"{API_BASE}/api/pipeline/run", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        job_id = result["job_id"]

        st.success(f"✅ Pipeline started! Job ID: {job_id}")
        st.session_state.active_job_id = job_id

        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f"Failed to start pipeline: {e}")


def render_active_jobs():
    st.markdown("### 📋 Pipeline Jobs")

    try:
        response = requests.get(f"{API_BASE}/api/pipeline/jobs?limit=10", timeout=10)
        response.raise_for_status()
        data = response.json()
        jobs = data["jobs"]

        if not jobs:
            st.info("No pipeline jobs yet")
            return

        for job in jobs:
            render_job_card(job)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API")
    except Exception as e:
        st.error(f"Failed to load jobs: {e}")


def render_job_card(job: dict):
    job_id = job["job_id"]
    status = job["status"]
    video_name = job.get("video_name", "")

    status_config = {
        "pending": "⏳",
        "running": "▶️",
        "done": "✅",
        "error": "❌",
        "cancelled": "🚫",
    }
    emoji = status_config.get(status, "❓")

    expanded = (status == "running")

    with st.expander(f"{emoji} {video_name} - {status.upper()}", expanded=expanded):
        st.write(f"Job ID: `{job_id}`")
        if job.get("progress"):
            progress = job["progress"]
            st.progress(progress["percent"] / 100.0)
            st.caption(f"Step {progress['current_step_index']}/{progress['total_steps']}: {progress['current_step']}")

        if job.get("logs"):
            st.markdown("**📄 Recent Logs:**")
            for log in job["logs"][-20:]:
                st.code(log, language="")

        if job.get("error"):
            st.error(f"**Error:** {job['error']}")

        # ✅ Debug File Paths section (NOW REAL CHECKS)
        st.markdown("### 🔍 Debug: File Paths")
        fps = int(st.session_state.get("fps", 5))
        dbg = validate_outputs(video_name=video_name, fps=fps)
        st.json(dbg)

        if status == "running":
            if st.button("🛑 Cancel", key=f"cancel_{job_id}"):
                cancel_job(job_id)
                st.rerun()

        if status in ["running", "pending"]:
            if st.button("🔄 Refresh", key=f"refresh_{job_id}"):
                st.rerun()


def cancel_job(job_id: str):
    try:
        response = requests.delete(f"{API_BASE}/api/pipeline/jobs/{job_id}", timeout=10)
        response.raise_for_status()
        st.success(f"✅ Job {job_id} cancelled")
    except Exception as e:
        st.error(f"Failed to cancel job: {e}")
