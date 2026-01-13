# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import json
# import os
# from pathlib import Path
# import re

# # =============================================================================
# # Page Configuration
# # =============================================================================
# st.set_page_config(
#     page_title="VideoEmotion Analytics",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown(
#     """
# <style>
#     .reportview-container { background: #f0f2f6 }
#     .big-font { font-size:30px !important; font-weight: bold; }
#     .metric-card {
#         background-color: #ffffff;
#         border-radius: 10px;
#         padding: 20px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }

#     /* --- VIDEO SIZE LIMIT (PATCH) --- */
#     .video-wrap {
#       max-width: 900px;   /* <-- change ici la limite si tu veux (ex: 1100) */
#       width: 100%;
#       margin: 0 auto;
#     }
#     .video-wrap video {
#       width: 100% !important;
#       height: auto !important;
#     }
# </style>
# """,
#     unsafe_allow_html=True
# )

# # =============================================================================
# # Helpers (robust parsing)
# # =============================================================================
# PERSON_IN_PATH_RE = re.compile(r"(?:^|[\\/])(person[_-]?\d+)(?:[\\/]|$)", re.IGNORECASE)
# T_MS_RE = re.compile(r"_t(\d+)", re.IGNORECASE)

# def safe_int(x, default=-1) -> int:
#     try:
#         return int(x)
#     except Exception:
#         return default

# def extract_person_id(path: str, rec: dict) -> str:
#     gpid = rec.get("global_person_id", None)
#     if isinstance(gpid, str) and gpid.strip():
#         return gpid.strip()

#     if rec.get("identity_id", None) is not None:
#         return f"person_{safe_int(rec.get('identity_id'), 0):04d}"

#     m = PERSON_IN_PATH_RE.search(path or "")
#     if m:
#         pid = m.group(1).lower().replace("-", "_")
#         return re.sub(r"person_?(\d+)", lambda mm: f"person_{int(mm.group(1)):04d}", pid)

#     return "person_0000"

# def extract_time_ms(path: str, rec: dict) -> int:
#     if rec.get("t_rel_ms") is not None:
#         return safe_int(rec.get("t_rel_ms"), 0)

#     if rec.get("time_ms", None) is not None:
#         return safe_int(rec.get("time_ms"), -1)

#     m = T_MS_RE.search(path or "")
#     if m:
#         return safe_int(m.group(1), -1)

#     return -1

# # =============================================================================
# # Data loaders
# # =============================================================================
# def load_realtime_data(json_path: Path) -> pd.DataFrame:
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except Exception as e:
#         st.error(f"Erreur chargement JSON realtime: {e}")
#         return pd.DataFrame()

#     if isinstance(data, dict):
#         records = data.get("records", [])
#     elif isinstance(data, list):
#         records = data
#     else:
#         records = []

#     time_series = []
#     for rec in records:
#         if not isinstance(rec, dict):
#             continue

#         t_ms = extract_time_ms("", rec)
#         if t_ms < 0:
#             t_ms = 0

#         emotion = rec.get("emotion") or "Unknown"
#         confidence = rec.get("confidence") or 0.0

#         try:
#             confidence = float(confidence)
#         except Exception:
#             confidence = 0.0

#         if confidence > 1.0 and confidence <= 100.0:
#             confidence /= 100.0

#         time_series.append(
#             {
#                 "time_sec": t_ms / 1000.0,
#                 "emotion": emotion,
#                 "confidence": confidence,
#                 "person_id": "person_0000",
#             }
#         )

#     df = pd.DataFrame(time_series)
#     if not df.empty:
#         df.sort_values(by="time_sec", inplace=True)
#     return df

# def load_time_series_data(summary_data: dict) -> pd.DataFrame:
#     time_series = []
#     input_files = summary_data.get("inputs", [])

#     for file_path in input_files:
#         path = Path(file_path)
#         if not path.exists():
#             st.warning(f"Fichier source introuvable: {path}")
#             continue

#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 data = json.load(f)

#             items_to_process = []
#             if isinstance(data, dict):
#                 for k, v in data.items():
#                     if isinstance(v, dict):
#                         items_to_process.append((k, v))
#             elif isinstance(data, list):
#                 for i, v in enumerate(data):
#                     if isinstance(v, dict):
#                         p = (
#                             v.get("path")
#                             or v.get("relative_path")
#                             or v.get("img_path")
#                             or v.get("file")
#                             or f"idx_{i}"
#                         )
#                         items_to_process.append((p, v))

#             for path_k, rec in items_to_process:
#                 t_ms = extract_time_ms(str(path_k), rec)
#                 if t_ms < 0:
#                     t_ms = 0

#                 person_id = extract_person_id(str(path_k), rec)

#                 emotion = (
#                     rec.get("smoothed_final_emotion")
#                     or rec.get("final_emotion")
#                     or rec.get("emotion")
#                     or "Unknown"
#                 )
#                 confidence = rec.get("final_confidence") or rec.get("confidence") or 0.0

#                 try:
#                     confidence = float(confidence)
#                 except Exception:
#                     confidence = 0.0

#                 if confidence > 1.0 and confidence <= 100.0:
#                     confidence /= 100.0

#                 time_series.append(
#                     {
#                         "time_sec": t_ms / 1000.0,
#                         "emotion": emotion,
#                         "confidence": confidence,
#                         "person_id": person_id,
#                     }
#                 )

#         except Exception as e:
#             st.error(f"Erreur lecture fichier source {path}: {e}")

#     df = pd.DataFrame(time_series)
#     if not df.empty:
#         df.sort_values(by="time_sec", inplace=True)
#     return df

# # =============================================================================
# # Video selection helpers
# # =============================================================================
# def _extract_video_name_from_label(label: str) -> str:
#     """
#     label peut être:
#       - "bedouk/frames_fps5/...."
#       - "offline/bedouk/frames_fps5/...."  (cas possible)
#       - "realtime/session_...." (côté realtime)
#     """
#     parts = [p for p in (label or "").split("/") if p]
#     if not parts:
#         return ""
#     if parts[0].lower() in ("offline", "realtime"):
#         return parts[1] if len(parts) >= 2 else ""
#     return parts[0]

# def guess_offline_video_path(selected_session_label: str):
#     video_name = _extract_video_name_from_label(selected_session_label)
#     if not video_name:
#         return "", []

#     per_dir = Path("output/visualizations") / video_name

#     candidates = [
#         # New visualize_results outputs (per-video folder)
#         per_dir / f"{video_name}_annotated_h264.mp4",
#         per_dir / f"{video_name}_annotated_raw.mp4",

#         # Older naming fallback (still in per-video folder)
#         per_dir / f"{video_name}_annotated_bbox_h264.mp4",
#         per_dir / f"{video_name}_annotated_bbox.mp4",

#         # Legacy root-level outputs
#         Path("output/visualizations") / f"{video_name}_annotated_bbox_h264.mp4",
#         Path("output/visualizations") / f"{video_name}_annotated_bbox.mp4",
#     ]

#     # Fallback to original input video
#     for ext in [".mp4", ".avi", ".mov", ".mkv"]:
#         candidates.append(Path("data/videos") / (video_name + ext))

#     for c in candidates:
#         if c.exists():
#             return str(c.resolve()), [str(x.resolve()) for x in candidates]

#     return "", [str(x.resolve()) for x in candidates]

# def guess_realtime_video_path(realtime_json_path: Path):
#     cand1 = realtime_json_path.parent / "session_h264.mp4"
#     cand2 = realtime_json_path.parent / "session.mp4"
#     if cand1.exists():
#         return str(cand1.resolve()), [str(cand1.resolve()), str(cand2.resolve())]
#     if cand2.exists():
#         return str(cand2.resolve()), [str(cand1.resolve()), str(cand2.resolve())]
#     return "", [str(cand1.resolve()), str(cand2.resolve())]

# # =============================================================================
# # Rendering
# # =============================================================================
# def load_and_display_report(file_path: Path, session_type: str, video_path: str | None):
#     summary = {}
#     df = pd.DataFrame()

#     if session_type == "offline":
#         try:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 summary = json.load(f)
#             df = load_time_series_data(summary)
#         except Exception as e:
#             st.error(f"Erreur chargement report offline: {e}")
#             return

#     elif session_type == "realtime":
#         df = load_realtime_data(file_path)
#         summary = {
#             "session": str(file_path.parent.name),
#             "total_frames": len(df) if not df.empty else 0,
#             "n_people": 1,
#             "people": [],
#             "global_dominant_emotion": df["emotion"].mode()[0] if not df.empty else "Unknown",
#         }
#         if not df.empty:
#             avg_conf = float(df["confidence"].mean())
#             summary["people"] = [
#                 {"person_id": "person_0000", "avg_confidence": avg_conf, "n_frames": len(df)}
#             ]

#     st.header("🎬 Vidéo")
#     if video_path and os.path.exists(video_path):
#         st.markdown('<div class="video-wrap">', unsafe_allow_html=True)
#         st.video(video_path)
#         st.markdown("</div>", unsafe_allow_html=True)
#     else:
#         st.info("Aucune vidéo trouvée pour cette session (ou chemin vide).")

#     st.markdown("---")

#     # Overview
#     st.header("1. Overview")
#     total_frames = summary.get("total_frames", 1)
#     people_list = summary.get("people", [])

#     weighted_conf_sum = 0.0
#     total_frames_calc = 0
#     for p in people_list:
#         n = int(p.get("n_frames", 0) or 0)
#         weighted_conf_sum += float(p.get("avg_confidence", 0.0) or 0.0) * n
#         total_frames_calc += n

#     if total_frames_calc == 0:
#         total_frames_calc = 1
#     global_avg_conf = weighted_conf_sum / total_frames_calc

#     uncertain_pct = 0.0
#     if not df.empty:
#         uncertain_count = df[df["confidence"] < 0.6].shape[0]
#         uncertain_pct = (uncertain_count / len(df)) * 100.0

#     c1, c2, c3, c4, c5 = st.columns(5)
#     with c1:
#         st.metric("Session", str(summary.get("session", "Unknown")).split("/")[-1])
#     with c2:
#         st.metric("Total Frames", int(total_frames))
#     with c3:
#         st.metric("People (approx)", int(summary.get("n_people", 0) or 0))
#     with c4:
#         st.metric("Avg Confidence", f"{global_avg_conf:.2f}")
#     with c5:
#         st.metric("Uncertain (<0.6)", f"{uncertain_pct:.1f}%")

#     st.metric("Dominant Emotion", summary.get("global_dominant_emotion", "Unknown"))
#     st.markdown("---")

#     if df.empty:
#         st.info("Aucune donnée time-series trouvée dans les fichiers sources.")
#         return

#     st.header("2. Analysis")
#     people = sorted(df["person_id"].unique())
#     col_filter, _ = st.columns([1, 4])

#     with col_filter:
#         st.subheader("Filtrer par personne")
#         selected_people = st.multiselect("Personnes", options=people, default=people)
#         st.markdown("---")
#         st.subheader("Type de graphe")
#         chart_type = st.radio("Style", ["Line", "Scatter", "Area"], index=0)

#     if not selected_people:
#         st.info("Sélectionne au moins une personne.")
#         return

#     filtered_df = df[df["person_id"].isin(selected_people)]

#     st.subheader("Emotion Confidence over Time")
#     if chart_type == "Scatter":
#         fig = px.scatter(filtered_df, x="time_sec", y="confidence", color="emotion", custom_data=["person_id"])
#     elif chart_type == "Area":
#         fig = px.area(filtered_df, x="time_sec", y="confidence", color="emotion", line_group="person_id", custom_data=["person_id"])
#     else:
#         fig = px.line(filtered_df, x="time_sec", y="confidence", color="emotion", line_group="person_id", custom_data=["person_id"])

#     fig.update_traces(
#         hovertemplate="<br>".join(
#             ["Time: %{x:.2f}s", "Confidence: %{y:.2f}", "Person: %{customdata[0]}"]
#         )
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     c_dist, c_heat = st.columns(2)
#     with c_dist:
#         st.subheader("Global Distribution")
#         st.plotly_chart(px.pie(filtered_df, names="emotion", hole=0.4), use_container_width=True)
#     with c_heat:
#         st.subheader("Intensity Heatmap")
#         st.plotly_chart(
#             px.density_heatmap(filtered_df, x="time_sec", y="emotion", z="confidence", nbinsx=50),
#             use_container_width=True
#         )

# # =============================================================================
# # Main
# # =============================================================================
# def main():
#     st.title("🎥 VideoEmotion Analytics Dashboard")
#     st.markdown("---")

#     st.sidebar.header("Configuration")

#     base_output_dir = Path("output/reports")
#     realtime_output_dir = Path("output/realtime")

#     session_source = st.sidebar.radio("Session Source", ["Offline", "Realtime"])

#     session_options: dict[str, tuple[Path, str]] = {}

#     if session_source == "Offline":
#         if base_output_dir.exists():
#             for p in base_output_dir.rglob("summary.json"):
#                 rel_path = str(p.parent.relative_to(base_output_dir)).replace("\\", "/")
#                 session_options[rel_path] = (p, "offline")
#         else:
#             st.sidebar.warning(f"Dossier offline introuvable: {base_output_dir}")

#     else:
#         if realtime_output_dir.exists():
#             for p in realtime_output_dir.rglob("realtime_emotions.json"):
#                 rel_path = str(p.parent.relative_to(realtime_output_dir)).replace("\\", "/")
#                 session_options[rel_path] = (p, "realtime")
#         else:
#             st.sidebar.warning(f"Dossier realtime introuvable: {realtime_output_dir}")

#     if not session_options:
#         st.sidebar.warning(f"Aucune session {session_source} trouvée.")
#         return

#     sorted_labels = sorted(session_options.keys(), reverse=True)
#     selected_session_label = st.sidebar.selectbox("Select Session", options=sorted_labels, index=0)

#     if not selected_session_label:
#         return

#     file_path, session_type = session_options[selected_session_label]

#     st.sidebar.markdown("---")
#     st.sidebar.header("Video Player")

#     attempted = []
#     if session_type == "offline":
#         auto_video_path, attempted = guess_offline_video_path(selected_session_label)
#     else:
#         auto_video_path, attempted = guess_realtime_video_path(file_path)

#     video_path_input = st.sidebar.text_input("Video File Path", value=auto_video_path)

#     final_video_path = ""
#     if video_path_input:
#         try:
#             final_video_path = str(Path(video_path_input).resolve())
#         except Exception:
#             final_video_path = video_path_input

#     st.sidebar.markdown("---")
#     st.sidebar.subheader("Debug vidéo")
#     if final_video_path:
#         st.sidebar.write(f"Chemin: {final_video_path}")
#         if os.path.exists(final_video_path):
#             try:
#                 size_mb = os.path.getsize(final_video_path) / (1024 * 1024)
#                 st.sidebar.write(f"Taille: {size_mb:.2f} Mo")
#             except Exception:
#                 pass
#         else:
#             st.sidebar.error("Le fichier vidéo n'existe pas (chemin invalide).")
#     else:
#         st.sidebar.warning("Chemin vidéo vide.")
#         if attempted:
#             st.sidebar.caption("Candidats testés:")
#             for c in attempted[:10]:
#                 st.sidebar.code(c)

#     load_and_display_report(file_path=file_path, session_type=session_type, video_path=final_video_path)

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from pathlib import Path
import re
import subprocess


# =============================================================================
# Helpers (robust parsing)
# =============================================================================
PERSON_IN_PATH_RE = re.compile(r"(?:^|[\\/])(person[_-]?\d+)(?:[\\/]|$)", re.IGNORECASE)
T_MS_RE = re.compile(r"_t(\d+)", re.IGNORECASE)


def safe_int(x, default=-1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def extract_person_id(path: str, rec: dict) -> str:
    gpid = rec.get("global_person_id", None)
    if isinstance(gpid, str) and gpid.strip():
        return gpid.strip()

    if rec.get("identity_id", None) is not None:
        return f"person_{safe_int(rec.get('identity_id'), 0):04d}"

    m = PERSON_IN_PATH_RE.search(path or "")
    if m:
        pid = m.group(1).lower().replace("-", "_")
        return re.sub(r"person_?(\d+)", lambda mm: f"person_{int(mm.group(1)):04d}", pid)

    return "person_0000"


def extract_time_ms(path: str, rec: dict) -> int:
    if rec.get("t_rel_ms") is not None:
        return safe_int(rec.get("t_rel_ms"), 0)

    if rec.get("time_ms", None) is not None:
        return safe_int(rec.get("time_ms"), -1)

    m = T_MS_RE.search(path or "")
    if m:
        return safe_int(m.group(1), -1)

    return -1


def file_size_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def read_video_bytes(video_path: Path) -> bytes:
    with open(video_path, "rb") as f:
        return f.read()


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


def transcode_to_h264(src: Path, dst: Path) -> tuple[bool, str]:
    """
    Create a browser-friendly mp4 (H264 yuv420p).
    Returns (ok, message).
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-movflags", "+faststart",
            "-an",
            str(dst),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ok = (proc.returncode == 0) and dst.exists()
        msg = proc.stderr[-1500:] if proc.stderr else ""
        return ok, msg
    except Exception as e:
        return False, str(e)


# =============================================================================
# Data loaders
# =============================================================================
def load_realtime_data(json_path: Path) -> pd.DataFrame:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Realtime JSON load error: {e}")
        return pd.DataFrame()

    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        records = []

    time_series = []
    for rec in records:
        if not isinstance(rec, dict):
            continue

        t_ms = extract_time_ms("", rec)
        if t_ms < 0:
            t_ms = 0

        emotion = rec.get("emotion") or "Unknown"
        confidence = rec.get("confidence") or 0.0

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        if confidence > 1.0 and confidence <= 100.0:
            confidence /= 100.0

        time_series.append(
            {
                "time_sec": t_ms / 1000.0,
                "emotion": emotion,
                "confidence": confidence,
                "person_id": "person_0000",
            }
        )

    df = pd.DataFrame(time_series)
    if not df.empty:
        df.sort_values(by="time_sec", inplace=True)
    return df


def load_time_series_data(summary_data: dict) -> pd.DataFrame:
    time_series = []
    input_files = summary_data.get("inputs", [])

    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items_to_process = []
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        items_to_process.append((k, v))
            elif isinstance(data, list):
                for i, v in enumerate(data):
                    if isinstance(v, dict):
                        p = (
                            v.get("path")
                            or v.get("relative_path")
                            or v.get("img_path")
                            or v.get("file")
                            or f"idx_{i}"
                        )
                        items_to_process.append((p, v))

            for path_k, rec in items_to_process:
                t_ms = extract_time_ms(str(path_k), rec)
                if t_ms < 0:
                    t_ms = 0

                person_id = extract_person_id(str(path_k), rec)

                emotion = (
                    rec.get("smoothed_final_emotion")
                    or rec.get("final_emotion")
                    or rec.get("emotion")
                    or "Unknown"
                )
                confidence = rec.get("final_confidence") or rec.get("confidence") or 0.0

                try:
                    confidence = float(confidence)
                except Exception:
                    confidence = 0.0

                if confidence > 1.0 and confidence <= 100.0:
                    confidence /= 100.0

                time_series.append(
                    {
                        "time_sec": t_ms / 1000.0,
                        "emotion": emotion,
                        "confidence": confidence,
                        "person_id": person_id,
                    }
                )

        except Exception as e:
            st.error(f"Source file read error ({path}): {e}")

    df = pd.DataFrame(time_series)
    if not df.empty:
        df.sort_values(by="time_sec", inplace=True)
    return df


# =============================================================================
# Video selection helpers
# =============================================================================
def _extract_video_name_from_label(label: str) -> str:
    parts = [p for p in (label or "").split("/") if p]
    if not parts:
        return ""
    if parts[0].lower() in ("offline", "realtime"):
        return parts[1] if len(parts) >= 2 else ""
    return parts[0]


def guess_offline_video_candidates(selected_session_label: str) -> list[Path]:
    video_name = _extract_video_name_from_label(selected_session_label)
    if not video_name:
        return []

    per_dir = Path("output/visualizations") / video_name

    return [
        # preferred
        per_dir / f"{video_name}_annotated_h264.mp4",
        per_dir / f"{video_name}_annotated_raw.mp4",
        # fallbacks
        per_dir / f"{video_name}_annotated_bbox_h264.mp4",
        per_dir / f"{video_name}_annotated_bbox.mp4",
        # original input as last resort
        Path("data/videos") / f"{video_name}.mp4",
        Path("data/videos") / f"{video_name}.avi",
        Path("data/videos") / f"{video_name}.mov",
        Path("data/videos") / f"{video_name}.mkv",
    ]


def guess_realtime_video_candidates(realtime_json_path: Path) -> list[Path]:
    return [
        realtime_json_path.parent / "session_h264.mp4",
        realtime_json_path.parent / "session.mp4",
    ]


def pick_first_existing(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


# =============================================================================
# Rendering
# =============================================================================
def render_video_block(video_path: Path | None, auto_transcode: bool):
    st.subheader("Video")

    if video_path is None:
        st.info("No video file found for this session.")
        return

    if not video_path.exists():
        st.info("Video path does not exist.")
        return

    use_path = video_path

    # optional on-the-fly transcode if not h264
    if auto_transcode and video_path.suffix.lower() == ".mp4" and "_h264" not in video_path.stem.lower():
        if ffmpeg_available():
            dst = video_path.with_name(video_path.stem + "_h264.mp4")
            if not dst.exists():
                with st.spinner("Transcoding to H264 for browser playback..."):
                    ok, msg = transcode_to_h264(video_path, dst)
                if ok:
                    use_path = dst
                else:
                    st.warning("H264 transcode failed. The raw video may not play in the browser.")
                    if msg:
                        with st.expander("ffmpeg log"):
                            st.code(msg)
            else:
                use_path = dst
        else:
            st.warning("ffmpeg is not available in PATH. Cannot auto-transcode.")

    meta = f"File: {use_path} | Size: {file_size_mb(use_path):.2f} MB"
    st.markdown(f"<div class='muted'>{meta}</div>", unsafe_allow_html=True)

    try:
        st.markdown('<div class="video-wrap">', unsafe_allow_html=True)
        st.video(read_video_bytes(use_path))
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Video display error: {e}")


def render_overview(summary: dict, df: pd.DataFrame):
    st.subheader("Overview")

    total_frames = int(summary.get("total_frames", 0) or 0)
    n_people = int(summary.get("n_people", 0) or 0)
    people_list = summary.get("people", []) or []

    weighted_conf_sum = 0.0
    total_frames_calc = 0
    for p in people_list:
        n = int(p.get("n_frames", 0) or 0)
        weighted_conf_sum += float(p.get("avg_confidence", 0.0) or 0.0) * n
        total_frames_calc += n

    if total_frames_calc <= 0:
        global_avg_conf = float(df["confidence"].mean()) if not df.empty else 0.0
    else:
        global_avg_conf = weighted_conf_sum / total_frames_calc

    uncertain_pct = 0.0
    if not df.empty:
        uncertain_count = df[df["confidence"] < 0.6].shape[0]
        uncertain_pct = (uncertain_count / len(df)) * 100.0

    dom_emotion = summary.get("global_dominant_emotion", "Unknown")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Session", str(summary.get("session", "Unknown")).split("/")[-1])
    c2.metric("Frames", total_frames)
    c3.metric("People (approx)", n_people)
    c4.metric("Avg confidence", f"{global_avg_conf:.2f}")
    c5.metric("Low confidence (<0.6)", f"{uncertain_pct:.1f}%")

    st.markdown(
        f"<div class='card'><b>Dominant emotion</b><div class='muted'>{dom_emotion}</div></div>",
        unsafe_allow_html=True,
    )


def render_analysis(df: pd.DataFrame):
    st.subheader("Analysis")

    if df.empty:
        st.info("No time-series data found.")
        return

    people = sorted(df["person_id"].unique())

    with st.sidebar:
        st.markdown("## Filters")
        selected_people = st.multiselect("People", options=people, default=people)
        chart_type = st.radio("Chart", ["Line", "Scatter", "Area"], index=0)

    if not selected_people:
        st.info("Select at least one person.")
        return

    filtered_df = df[df["person_id"].isin(selected_people)]

    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    st.caption("Confidence over time")

    if chart_type == "Scatter":
        fig = px.scatter(filtered_df, x="time_sec", y="confidence", color="emotion", custom_data=["person_id"])
    elif chart_type == "Area":
        fig = px.area(filtered_df, x="time_sec", y="confidence", color="emotion", line_group="person_id", custom_data=["person_id"])
    else:
        fig = px.line(filtered_df, x="time_sec", y="confidence", color="emotion", line_group="person_id", custom_data=["person_id"])

    fig.update_traces(
        hovertemplate="<br>".join(
            ["Time: %{x:.2f}s", "Confidence: %{y:.2f}", "Person: %{customdata[0]}"]
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    c_dist, c_heat = st.columns(2)
    with c_dist:
        st.caption("Distribution")
        st.plotly_chart(px.pie(filtered_df, names="emotion", hole=0.45), use_container_width=True)
    with c_heat:
        st.caption("Heatmap")
        st.plotly_chart(
            px.density_heatmap(filtered_df, x="time_sec", y="emotion", z="confidence", nbinsx=50),
            use_container_width=True
        )


def load_and_prepare(file_path: Path, session_type: str) -> tuple[dict, pd.DataFrame]:
    if session_type == "offline":
        with open(file_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        df = load_time_series_data(summary)
        return summary, df

    df = load_realtime_data(file_path)
    summary = {
        "session": str(file_path.parent.name),
        "total_frames": len(df) if not df.empty else 0,
        "n_people": 1,
        "people": [],
        "global_dominant_emotion": df["emotion"].mode()[0] if not df.empty else "Unknown",
    }
    if not df.empty:
        avg_conf = float(df["confidence"].mean())
        summary["people"] = [
            {"person_id": "person_0000", "avg_confidence": avg_conf, "n_frames": len(df)}
        ]
    return summary, df


# =============================================================================
# Main
# =============================================================================
def main():
    # Page Configuration - MUST BE FIRST
    st.set_page_config(
        page_title="VideoEmotion Analytics",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
      /* App background */
      .reportview-container { background: #f6f7fb }
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      /* Typography */
      h1, h2, h3 { letter-spacing: -0.3px; }
      .muted { color: #6b7280; font-size: 0.92rem; }

      /* Cards */
      .card {
        background: #ffffff;
        border: 1px solid #e9ebef;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        padding: 14px 16px;
      }

      /* Video width limit */
      .video-wrap {
        max-width: 860px;   /* Change this to set video width limit */
        width: 100%;
        margin: 0 auto;
      }
      .video-wrap video {
        width: 100% !important;
        height: auto !important;
      }

      /* Section spacing */
      .section { margin-top: 0.8rem; margin-bottom: 0.8rem; }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
    st.title("VideoEmotion Analytics")
    st.caption("Offline and realtime session exploration")

    base_output_dir = Path("output/reports")
    realtime_output_dir = Path("output/realtime")

    with st.sidebar:
        st.markdown("## Session")
        session_source = st.radio("Source", ["Offline", "Realtime"], index=0)
        st.markdown("## Playback")
        auto_transcode = st.toggle("Auto-transcode to H264", value=True)
        st.caption("If a raw mp4v video fails to play, H264 improves browser compatibility.")
        st.markdown("---")

    session_options: dict[str, tuple[Path, str]] = {}

    if session_source == "Offline":
        if base_output_dir.exists():
            for p in base_output_dir.rglob("summary.json"):
                rel_path = str(p.parent.relative_to(base_output_dir)).replace("\\", "/")
                session_options[rel_path] = (p, "offline")
        else:
            st.sidebar.warning(f"Offline folder not found: {base_output_dir}")
    else:
        if realtime_output_dir.exists():
            for p in realtime_output_dir.rglob("realtime_emotions.json"):
                rel_path = str(p.parent.relative_to(realtime_output_dir)).replace("\\", "/")
                session_options[rel_path] = (p, "realtime")
        else:
            st.sidebar.warning(f"Realtime folder not found: {realtime_output_dir}")

    if not session_options:
        st.sidebar.warning("No sessions found.")
        return

    sorted_labels = sorted(session_options.keys(), reverse=True)

    with st.sidebar:
        selected_session_label = st.selectbox("Session", options=sorted_labels, index=0)

    file_path, session_type = session_options[selected_session_label]

    # Candidate video paths
    if session_type == "offline":
        candidates = guess_offline_video_candidates(selected_session_label)
    else:
        candidates = guess_realtime_video_candidates(file_path)

    chosen = pick_first_existing(candidates)

    with st.sidebar:
        st.markdown("## Video path")
        manual = st.text_input("Override path", value=str(chosen) if chosen else "")
        st.markdown("## Debug")
        if manual.strip():
            mp = Path(manual)
            st.caption(str(mp))
            st.caption(f"Exists: {mp.exists()}")
            if mp.exists():
                st.caption(f"Size: {file_size_mb(mp):.2f} MB")
        else:
            st.caption("No video found automatically.")
        with st.expander("Tried candidates"):
            for c in candidates[:30]:
                try:
                    st.code(str(c.resolve()))
                except Exception:
                    st.code(str(c))

    final_video_path = None
    if manual.strip():
        try:
            final_video_path = Path(manual).resolve()
        except Exception:
            final_video_path = Path(manual)

    # Load data
    try:
        summary, df = load_and_prepare(file_path=file_path, session_type=session_type)
    except Exception as e:
        st.error(f"Failed to load session data: {e}")
        return

    # Layout: tabs
    tab_overview, tab_timeline, tab_people = st.tabs(["Overview", "Timeline", "People"])

    with tab_overview:
        render_video_block(final_video_path, auto_transcode=auto_transcode)
        st.divider()
        render_overview(summary, df)

    with tab_timeline:
        render_analysis(df)

    with tab_people:
        if df.empty:
            st.info("No data.")
        else:
            st.subheader("People summary")
            g = (
                df.groupby("person_id")
                .agg(
                    n=("person_id", "size"),
                    avg_conf=("confidence", "mean"),
                    top_emotion=("emotion", lambda x: x.value_counts().index[0] if len(x) else "Unknown"),
                )
                .reset_index()
                .sort_values("n", ascending=False)
            )
            st.dataframe(g, use_container_width=True)

if __name__ == "__main__":
    main()
