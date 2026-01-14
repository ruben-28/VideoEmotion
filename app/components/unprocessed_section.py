"""
Unprocessed Videos Section Component for VideoEmotion Dashboard.
Displays videos that haven't been processed yet.
"""

import streamlit as st
import requests
from datetime import datetime
import time


API_BASE = st.session_state.get("api_base", "http://localhost:8000")


def render_unprocessed_section():
    """Render the unprocessed videos section"""
    st.header("📥 Unprocessed Videos")
    
    st.write("Videos found on disk that haven't been processed through the pipeline yet.")
    
    # Initialize selection
    if "unprocessed_selection" not in st.session_state:
        st.session_state.unprocessed_selection = set()

    # Actions bar
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Refresh", key="unprocessed_refresh"):
            # Trigger backend scan
            try:
                response = requests.post(f"{API_BASE}/api/videos/scan", timeout=5)
                if response.ok:
                    st.toast("🔄 Scanning videos...")
            except:
                pass  # Silent fail, will still refresh UI
            st.session_state.unprocessed_selection = set() # Clear
            st.rerun()
    
    st.markdown("---")
    
    # Fetch unprocessed videos
    try:
        response = requests.get(f"{API_BASE}/api/videos/unprocessed", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        videos = data["videos"]
        total = data["total"]
        
        if total == 0:
            st.success("✅ All videos have been processed!")
            st.session_state.unprocessed_selection = set()
            return

        all_ids = {v["id"] for v in videos}
        
        # Batch Selection Bar
        col_sel, col_stats = st.columns([1, 4])
        with col_sel:
            is_all_selected = len(st.session_state.unprocessed_selection) == total and total > 0
            if st.checkbox("Select All", value=is_all_selected, key="unproc_select_all"):
                st.session_state.unprocessed_selection = all_ids
                for vid in all_ids:
                    st.session_state[f"sel_unp_{vid}"] = True
            else:
                if is_all_selected:
                    st.session_state.unprocessed_selection = set()
                    for vid in all_ids:
                        st.session_state[f"sel_unp_{vid}"] = False
        
        with col_stats:
            num_selected = len(st.session_state.unprocessed_selection)
            if num_selected > 0:
                st.info(f"☑️ {num_selected} selected")
            else:
                st.write(f"**{total} unprocessed videos found**")

        # Placeholder for batch buttons
        batch_actions_placeholder = st.empty()
        
        # Display unprocessed videos
        st.markdown("---")
        for video in videos:
            render_unprocessed_video(video)
            
        # LATE RENDER: Batch Actions
        num_selected = len(st.session_state.unprocessed_selection)
        if num_selected > 0:
            with batch_actions_placeholder.container():
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"▶️ Process Selected ({num_selected})", type="primary", use_container_width=True):
                        process_batch(list(st.session_state.unprocessed_selection), videos)
                
                with c2:
                    if st.button(f"🗑️ Move to Trash ({num_selected})", type="secondary", use_container_width=True):
                        trash_batch(list(st.session_state.unprocessed_selection))

                st.divider()
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error("❌ Unprocessed videos endpoint not found (404)")
            st.info("The API endpoint `/api/videos/unprocessed` may not be implemented correctly.")
            st.info("Make sure the API server is running the latest version: `python app/api.py`")
        else:
            st.error(f"❌ HTTP Error: {e.response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running: `python app/api.py`")
    except Exception as e:
        st.error(f"Failed to load unprocessed videos: {e}")


def toggle_unprocessed_selection(item_id):
    """Callback to toggle individual item selection"""
    key = f"sel_unp_{item_id}"
    if st.session_state.get(key, False):
        st.session_state.unprocessed_selection.add(item_id)
    else:
        st.session_state.unprocessed_selection.discard(item_id)


def render_unprocessed_video(video: dict):
    """Render a single unprocessed video"""
    video_id = video["id"]
    video_name = video["name"]
    
    with st.container():
        c1, c2, c3 = st.columns([0.5, 3, 1])
        
        with c1:
             # FORCE SYNC
             is_selected = video_id in st.session_state.unprocessed_selection
             st.session_state[f"sel_unp_{video_id}"] = is_selected
             
             st.checkbox(
                 "Sel", 
                 key=f"sel_unp_{video_id}", 
                 on_change=toggle_unprocessed_selection,
                 args=(video_id,),
                 label_visibility="collapsed"
             )

        with c2:
            st.markdown(f"**💾 {video_name}**")
            created_at = datetime.fromisoformat(video["created_at"])
            size_str = f" • {video['file_size_mb']:.1f} MB" if video.get("file_size_mb") else ""
            st.caption(f"Added: {created_at.strftime('%Y-%m-%d %H:%M')}{size_str}")
        
        with c3:
            if st.button("process", key=f"process_{video_id}", help="Process this video"):
                st.session_state.pipeline_video = video_name
                st.session_state.page = "⚙️ Pipeline Runner"
                st.rerun()
        
        st.markdown("---")


def process_batch(video_ids: list, all_videos: list):
    """Start batch processing"""
    # Create lookup for name
    id_to_name = {v["id"]: v["name"] for v in all_videos}
    
    success_count = 0
    progress = st.progress(0)
    
    for i, vid in enumerate(video_ids):
        name = id_to_name.get(vid)
        if name:
            try:
                requests.post(f"{API_BASE}/api/pipeline/run", json={"video_name": name, "options": {}}, timeout=5)
                success_count += 1
            except Exception as e:
                st.error(f"Failed to start pipeline for {name}: {e}")
        progress.progress((i + 1) / len(video_ids))
    
    if success_count > 0:
        st.success(f"✅ Started {success_count} pipeline jobs!")
        st.info("Jobs are running in the background. Check the 'Pipeline Runner' page for status.")
        st.session_state.unprocessed_selection = set() # Clear
        time.sleep(2)
        st.rerun()

def trash_batch(video_ids: list):
    """Move batch to trash"""
    success_count = 0
    progress = st.progress(0)
    
    for i, vid in enumerate(video_ids):
        try:
             requests.delete(f"{API_BASE}/api/videos/{vid}", timeout=30)
             success_count += 1
        except Exception as e:
             st.error(f"Failed to move {vid} to trash: {e}")
        progress.progress((i + 1) / len(video_ids))
    
    st.success(f"✅ Moved {success_count} videos to trash")
    st.session_state.unprocessed_selection = set()
    st.rerun()
