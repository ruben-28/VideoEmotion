"""
Unprocessed Videos Section Component for VideoEmotion Dashboard.
Displays videos that haven't been processed yet.
"""

import streamlit as st
import requests
from datetime import datetime


API_BASE = st.session_state.get("api_base", "http://localhost:8000")


def render_unprocessed_section():
    """Render the unprocessed videos section"""
    st.header("📥 Unprocessed Videos")
    
    st.write("Videos found on disk that haven't been processed through the pipeline yet.")
    
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
            return
        
        st.write(f"**{total} unprocessed videos found**")
        st.markdown("---")
        
        # Display unprocessed videos
        for video in videos:
            render_unprocessed_video(video)
    
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


def render_unprocessed_video(video: dict):
    """Render a single unprocessed video"""
    video_id = video["id"]
    video_name = video["name"]
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 💾 {video_name}")
            
            created_at = datetime.fromisoformat(video["created_at"])
            st.caption(f"**Added:** {created_at.strftime('%Y-%m-%d %H:%M')}")
            
            if video.get("file_size_mb"):
                st.caption(f"**Size:** {video['file_size_mb']:.1f} MB")
        
        with col2:
            if st.button("▶️ Process", key=f"process_{video_id}", use_container_width=True):
                st.session_state.pipeline_video = video_name
                st.session_state.page = "⚙️ Pipeline Runner"
                st.rerun()
        
        st.markdown("---")
