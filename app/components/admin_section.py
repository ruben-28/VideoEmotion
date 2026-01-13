"""
Administration Section Component for VideoEmotion Dashboard.
Displays all videos with filters, sorting, and management actions.
Includes detail view with video player and analytics.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
import sys

# Import dashboard functions for video display
from .common import (
    load_and_prepare,
    render_video_block,
    render_overview,
    render_analysis,
)

# Import path resolver
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "core"))
from path_resolver import PathResolver

API_BASE = st.session_state.get("api_base", "http://localhost:8000")

# Initialize path resolver
project_root = Path(__file__).parent.parent.parent
path_resolver = PathResolver(project_root)



def render_admin_section():
    """Render the administration section with list/detail view modes"""
    # Initialize session state for view mode
    if "admin_view_mode" not in st.session_state:
        st.session_state.admin_view_mode = "list"
    if "admin_selected_video" not in st.session_state:
        st.session_state.admin_selected_video = None
    
    # Check if we should show detail view
    if st.session_state.admin_view_mode == "detail" and st.session_state.admin_selected_video:
        render_video_detail(st.session_state.admin_selected_video)
    else:
        render_video_list()


def render_video_list():
    """Render the video list view"""
    st.header("🎬 Administration")
    
    # Initialize auto-refresh state
    if "auto_refresh_enabled" not in st.session_state:
        st.session_state.auto_refresh_enabled = False
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    
    # Filters
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
    
    with col1:
        mode_filter = st.selectbox(
            "Mode",
            options=["all", "offline", "realtime"],
            key="admin_mode_filter"
        )
    
    with col2:
        status_filter = st.selectbox(
            "Status",
            options=["all", "processed", "partial", "unprocessed"],
            key="admin_status_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=["created_at", "name", "status"],
            key="admin_sort_by"
        )
    
    with col4:
        # Auto-refresh toggle
        auto_refresh = st.toggle(
            "Auto-refresh",
            value=st.session_state.auto_refresh_enabled,
            key="auto_refresh_toggle",
            help="Automatically refresh every 60 seconds"
        )
        if auto_refresh != st.session_state.auto_refresh_enabled:
            st.session_state.auto_refresh_enabled = auto_refresh
            st.session_state.last_refresh_time = time.time()
    
    with col5:
        if st.button("🔄 Refresh", key="admin_refresh"):
            # Trigger backend scan
            try:
                response = requests.post(f"{API_BASE}/api/videos/scan", timeout=5)
                if response.ok:
                    st.toast("🔄 Scanning videos...")
                    time.sleep(1)  # Give scan a moment to start
            except:
                pass  # Silent fail, will still refresh UI
            
            st.session_state.last_refresh_time = time.time()
            st.rerun()
    
    # Auto-refresh logic
    if st.session_state.auto_refresh_enabled:
        elapsed = time.time() - st.session_state.last_refresh_time
        if elapsed >= 60:
            st.session_state.last_refresh_time = time.time()
            st.rerun()
        else:
            # Show countdown
            remaining = int(60 - elapsed)
            st.caption(f"⏱️ Next refresh in {remaining}s")
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")
    
    # Fetch videos
    try:
        params = {
            "sort_by": sort_by,
            "sort_order": "desc",
            "per_page": 100
        }
        
        if mode_filter != "all":
            params["mode"] = mode_filter
        if status_filter != "all":
            params["status"] = status_filter
        
        response = requests.get(f"{API_BASE}/api/videos", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        videos = data["videos"]
        total = data["total"]
        
        # Display stats
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{total} videos found**")
        with col2:
            if st.button("📊 View Stats", key="view_stats"):
                st.session_state.show_stats = True
        
        if not videos:
            st.info("No videos found matching the filters.")
            return
        
        # Display videos in grid
        cols_per_row = 3
        for i in range(0, len(videos), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(videos):
                    video = videos[i + j]
                    with col:
                        render_video_card(video)
        
        # Show stats modal if requested
        if st.session_state.get("show_stats", False):
            render_stats_modal()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running: `python app/api.py`")
    except Exception as e:
        st.error(f"Failed to load videos: {e}")



def render_video_card(video: dict):
    """Render a single video card"""
    video_id = video["id"]
    
    # Status emoji
    status_emoji = {
        "processed": "✅",
        "partial": "⚠️",
        "unprocessed": "❌"
    }
    
    # Mode emoji
    mode_emoji = {
        "offline": "💾",
        "realtime": "🎥"
    }
    
    with st.container():
        st.markdown(
            f"""
            <div style="
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Title
        st.markdown(f"### {mode_emoji.get(video['mode'], '📹')} {video['name']}")
        
        # Info
        st.caption(f"**Mode:** {video['mode'].title()}")
        st.caption(f"**Status:** {status_emoji.get(video['status'], '❓')} {video['status'].title()}")
        
        # Date
        created_at = datetime.fromisoformat(video["created_at"])
        st.caption(f"**Created:** {created_at.strftime('%Y-%m-%d %H:%M')}")
        
        # Size
        if video.get("file_size_mb"):
            st.caption(f"**Size:** {video['file_size_mb']:.1f} MB")
        
        # Actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👁️ View", key=f"view_{video_id}", use_container_width=True):
                # Switch to detail view mode
                st.session_state.admin_view_mode = "detail"
                st.session_state.admin_selected_video = video
                st.rerun()
        
        with col2:
            # Delete button with confirmation
            delete_key = f"delete_{video_id}"
            confirm_key = f"confirm_delete_{video_id}"
            
            if st.session_state.get(confirm_key, False):
                if st.button("⚠️ Confirm Delete", key=delete_key, type="primary", use_container_width=True):
                    delete_video(video_id, video['name'])
                    st.session_state[confirm_key] = False
                    st.rerun()
            else:
                if st.button("🗑️ Delete", key=delete_key, use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()
        
        # Show confirmation message
        if st.session_state.get(confirm_key, False):
            st.warning("⚠️ Click 'Confirm Delete' to move to trash")



def delete_video(video_id: str, video_name: str):
    """Delete a video (move to trash)"""
    try:
        response = requests.delete(f"{API_BASE}/api/videos/{video_id}", timeout=30)
        response.raise_for_status()
        
        st.success(f"✅ '{video_name}' moved to trash")
        st.info("📊 Statistics are being recalculated in the background...")
    
    except Exception as e:
        st.error(f"Failed to delete video: {e}")


def render_video_detail(video: dict):
    """Render detailed video view with player and analytics"""
    st.header(f"🎬 {video['name']}")
    
    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ Back to List", key="back_to_list"):
            st.session_state.admin_view_mode = "list"
            st.session_state.admin_selected_video = None
            st.rerun()
    
    with col2:
        st.caption(f"**Mode:** {video['mode'].title()} | **Status:** {video['status'].title()}")
    
    st.markdown("---")
    
    # Use path resolver to get correct paths
    video_name = video['name']
    mode = video['mode']
    
    try:
        # Get all paths using path resolver
        paths = path_resolver.get_video_paths(video_name, mode)
        video_path = paths.get("video")
        summary_path = paths.get("summary")
        session_dir = paths.get("session_dir")
        
        # Debug info
        with st.expander("🔍 Debug: File Paths"):
            st.write(f"**Video path:** `{video_path}`")
            st.write(f"**Summary path:** `{summary_path}`")
            st.write(f"**Session dir:** `{session_dir}`")
            validation = path_resolver.validate_video_files(video_name, mode)
            st.write(f"**Validation:** {validation}")
        
        # Action Bar for Partial/Unprocessed Videos
        if video['status'] in ["partial", "unprocessed"] and mode == "offline":
            st.info("⚠️ This video is not fully processed or is missing some steps.")
            if st.button("⚙️ Process Missing Steps", type="primary", key="rerun_pipeline"):
                st.session_state.pipeline_video = video_name
                st.session_state.page = "⚙️ Pipeline Runner"
                st.rerun()
        
        # Check if files exist
        if not summary_path or not summary_path.exists():
            st.warning(f"⚠️ Summary file not found")
            if mode == "offline":
                st.info("This video may not have been processed yet.")
            return
        
        # Load data based on mode
        if mode == "offline":
            summary, df = load_and_prepare(summary_path, "offline")
        else:  # realtime
            summary, df = load_and_prepare(summary_path, "realtime")
        
        # Render video player
        if video_path and video_path.exists():
            render_video_block(video_path, auto_transcode=True)
        else:
            st.info("📹 Video file not found")
            if mode == "offline":
                st.caption(f"Expected: `output/visualizations/{video_name}/{video_name}_annotated_raw.mp4`")
        
        st.divider()
        
        # Render analytics
        render_overview(summary, df)
        
        st.divider()
        
        # Render timeline analysis
        render_analysis(df)
        

    
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Make sure the video has been processed through the pipeline.")
    except Exception as e:
        st.error(f"❌ Failed to load video details: {e}")
        st.info("Make sure the video has been processed through the pipeline.")
        with st.expander("🐛 Error Details"):
            import traceback
            st.code(traceback.format_exc())


def render_stats_modal():
    """Render statistics modal"""
    try:
        response = requests.get(f"{API_BASE}/api/stats", timeout=10)
        response.raise_for_status()
        stats = response.json()
        
        st.markdown("### 📊 Global Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Videos", stats["total_videos"])
            st.metric("Offline", stats["offline_videos"])
            st.metric("Realtime", stats["realtime_videos"])
        
        with col2:
            st.metric("Processed", stats["processed"])
            st.metric("Partial", stats["partial"])
            st.metric("Unprocessed", stats["unprocessed"])
        
        with col3:
            st.metric("Total Size", f"{stats['total_size_mb']:.1f} MB")
            trash_stats = stats.get("trash_stats", {})
            st.metric("Trash Items", trash_stats.get("total_items", 0))
            st.metric("Trash Size", f"{trash_stats.get('total_size_mb', 0):.1f} MB")
        
        if st.button("Close", key="close_stats"):
            st.session_state.show_stats = False
            st.rerun()
    
    except Exception as e:
        st.error(f"Failed to load statistics: {e}")
