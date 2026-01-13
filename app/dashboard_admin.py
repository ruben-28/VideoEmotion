"""
VideoEmotion Dashboard - Unified interface for analytics and administration.
Combines existing analytics with new administration features.
"""

import streamlit as st
import sys
from pathlib import Path

# =============================================================================
# Page Configuration - MUST BE FIRST
# =============================================================================
st.set_page_config(
    page_title="VideoEmotion - Administration",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add components to path
sys.path.insert(0, str(Path(__file__).parent))

from components import (
    render_admin_section,
    render_trash_section,
    render_unprocessed_section,
    render_pipeline_runner,
)

# Custom CSS
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
            max-width: 860px;
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

# =============================================================================
# Session State Initialization
# =============================================================================
if "api_base" not in st.session_state:
    st.session_state.api_base = "http://localhost:8000"

if "page" not in st.session_state:
    st.session_state.page = "🎬 Administration"

# =============================================================================
# Main Application
# =============================================================================
def main():
    # Sidebar navigation
    st.sidebar.title("🎬 VideoEmotion")
    st.sidebar.markdown("---")
    
    # Page selection
    pages = [
        "🎬 Administration",
        "🗑️ Trash",
        "📥 Unprocessed Videos",
        "⚙️ Pipeline Runner",
    ]
    
    # Use index-based selection to avoid double-click issues
    current_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0
    
    selected_page = st.sidebar.radio(
        "Navigation",
        options=pages,
        index=current_index,
    )
    
    # Update session state only if changed
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # API Configuration
    with st.sidebar.expander("⚙️ API Configuration"):
        api_base = st.text_input(
            "API Base URL",
            value=st.session_state.api_base,
            help="Base URL for the FastAPI backend"
        )
        if api_base != st.session_state.api_base:
            st.session_state.api_base = api_base
            st.rerun()
        
        # Test connection
        if st.button("🔌 Test Connection"):
            import requests
            try:
                response = requests.get(f"{api_base}/health", timeout=5)
                if response.ok:
                    st.success("✅ Connected!")
                else:
                    st.error(f"❌ Error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("VideoEmotion v1.0")
    st.sidebar.caption("Administration System")
    
    # Render selected page
    if selected_page == "🎬 Administration":
        render_admin_section()
    
    elif selected_page == "🗑️ Trash":
        render_trash_section()
    
    elif selected_page == "📥 Unprocessed Videos":
        render_unprocessed_section()
    
    elif selected_page == "⚙️ Pipeline Runner":
        render_pipeline_runner()



if __name__ == "__main__":
    main()
