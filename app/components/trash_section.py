"""
Trash Section Component for VideoEmotion Dashboard.
Displays trashed videos with restore and permanent delete actions.
"""

import streamlit as st
import requests
from datetime import datetime


API_BASE = st.session_state.get("api_base", "http://localhost:8000")


def render_trash_section():
    """Render the trash section"""
    st.header("🗑️ Trash")
    
    # Actions bar
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Deleted videos can be restored or permanently deleted.")
    
    with col2:
        if st.button("🔄 Refresh", key="trash_refresh"):
            st.rerun()
    
    st.markdown("---")
    
    # Fetch trash items
    try:
        response = requests.get(f"{API_BASE}/api/trash", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        trash_items = data["trash_items"]
        total = data["total"]
        
        if total == 0:
            st.info("🎉 Trash is empty!")
            return
        
        # Show empty trash button
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{total} items in trash**")
        with col3:
            if st.button("🗑️ Empty Trash", key="empty_trash", type="secondary"):
                st.session_state.confirm_empty_trash = True
        
        # Confirm empty trash
        if st.session_state.get("confirm_empty_trash", False):
            st.warning("⚠️ **Warning:** This will permanently delete ALL items in trash!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Empty Trash", key="confirm_empty_yes", type="primary"):
                    empty_trash()
                    st.session_state.confirm_empty_trash = False
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", key="confirm_empty_no"):
                    st.session_state.confirm_empty_trash = False
                    st.rerun()
            st.markdown("---")
        
        # Display trash items
        for item in trash_items:
            render_trash_item(item)
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running: `python app/api.py`")
    except Exception as e:
        st.error(f"Failed to load trash: {e}")


def render_trash_item(item: dict):
    """Render a single trash item"""
    trash_id = item["trash_id"]
    
    # Mode emoji
    mode_emoji = {
        "offline": "💾",
        "realtime": "🎥"
    }
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### {mode_emoji.get(item['mode'], '📹')} {item['video_name']}")
            
            deleted_at = datetime.fromisoformat(item["deleted_at"])
            st.caption(f"**Deleted:** {deleted_at.strftime('%Y-%m-%d %H:%M')}")
            st.caption(f"**Mode:** {item['mode'].title()}")
            
            if item.get("size_mb"):
                st.caption(f"**Size:** {item['size_mb']:.1f} MB")
        
        with col2:
            if st.button("♻️ Restore", key=f"restore_{trash_id}", use_container_width=True):
                restore_video(trash_id, item['video_name'])
                st.rerun()
        
        with col3:
            # Permanent delete with confirmation
            perm_delete_key = f"perm_delete_{trash_id}"
            confirm_perm_key = f"confirm_perm_{trash_id}"
            
            if st.session_state.get(confirm_perm_key, False):
                if st.button("⚠️ Confirm", key=perm_delete_key, type="primary", use_container_width=True):
                    delete_permanently(trash_id, item['video_name'])
                    st.session_state[confirm_perm_key] = False
                    st.rerun()
            else:
                if st.button("🔥 Delete Forever", key=perm_delete_key, use_container_width=True):
                    st.session_state[confirm_perm_key] = True
                    st.rerun()
        
        # Show confirmation warning
        if st.session_state.get(confirm_perm_key, False):
            st.error("⚠️ **This is permanent!** Click 'Confirm' to delete forever.")
        
        st.markdown("---")


def restore_video(trash_id: str, video_name: str):
    """Restore a video from trash"""
    try:
        response = requests.post(f"{API_BASE}/api/trash/{trash_id}/restore", timeout=60)
        response.raise_for_status()
        
        st.success(f"✅ '{video_name}' restored successfully!")
        st.info("📊 Statistics are being recalculated in the background...")
    
    except Exception as e:
        st.error(f"Failed to restore video: {e}")


def delete_permanently(trash_id: str, video_name: str):
    """Permanently delete a video"""
    try:
        response = requests.delete(f"{API_BASE}/api/trash/{trash_id}", timeout=30)
        response.raise_for_status()
        
        result = response.json()
        freed_mb = result.get("freed_space_mb", 0)
        
        st.success(f"✅ '{video_name}' permanently deleted! Freed {freed_mb:.1f} MB")
    
    except Exception as e:
        st.error(f"Failed to permanently delete video: {e}")


def empty_trash():
    """Empty entire trash"""
    try:
        response = requests.post(f"{API_BASE}/api/trash/empty", timeout=120)
        response.raise_for_status()
        
        result = response.json()
        count = result.get("message", "").split()[1] if "message" in result else "all"
        freed_mb = result.get("freed_space_mb", 0)
        
        st.success(f"✅ Trash emptied! Deleted {count} items, freed {freed_mb:.1f} MB")
    
    except Exception as e:
        st.error(f"Failed to empty trash: {e}")
