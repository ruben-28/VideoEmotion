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
    
    # Initialize selection state
    if "trash_selection" not in st.session_state:
        st.session_state.trash_selection = set()
    
    # Actions bar
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Deleted videos can be restored or permanently deleted.")
    
    with col2:
        if st.button("🔄 Refresh", key="trash_refresh"):
            st.session_state.trash_selection = set() # Clear selection on refresh
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
            st.session_state.trash_selection = set()
            return
        
        # Batch Actions Bar
        all_ids = {item["trash_id"] for item in trash_items}
        
        # Select All logic
        col_sel, col_stats, col_empty = st.columns([1, 2, 1])
        with col_sel:
            is_all_selected = len(st.session_state.trash_selection) == total and total > 0
            if st.checkbox("Select All", value=is_all_selected, key="trash_select_all"):
                st.session_state.trash_selection = all_ids
                # Sync all widget states
                for tid in all_ids:
                    st.session_state[f"sel_{tid}"] = True
            else:
                if is_all_selected: # Was selected, now unchecked
                    st.session_state.trash_selection = set()
                    # Sync all widget states (clear)
                    for tid in all_ids:
                        st.session_state[f"sel_{tid}"] = False
        
        with col_stats:
            num_selected = len(st.session_state.trash_selection)
            if num_selected > 0:
                st.info(f"☑️ {num_selected} selected")
            else:
                st.write(f"**{total} items in trash**")
                
        with col_empty:
             if st.button("🗑️ Empty Trash", key="empty_trash", type="secondary"):
                st.session_state.confirm_empty_trash = True

        # Placeholder for batch buttons (rendered late to catch state changes)
        batch_actions_placeholder = st.empty()
        
        st.divider()

        # Confirm empty trash (Global)
        if st.session_state.get("confirm_empty_trash", False):
            st.warning("⚠️ **Warning:** This will permanently delete ALL items in trash!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Empty Trash", key="confirm_empty_yes", type="primary"):
                    empty_trash()
                    st.session_state.confirm_empty_trash = False
                    st.session_state.trash_selection = set()
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", key="confirm_empty_no"):
                    st.session_state.confirm_empty_trash = False
                    st.rerun()
            st.markdown("---")
        
        # Display trash items
        for item in trash_items:
            render_trash_item(item)
            
        # LATE RENDER: Batch Action Buttons
        # We render this here so it reflects checkboxes clicked in this run
        num_selected = len(st.session_state.trash_selection)
        if num_selected > 0:
            with batch_actions_placeholder.container():
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button(f"♻️ Restore Selected ({num_selected})", type="primary", use_container_width=True):
                        restore_batch(list(st.session_state.trash_selection))
                with b_col2:
                    if st.button(f"🔥 Delete Selected ({num_selected})", type="secondary", use_container_width=True):
                         st.session_state.confirm_batch_delete = True
                
                if st.session_state.get("confirm_batch_delete", False):
                     st.warning(f"⚠️ Permanently delete {num_selected} items?")
                     if st.button("✅ Yes, Delete", key="confirm_batch_del_btn"):
                         delete_batch(list(st.session_state.trash_selection))
                         st.session_state.confirm_batch_delete = False
                         st.session_state.trash_selection = set() # Clear after delete
                         st.rerun()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running: `python app/api.py`")
    except Exception as e:
        st.error(f"Failed to load trash: {e}")


def toggle_trash_selection(item_id):
    """Callback to toggle individual item selection"""
    key = f"sel_{item_id}"
    if st.session_state.get(key, False):
        st.session_state.trash_selection.add(item_id)
    else:
        st.session_state.trash_selection.discard(item_id)


def render_trash_item(item: dict):
    """Render a single trash item"""
    trash_id = item["trash_id"]
    
    # Mode emoji
    mode_emoji = {
        "offline": "💾",
        "realtime": "🎥"
    }
    
    with st.container():
        # Checkbox for selection
        c1, c2, c3, c4 = st.columns([0.5, 3, 1, 1])
        
        with c1:
            # FORCE SYNC: Ensure widget state matches model
            is_selected = trash_id in st.session_state.trash_selection
            st.session_state[f"sel_{trash_id}"] = is_selected
            
            st.checkbox(
                "Select", 
                key=f"sel_{trash_id}", 
                on_change=toggle_trash_selection, 
                args=(trash_id,),
                label_visibility="collapsed"
            )

        with c2:
            st.markdown(f"**{mode_emoji.get(item['mode'], '📹')} {item['video_name']}**")
            deleted_at = datetime.fromisoformat(item["deleted_at"])
            size_str = f" • {item['size_mb']:.1f} MB" if item.get("size_mb") else ""
            st.caption(f"Deleted: {deleted_at.strftime('%Y-%m-%d %H:%M')}{size_str}")
        
        with c3:
            if st.button("♻️", key=f"restore_{trash_id}", help="Restore"):
                restore_video(trash_id, item['video_name'])
                st.rerun()
        
        with c4:
             # Permanent delete with confirmation
            perm_delete_key = f"perm_delete_{trash_id}"
            confirm_perm_key = f"confirm_perm_{trash_id}"
            
            if st.session_state.get(confirm_perm_key, False):
                if st.button("⚠️", key=perm_delete_key, type="primary", help="Confirm Delete"):
                    delete_permanently(trash_id, item['video_name'])
                    st.session_state[confirm_perm_key] = False
                    st.rerun()
            else:
                if st.button("🗑️", key=perm_delete_key, help="Delete Forever"):
                    st.session_state[confirm_perm_key] = True
                    st.rerun()
        
        st.markdown("---")


def restore_batch(trash_ids: list):
    """Restore multiple videos"""
    success_count = 0
    progress = st.progress(0)
    
    for i, trash_id in enumerate(trash_ids):
        try:
             requests.post(f"{API_BASE}/api/trash/{trash_id}/restore", timeout=60)
             success_count += 1
        except Exception as e:
            st.error(f"Failed to restore item {trash_id}: {e}")
        progress.progress((i + 1) / len(trash_ids))
        
    st.success(f"✅ Restored {success_count}/{len(trash_ids)} items")
    st.rerun()

def delete_batch(trash_ids: list):
    """Delete multiple videos forever"""
    success_count = 0
    progress = st.progress(0)
    
    for i, trash_id in enumerate(trash_ids):
        try:
             requests.delete(f"{API_BASE}/api/trash/{trash_id}", timeout=30)
             success_count += 1
        except Exception as e:
             st.error(f"Failed to delete item {trash_id}: {e}")
        progress.progress((i + 1) / len(trash_ids))

    st.success(f"✅ Permanently deleted {success_count}/{len(trash_ids)} items")
    st.rerun()


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
