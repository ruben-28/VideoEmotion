
import requests
import time
import sys
import os

BASE_URL = "http://localhost:8000/api"

def check_api():
    try:
        r = requests.get(f"{BASE_URL}/videos")
        return r.status_code == 200
    except:
        return False

def run_test():
    print("--- Starting Trash Lifecycle Test ---")
    
    # 1. Wait for API
    print("Waiting for API...")
    for _ in range(10):
        if check_api():
            break
        time.sleep(1)
    else:
        print("FATAL: API not reachable at http://localhost:8000")
        sys.exit(1)
    print("API is UP.")

    # 2. Trigger Scan to ensure trash_test.mp4 is found
    print("Triggering Scan...")
    r = requests.post(f"{BASE_URL}/videos/scan")
    if r.status_code != 200:
        print(f"FATAL: Scan failed {r.status_code}")
        sys.exit(1)
    print("Scan OK.")

    # 3. Find video ID
    print("Finding 'trash_test.mp4'...")
    r = requests.get(f"{BASE_URL}/videos")
    data = r.json()
    videos = data.get('videos', [])
    video_id = None
    
    print(f"Available videos: {[v['name'] for v in videos]}")
    
    for v in videos:
        # Check against name (usually stem) or filename if provided
        if v['name'] == 'trash_test.mp4' or v['name'] == 'trash_test':
            video_id = v['id']
            break
    
    if not video_id:
        print("FATAL: 'trash_test.mp4' not found in video list. Did you create it?")
        sys.exit(1)
    print(f"Found Video ID: {video_id}")

    # 4. Move to Trash
    print(f"Moving {video_id} to trash...")
    r = requests.delete(f"{BASE_URL}/videos/{video_id}")
    if r.status_code != 200:
        print(f"FATAL: Delete/Move to trash failed {r.status_code}: {r.text}")
        sys.exit(1)
    print("Moved to trash OK.")

    # 5. Verify in Trash
    print("Verifying in trash...")
    r = requests.get(f"{BASE_URL}/trash")
    trash_data = r.json()
    trash_items = trash_data.get('trash_items', [])
    trash_id = None
    for item in trash_items:
        # Check original_path or similar identifier. 
        if 'trash_test' in item.get('video_name', ''):
            trash_id = item['trash_id'] # Note: key is trash_id in API response model
            break
    
    if not trash_id:
        print("FATAL: Item not found in /api/trash list")
        print(f"Trash contents: {trash_items}")
        sys.exit(1)
    print(f"Found Trash ID: {trash_id}")

    # 6. Restore
    print(f"Restoring {trash_id}...")
    r = requests.post(f"{BASE_URL}/trash/{trash_id}/restore")
    if r.status_code != 200:
        print(f"FATAL: Restore failed {r.status_code}: {r.text}")
        sys.exit(1)
    print("Restore OK.")

    # 7. Verify back in Videos
    print("Verifying restoration...")
    r = requests.get(f"{BASE_URL}/videos")
    data = r.json()
    videos = data.get('videos', [])
    found = any(v['id'] == video_id for v in videos)
    if not found:
        print("FATAL: Video not found in list after restore")
        sys.exit(1)
    
    # 8. Verify NOT in Trash
    r = requests.get(f"{BASE_URL}/trash")
    trash_data = r.json()
    trash_items = trash_data.get('trash_items', [])
    if any(t['trash_id'] == trash_id for t in trash_items):
         print("FATAL: Item still in trash after restore")
         sys.exit(1)
    print("Restoration verified.")

    # 9. Move to Trash Again (for permanent delete)
    print("Moving to trash again...")
    r = requests.delete(f"{BASE_URL}/videos/{video_id}")
    if r.status_code != 200:
        print("FATAL: Move to trash (2nd time) failed")
        sys.exit(1)
    
    # Get new trash ID (it might change?)
    r = requests.get(f"{BASE_URL}/trash")
    trash_data = r.json()
    trash_items = trash_data.get('trash_items', [])
    trash_id = None
    for item in trash_items:
        if 'trash_test' in item.get('video_name', ''):
             trash_id = item['trash_id']
             break
    if not trash_id:
        print("FATAL: Item not found in trash (2nd time)")
        sys.exit(1)
    print(f"New Trash ID: {trash_id}")

    # 10. Delete Permanently
    print(f"Deleting permanently {trash_id}...")
    r = requests.delete(f"{BASE_URL}/trash/{trash_id}")
    if r.status_code != 200:
        print(f"FATAL: Permanent delete failed {r.status_code}: {r.text}")
        sys.exit(1)
    print("Permanent Delete OK.")

    # 11. Verify Gone Specifically
    print("Verifying gone...")
    r = requests.get(f"{BASE_URL}/trash")
    trash_data = r.json()
    trash_items = trash_data.get('trash_items', [])
    if any(t['trash_id'] == trash_id for t in trash_items):
        print("FATAL: Item still in trash after delete")
        sys.exit(1)
    
    r = requests.get(f"{BASE_URL}/videos")
    data = r.json()
    videos = data.get('videos', [])
    if any(v['id'] == video_id for v in videos):
        print("FATAL: Item still in videos after delete")
        sys.exit(1)
    
    print("✅ Trash Lifecycle Test Passed")

if __name__ == "__main__":
    run_test()
