import sys
import os
import traceback

# Ensure CWD is in path to find 'src'
sys.path.insert(0, os.getcwd())

try:
    # Test that we can import api which now uses src.core
    import app.api
    
    # Optional: Check if we can create a RealtimeConfig and pass it to a RealtimeManager
    from src.core.realtime_manager import RealtimeConfig, RealtimeManager
    
    print("Import successful")
except Exception:
    with open("error.log", "w") as f:
        traceback.print_exc(file=f)
    print("Import failed, check error.log")
