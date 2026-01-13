import sys
import logging
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from src.core.pipeline_executor import PipelineExecutor
from src.core.models import PipelineConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestExecutor")

def main():
    print("Initializing PipelineExecutor...")
    executor = PipelineExecutor(project_root)
    
    video_name = "gemini2.mp4" # Adjust as needed
    
    print(f"Creating job for {video_name}...")
    config = PipelineConfig(
        fps=5,
        smoothing=True,
        tta=True,
        backend="hsemotion",
        visualize=True,
        export_bboxes=True,
        overwrite=True, # Force overwrite to ensure full run
        no_extract=False,
        no_detect=False,
        no_analyze=False, # Make sure these are False to run mostly everything
        no_summary=False,
        no_visualize=False
    )
    
    job_id = executor.create_job(video_name, config)
    print(f"Job created: {job_id}")
    
    print("Executing job...")
    # Run synchronously for test
    success = executor.execute_job(job_id)
    
    print(f"Execution finished. Success: {success}")
    
    # Check logs
    job = executor.get_job(job_id)
    if job:
        print("\n--- JOB LOGS ---")
        for log in job.logs:
            print(log)
        print("----------------")
        
        if job.error:
            print(f"ERROR: {job.error}")

if __name__ == "__main__":
    main()
