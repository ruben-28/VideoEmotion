"""
Pipeline Log Parser - Parses pipeline output logs into structured progress events.
"""

import re
from typing import Optional, Tuple, Dict

# Standardized Step Constants
STEP_EXTRACT = "extract_frames"
STEP_DETECT = "detect_faces"
STEP_ANALYZE = "analyze_emotion"
STEP_SUMMARY = "generate_summary"
STEP_VISUALIZE = "visualize_results"
STEP_COMPLETED = "completed"

# Step definitions: (Step Name, Progress Percentage)
STEP_INFO: Dict[int, Tuple[str, float]] = {
    1: (STEP_EXTRACT, 20.0),
    2: (STEP_DETECT, 40.0),
    3: (STEP_ANALYZE, 60.0),
    4: (STEP_SUMMARY, 80.0),
    5: (STEP_VISUALIZE, 90.0),
}


class PipelineLogParser:
    """Parses pipeline stdout to detect progress steps."""

    def __init__(self):
        # Regex to capture [X/Y] pattern
        self._progress_pattern = re.compile(r"\[(\d+)/5\]")

    def parse_line(self, line: str) -> Optional[Tuple[int, str, float]]:
        """
        Parse a log line to determine if it indicates a new step.
        Returns (step_index, step_name, percent) or None.
        """
        match = self._progress_pattern.search(line)
        if match:
            try:
                step_idx = int(match.group(1))
                if step_idx in STEP_INFO:
                    name, percent = STEP_INFO[step_idx]
                    return step_idx, name, percent
            except ValueError:
                pass
        
        return None
