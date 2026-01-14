
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class StatsCalculator:
    """
    Responsible ONLY for parsing and calculating statistics from source files.
    """
    
    def load_offline_stats(self, reports_dir: Path) -> Optional[Dict]:
        """Load stats from recursive search for summary.json"""
        if not reports_dir.exists():
            return None
            
        try:
            summaries = list(reports_dir.rglob("summary.json"))
            if summaries:
                latest_summary = max(summaries, key=lambda p: p.stat().st_mtime)
                with open(latest_summary, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Normalize schema to match realtime stats (frontend expects avg_emotion)
                    if "global_distribution" in data and "avg_emotion" not in data:
                        data["avg_emotion"] = data["global_distribution"]
                        
                    return data
        except Exception as e:
            logger.warning(f"Failed to load offline stats: {e}")
        return None

    def calculate_realtime_stats(self, json_path: Path) -> Optional[Dict]:
        """Calculate stats from a realtime_emotions.json file"""
        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
             logger.warning(f"Failed to load realtime stats: {e}")
             return None
            
        records = data.get("records", [])
        if not records:
            return None
            
        emotion_counts = Counter()
        timeline_buckets = defaultdict(Counter)
        valid_records = 0
        
        for rec in records:
            emo = rec.get("emotion")
            if not emo:
                continue
            
            emo = emo.strip().lower()
            emotion_counts[emo] += 1
            valid_records += 1
            
            t_ms = rec.get("t_rel_ms", rec.get("time_ms", 0))
            if t_ms is None: t_ms = 0
            sec = int(t_ms / 1000)
            timeline_buckets[sec][emo] += 1
            
        if valid_records == 0:
            return None
            
        global_distribution = {k: v / valid_records for k, v in emotion_counts.items()}
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        timeline = []
        if timeline_buckets:
            max_sec = max(timeline_buckets.keys())
            for s in range(max_sec + 1):
                counts = timeline_buckets.get(s, Counter())
                total = sum(counts.values())
                if total > 0:
                    dist = {k: round(v / total, 3) for k, v in counts.items()}
                else:
                    dist = {}
                timeline.append({
                    "timestamp": s,
                    "emotions": dist
                })
        
        return {
            "global_distribution": global_distribution,
            "dominant_emotion": dominant_emotion,
            "timeline": timeline,
            "avg_emotion": global_distribution 
        }
