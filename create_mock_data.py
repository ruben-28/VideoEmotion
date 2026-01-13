import json
from pathlib import Path

def create_mock_data():
    base_dir = Path("output")
    reports_dir = base_dir / "reports" / "mock_video" / "2024-01-01_12-00-00"
    results_dir = base_dir / "emotion_results" / "mock_video" / "2024-01-01_12-00-00"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Absolute path for the input file (simulating how the pipeline works)
    analyzed_path = results_dir / "analyzed_emotions_final.json"
    
    # 1. Create analyzed_emotions_final.json (Hard case: Dict format, info in path)
    emotions = {
        "person_0001/frame_t0000.jpg": {"emotion": "neutral", "confidence": 0.95},
        "person_0001/frame_t1000.jpg": {"emotion": "happy", "confidence": 0.88},
        "person_0001/frame_t2000.jpg": {"emotion": "happy", "confidence": 0.92},
        "person_0001/frame_t3000.jpg": {"emotion": "surprise", "confidence": 0.75},
        "person_0001/frame_t4000.jpg": {"emotion": "surprise", "confidence": 0.80},
        "person_0001/frame_t5000.jpg": {"emotion": "neutral", "confidence": 0.90},
    }
    
    with open(analyzed_path, 'w') as f:
        json.dump(emotions, f, indent=2)
        
    print(f"Created {analyzed_path}")
    
    # 2. Create summary.json
    summary = {
        "session": "mock_video/2024-01-01_12-00-00",
        "total_frames": 6,
        "n_people": 1,
        "global_dominant_emotion": "happy",
        "inputs": [str(analyzed_path.resolve())],
        "people": [
            {
                "person_id": "person_0001",
                "dominant_emotion": "happy",
                "stability_score": 0.6,
                "top_emotions": [
                    {"emotion": "happy", "ratio": 0.33},
                    {"emotion": "surprise", "ratio": 0.33},
                    {"emotion": "neutral", "ratio": 0.33}
                ]
            }
        ]
    }
    
    summary_path = reports_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Created {summary_path}")
    
    # 3. Create mock Realtime Session
    realtime_dir = Path("output/realtime/session_mock_realtime")
    realtime_dir.mkdir(parents=True, exist_ok=True)
    
    realtime_records = []
    for i in range(10):
        realtime_records.append({
            "time_ms": i * 1000,
            "t_rel_ms": i * 1000,
            "emotion": "happy" if i % 2 == 0 else "neutral",
            "confidence": 0.85,
            "is_uncertain": False
        })
        
    with open(realtime_dir / "realtime_emotions.json", "w") as f:
        json.dump({"session": "mock_realtime", "records": realtime_records}, f)
        
    # 4. Create mock Visualized Video (dummy file)
    # User requested path: output/visualizations/{video_name}_annotated_bbox.mp4
    viz_dir = Path("output/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    with open(viz_dir / "mock_video_annotated_bbox.mp4", "w") as f:
        f.write("fake video content")

    print("[OK] Mock data created (Offline + Realtime + Visualizations path)")

if __name__ == "__main__":
    create_mock_data()
