"""
Unified plotting script for VideoEmotion.
Handles both Realtime (JSON records) and Offline (JSON frame index) data.
Generates:
- Timeline (Confidence over time)
- Distribution (Pie/Bar)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(json_path: Path) -> pd.DataFrame:
    """
    Load data from JSON and normalize to a common DataFrame format:
    Columns: [time_sec, emotion, confidence, is_uncertain, source_type]
    """
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    source_type = "unknown"

    # 1. Detect format
    # Realtime format: {"session": "...", "records": [...]}
    if isinstance(raw, dict) and "records" in raw and isinstance(raw["records"], list):
        # Realtime
        source_type = "realtime"
        for r in raw["records"]:
            # Rec keys: time_ms, t_rel_ms (optional), emotion, confidence
            t_ms = r.get("t_rel_ms", r.get("time_ms", 0))
            records.append({
                "time_sec": t_ms / 1000.0,
                "emotion": r.get("emotion"),
                "confidence": r.get("confidence", 0.0),
                "is_uncertain": r.get("is_uncertain", False)
            })

    # Offline format (master): {"filename": {...}, ...} -> we need to handle specific video aggregation?
    # Or per-video results.json: {"frame_0": {...}, ...} ??
    # Actually offline results usually are per-video: {"key": {"frame_index":...}}
    # Let's support the standard offline output format found in 'emotion_results'
    elif isinstance(raw, dict):
        # Offline? Check first item
        first_key = next(iter(raw))
        first_val = raw[first_key]
        if isinstance(first_val, dict) and "final_emotion" in first_val:
            source_type = "offline"
            for k, v in raw.items():
                if not isinstance(v, dict): continue
                
                # Check uncertain
                is_unc = False
                emo = v.get("final_emotion")
                
                # Offline might use "smoothed_final_emotion"
                if "smoothed_final_emotion" in v and v["smoothed_final_emotion"]:
                    emo = v["smoothed_final_emotion"]
                
                if not emo or emo == "Unknown":
                    is_unc = True
                    emo = "Uncertain"
                
                # Timestamp handling
                # Offline usually has "timestamp" (float) or we infer from frame_index * fps
                t_sec = v.get("timestamp_sec", v.get("timestamp", 0.0))
                
                records.append({
                    "time_sec": float(t_sec),
                    "emotion": emo,
                    "confidence": float(v.get("final_confidence", v.get("confidence", 0.0))),
                    "is_uncertain": is_unc
                })
    
    # List format (sometimes used)
    elif isinstance(raw, list):
        source_type = "list"
        for i, v in enumerate(raw):
            if not isinstance(v, dict): continue
             # Try to normalize from mixed keys
            emo = v.get("emotion", v.get("final_emotion"))
            conf = v.get("confidence", v.get("final_confidence", 0.0))
            t = v.get("time_sec", v.get("timestamp", i * 0.2)) # Fallback 5fps
            
            records.append({
                "time_sec": float(t),
                "emotion": emo,
                "confidence": float(conf),
                "is_uncertain": False # Basic assumption
            })

    df = pd.DataFrame(records)
    if not df.empty:
        # Clean emotions
        df["emotion"] = df["emotion"].fillna("Uncertain")
        df["emotion"] = df["emotion"].replace(["", "None", "unknown"], "Uncertain")
    
    return df, source_type


def plot_timeline(df: pd.DataFrame, out_path: Path):
    if df.empty: return
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="darkgrid")
    
    # Plot confidence line for each emotion
    # Strategy: Scatter plot with hue=emotion, plus a smoothed line?
    # Or just points?
    
    sns.scatterplot(data=df, x="time_sec", y="confidence", hue="emotion", s=50, alpha=0.7)
    
    plt.title("Emotion Confidence Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_distribution(df: pd.DataFrame, out_path: Path):
    if df.empty: return
    
    # Count emotions
    counts = df["emotion"].value_counts()
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("pastel")
    
    plt.pie(counts, labels=counts.index, autopct='%.1f%%', colors=colors)
    plt.title("Emotion Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots from emotion JSON.")
    parser.add_argument("--input", required=True, help="Input JSON file.")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: output/reports/<type>/<session>).")
    args = parser.parse_args()
    
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: {in_path} not found")
        sys.exit(1)
    
    print(f"Loading {in_path}...")
    df, src_type = load_data(in_path)
    print(f"Detected format: {src_type}, Records: {len(df)}")
    
    if df.empty:
        print("No valid records found.")
        return

    # Determine Output Directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        # Auto-structuring: output/reports/{type}/{session_name}
        # Try to infer session name from parent folder
        session_name = in_path.parent.name
        
        # Go up to project root (approximate)
        # Assuming src/plots.py -> project/src/plots.py -> project
        project_root = Path(__file__).resolve().parents[1]
        
        out_root = project_root / "output" / "reports" / src_type
        out_dir = out_root / session_name
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Generate plots
    plot_timeline(df, out_dir / "plot_timeline.png")
    plot_distribution(df, out_dir / "plot_distribution.png")
    
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
