# summarize_emotions.py
# Étape 1 — Résumé intelligent des émotions (par personne)
# Usage:
#   python summarize_emotions.py --input master_results.json --outdir out_summary
#
# Sorties:
#   out_summary/summary.json
#   out_summary/summary_people.csv
#   out_summary/report.txt

import os
import re
import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

# -----------------------------
# Helpers: parsing & time
# -----------------------------

PERSON_RE = re.compile(r"(?:^|[\\/])(person[_-]?\d+)(?:[\\/]|$)", re.IGNORECASE)
T_MS_RE = re.compile(r"_t(\d+)", re.IGNORECASE)       # ex: _t00000200
FRAME_RE = re.compile(r"frame[_-]?(\d+)", re.IGNORECASE)

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=-1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def format_time_from_ms(ms: int) -> str:
    """ms -> HH:MM:SS.mmm"""
    if ms is None or ms < 0:
        return "N/A"
    s = ms // 1000
    mmm = ms % 1000
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"

def extract_person_id(path: str, rec: Dict[str, Any]) -> str:
    # 1) identity_id si présent
    if "identity_id" in rec and rec["identity_id"] is not None:
        return f"person_{safe_int(rec['identity_id'], 0):04d}"
    # 2) sinon, essayer de trouver "person_0000" dans le path
    m = PERSON_RE.search(path or "")
    if m:
        pid = m.group(1).lower().replace("-", "_")
        # normaliser "person0000" -> "person_0000" si besoin
        pid = re.sub(r"person_?(\d+)", lambda mm: f"person_{int(mm.group(1)):04d}", pid)
        return pid
    # 3) fallback
    return "person_0000"

def extract_time_ms(path: str, rec: Dict[str, Any]) -> int:
    # 1) si une clé time_ms existe
    if "time_ms" in rec and rec["time_ms"] is not None:
        return safe_int(rec["time_ms"], -1)

    # 2) parse depuis filename: _t00000200
    m = T_MS_RE.search(path or "")
    if m:
        return safe_int(m.group(1), -1)

    # 3) fallback: si frame_index et fps
    # (pas fiable sans fps; on laisse -1)
    return -1

def extract_frame_index(path: str, rec: Dict[str, Any]) -> int:
    if "frame_index" in rec and rec["frame_index"] is not None:
        return safe_int(rec["frame_index"], -1)
    m = FRAME_RE.search(path or "")
    if m:
        return safe_int(m.group(1), -1)
    return -1

def choose_emotion(rec: Dict[str, Any], prefer_smoothed: bool = True) -> str:
    # priorité: smoothed_final_emotion -> final_emotion -> emotion
    keys = []
    if prefer_smoothed:
        keys += ["smoothed_final_emotion", "final_emotion", "emotion"]
    else:
        keys += ["final_emotion", "smoothed_final_emotion", "emotion"]
    for k in keys:
        v = rec.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "Unknown"

def choose_confidence(rec: Dict[str, Any]) -> float:
    # final_confidence sinon confidence
    v = rec.get("final_confidence", rec.get("confidence", None))
    c = safe_float(v, 0.0)
    # clamp
    if c < 0: c = 0.0
    if c > 1.0 and c <= 100.0:
        # certains backends retournent 0..100
        c = c / 100.0
    if c > 1.0:
        c = 1.0
    return c

# -----------------------------
# Stats / summary
# -----------------------------

@dataclass
class PersonSummary:
    person_id: str
    n_frames: int
    dominant_emotion: str
    dominant_ratio: float
    avg_confidence: float
    stability_score: float
    n_transitions: int
    change_rate: float
    top_emotions: List[Tuple[str, float]]  # (emotion, ratio)
    most_intense_emotion: str
    most_intense_confidence: float
    most_intense_time: str
    most_intense_frame: int

def compute_stability(emotions_in_time: List[str]) -> Tuple[float, int, float]:
    """
    Retourne (stability_score, n_transitions, change_rate)
    - transitions: nombre de fois où l'émotion change entre t-1 et t
    - change_rate: transitions / (n-1)
    - stability_score: 1 - change_rate (0..1)
    """
    n = len(emotions_in_time)
    if n <= 1:
        return 1.0, 0, 0.0
    transitions = 0
    for i in range(1, n):
        if emotions_in_time[i] != emotions_in_time[i - 1]:
            transitions += 1
    change_rate = transitions / max(1, (n - 1))
    stability = 1.0 - change_rate
    if stability < 0: stability = 0.0
    if stability > 1: stability = 1.0
    return stability, transitions, change_rate

def summarize_person(items: List[Dict[str, Any]], person_id: str) -> PersonSummary:
    """
    items: liste de dicts enrichis: {path, frame_index, time_ms, emotion, confidence}
    """
    # trier par temps / frame
    items_sorted = sorted(
        items,
        key=lambda x: (x.get("time_ms", -1), x.get("frame_index", -1), x.get("path", ""))
    )

    emotions = [it["emotion"] for it in items_sorted]
    confidences = [it["confidence"] for it in items_sorted]
    n = len(items_sorted)

    counter = Counter(emotions)
    dominant_emotion, dominant_count = counter.most_common(1)[0] if counter else ("Unknown", 0)
    dominant_ratio = dominant_count / n if n else 0.0

    # top emotions ratios
    top_emotions = []
    for emo, cnt in counter.most_common(7):
        top_emotions.append((emo, cnt / n if n else 0.0))

    avg_conf = sum(confidences) / n if n else 0.0

    stability, transitions, change_rate = compute_stability(emotions)

    # "moment le plus intense" = max confidence globale (ou sur émotion dominante si tu préfères)
    best = None
    for it in items_sorted:
        if best is None or it["confidence"] > best["confidence"]:
            best = it

    most_intense_emotion = best["emotion"] if best else "Unknown"
    most_intense_confidence = best["confidence"] if best else 0.0
    most_intense_time = format_time_from_ms(best["time_ms"]) if best else "N/A"
    most_intense_frame = best["frame_index"] if best else -1

    return PersonSummary(
        person_id=person_id,
        n_frames=n,
        dominant_emotion=dominant_emotion,
        dominant_ratio=dominant_ratio,
        avg_confidence=avg_conf,
        stability_score=stability,
        n_transitions=transitions,
        change_rate=change_rate,
        top_emotions=top_emotions,
        most_intense_emotion=most_intense_emotion,
        most_intense_confidence=most_intense_confidence,
        most_intense_time=most_intense_time,
        most_intense_frame=most_intense_frame,
    )

# -----------------------------
# IO
# -----------------------------

def load_master_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Le JSON maître doit être un objet (dict) {path: record}.")
    return data

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_summary_json(out_path: str, summary: Dict[str, Any]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def write_people_csv(out_path: str, people: List[PersonSummary]) -> None:
    fieldnames = [
        "person_id",
        "n_frames",
        "dominant_emotion",
        "dominant_ratio",
        "avg_confidence",
        "stability_score",
        "n_transitions",
        "change_rate",
        "most_intense_emotion",
        "most_intense_confidence",
        "most_intense_time",
        "most_intense_frame",
        "top_emotions",  # string
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ps in people:
            row = asdict(ps)
            row["top_emotions"] = "; ".join([f"{e}:{r:.3f}" for e, r in ps.top_emotions])
            w.writerow(row)

def write_report_txt(out_path: str, people: List[PersonSummary]) -> None:
    lines = []
    lines.append("=== Rapport Résumé Émotions (Étape 1) ===\n")
    for ps in sorted(people, key=lambda x: x.person_id):
        lines.append(f"[{ps.person_id}]")
        lines.append(f"- Frames analysées : {ps.n_frames}")
        lines.append(f"- Émotion dominante : {ps.dominant_emotion} ({ps.dominant_ratio*100:.1f}%)")
        lines.append(f"- Confiance moyenne : {ps.avg_confidence:.3f}")
        lines.append(f"- Stabilité : {ps.stability_score:.3f} (change_rate={ps.change_rate:.3f}, transitions={ps.n_transitions})")
        lines.append(f"- Moment le plus intense : {ps.most_intense_emotion} (conf={ps.most_intense_confidence:.3f}) à {ps.most_intense_time} (frame={ps.most_intense_frame})")
        lines.append(f"- Distribution (top) : " + ", ".join([f"{e} {r*100:.1f}%" for e, r in ps.top_emotions]))
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -----------------------------
# Main pipeline
# -----------------------------

def build_enriched_records(master: Dict[str, Any], prefer_smoothed: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retourne {person_id: [ {path, frame_index, time_ms, emotion, confidence} ... ]}
    """
    per_person = defaultdict(list)

    for path, rec in master.items():
        if not isinstance(rec, dict):
            continue

        emotion = choose_emotion(rec, prefer_smoothed=prefer_smoothed)
        conf = choose_confidence(rec)
        pid = extract_person_id(path, rec)
        tms = extract_time_ms(path, rec)
        fi = extract_frame_index(path, rec)

        per_person[pid].append({
            "path": path,
            "frame_index": fi,
            "time_ms": tms,
            "emotion": emotion,
            "confidence": conf,
        })

    return per_person

def run(input_json: str, outdir: str, prefer_smoothed: bool = True) -> None:
    ensure_dir(outdir)

    master = load_master_json(input_json)
    per_person = build_enriched_records(master, prefer_smoothed=prefer_smoothed)

    people_summaries: List[PersonSummary] = []
    for pid, items in per_person.items():
        if not items:
            continue
        people_summaries.append(summarize_person(items, pid))

    # résumé global
    total_frames = sum(ps.n_frames for ps in people_summaries)
    global_dominant = None
    global_counter = Counter()
    for pid, items in per_person.items():
        for it in items:
            global_counter[it["emotion"]] += 1
    if global_counter:
        global_dominant = global_counter.most_common(1)[0][0]
    else:
        global_dominant = "Unknown"

    summary = {
        "input": os.path.abspath(input_json),
        "prefer_smoothed": prefer_smoothed,
        "n_people": len(people_summaries),
        "total_frames": total_frames,
        "global_dominant_emotion": global_dominant,
        "global_distribution": {k: v / total_frames if total_frames else 0.0 for k, v in global_counter.items()},
        "people": [
            {
                **asdict(ps),
                "top_emotions": [{"emotion": e, "ratio": r} for e, r in ps.top_emotions],
            }
            for ps in sorted(people_summaries, key=lambda x: x.person_id)
        ],
    }

    write_summary_json(os.path.join(outdir, "summary.json"), summary)
    write_people_csv(os.path.join(outdir, "summary_people.csv"), people_summaries)
    write_report_txt(os.path.join(outdir, "report.txt"), people_summaries)

    print(f"[OK] Résumé généré dans: {outdir}")
    print(f" - summary.json")
    print(f" - summary_people.csv")
    print(f" - report.txt")

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Étape 1 — Résumé intelligent des émotions (par personne)")
    ap.add_argument("--input", required=True, help="Chemin vers le JSON maître (path -> record)")
    ap.add_argument("--outdir", default="out_summary", help="Dossier de sortie")
    ap.add_argument("--prefer_smoothed", action="store_true", help="Utilise smoothed_final_emotion en priorité")
    ap.add_argument("--prefer_raw", action="store_true", help="Utilise final_emotion en priorité (ignore smoothing)")

    args = ap.parse_args()

    prefer_smoothed = True
    if args.prefer_raw:
        prefer_smoothed = False
    if args.prefer_smoothed:
        prefer_smoothed = True

    run(args.input, args.outdir, prefer_smoothed=prefer_smoothed)
