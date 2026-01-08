# src/emotion_summary_report.py
# Résumé des émotions (par personne) — REGROUPÉ PAR VIDÉO (frames_... + TIMESTAMP DU RÉSUMÉ)
#
# ✅ Run sans arguments:
#   python src/emotion_summary_report.py
#
# Entrée attendue (format analyze_emotion.py):
#   output/emotion_results/<...>/latest/analyzed_emotions_final.json
#   (fallback: analyzed_emotions.json, emotion_results_master.json)
#
# Sortie:
#   output/reports/<base>/<SUMMARY_TIMESTAMP>/
#     - summary.json
#     - summary_people.csv
#     - report.txt
#
# ✅ Anti-duplication:
# - Si un résumé existe déjà pour la même vidéo (n'importe quel timestamp), on SKIP.
#
# ✅ Cas master global:
# - Si le fichier analysé est directement sous output/emotion_results (ex: emotion_results_master.json),
#   la session devient "ALL_SESSIONS_MASTER" (au lieu de unknown_session)
#
# Notes:
# - Supporte JSON dict {path: record} ET JSON list [{...}, ...]
# - person_id: global_person_id (si présent) -> identity_id -> path (person_0000) -> fallback
# - tri temporel: time_ms / _tXXXX -> frame_index -> path

import re
import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Union
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime


# =============================================================================
# Helpers
# =============================================================================

PERSON_DIR_RE = re.compile(r"^person[_-]?\d+$", re.IGNORECASE)
PERSON_IN_PATH_RE = re.compile(r"(?:^|[\\/])(person[_-]?\d+)(?:[\\/]|$)", re.IGNORECASE)
T_MS_RE = re.compile(r"_t(\d+)", re.IGNORECASE)
FRAME_RE = re.compile(r"frame[_-]?(\d+)", re.IGNORECASE)


def now_ts() -> str:
    """Timestamp du résumé (moment où on lance le script)."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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
    if ms is None or ms < 0:
        return "N/A"
    s = ms // 1000
    mmm = ms % 1000
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"


def extract_person_id(path: str, rec: Dict[str, Any]) -> str:
    """
    Priorité:
      1) global_person_id (unique cross-videos)
      2) identity_id (local)
      3) person_0000 dans le path
      4) fallback
    """
    gpid = rec.get("global_person_id", None)
    if isinstance(gpid, str) and gpid.strip():
        return gpid.strip()

    if rec.get("identity_id", None) is not None:
        return f"person_{safe_int(rec.get('identity_id'), 0):04d}"

    m = PERSON_IN_PATH_RE.search(path or "")
    if m:
        pid = m.group(1).lower().replace("-", "_")
        pid = re.sub(r"person_?(\d+)", lambda mm: f"person_{int(mm.group(1)):04d}", pid)
        return pid

    return "person_0000"


def extract_time_ms(path: str, rec: Dict[str, Any]) -> int:
    if rec.get("time_ms", None) is not None:
        return safe_int(rec.get("time_ms"), -1)

    m = T_MS_RE.search(path or "")
    if m:
        return safe_int(m.group(1), -1)

    return -1


def extract_frame_index(path: str, rec: Dict[str, Any]) -> int:
    if rec.get("frame_index", None) is not None:
        return safe_int(rec.get("frame_index"), -1)

    m = FRAME_RE.search(path or "")
    if m:
        return safe_int(m.group(1), -1)

    return -1


def choose_emotion(rec: Dict[str, Any], prefer_smoothed: bool = True) -> str:
    keys = ["smoothed_final_emotion", "final_emotion", "emotion"] if prefer_smoothed else \
           ["final_emotion", "smoothed_final_emotion", "emotion"]

    for k in keys:
        v = rec.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return "Unknown"


def choose_confidence(rec: Dict[str, Any]) -> float:
    v = rec.get("final_confidence", rec.get("confidence", None))
    c = safe_float(v, 0.0)

    if c < 0:
        c = 0.0
    if c > 1.0 and c <= 100.0:
        c = c / 100.0
    if c > 1.0:
        c = 1.0

    return c


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data model
# =============================================================================

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
    top_emotions: List[Tuple[str, float]]
    most_intense_emotion: str
    most_intense_confidence: float
    most_intense_time: str
    most_intense_frame: int


def compute_stability(emotions_in_time: List[str]) -> Tuple[float, int, float]:
    n = len(emotions_in_time)
    if n <= 1:
        return 1.0, 0, 0.0

    transitions = 0
    for i in range(1, n):
        if emotions_in_time[i] != emotions_in_time[i - 1]:
            transitions += 1

    change_rate = transitions / max(1, (n - 1))
    stability = 1.0 - change_rate
    stability = min(1.0, max(0.0, stability))
    return stability, transitions, change_rate


def summarize_person(items: List[Dict[str, Any]], person_id: str) -> PersonSummary:
    items_sorted = sorted(
        items,
        key=lambda x: (x.get("time_ms", -1), x.get("frame_index", -1), x.get("path", ""))
    )

    emotions = [it["emotion"] for it in items_sorted]
    confidences = [it["confidence"] for it in items_sorted]
    n = len(items_sorted)

    counter = Counter(emotions)
    dominant_emotion, dominant_count = counter.most_common(1)[0] if counter else ("Unknown", 0)
    dominant_ratio = (dominant_count / n) if n else 0.0

    top_emotions = [(emo, cnt / n if n else 0.0) for emo, cnt in counter.most_common(7)]
    avg_conf = (sum(confidences) / n) if n else 0.0

    stability, transitions, change_rate = compute_stability(emotions)

    best = None
    for it in items_sorted:
        if best is None or it["confidence"] > best["confidence"]:
            best = it

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
        most_intense_emotion=best["emotion"] if best else "Unknown",
        most_intense_confidence=best["confidence"] if best else 0.0,
        most_intense_time=format_time_from_ms(best["time_ms"]) if best else "N/A",
        most_intense_frame=best["frame_index"] if best else -1,
    )


# =============================================================================
# IO + compat formats
# =============================================================================

def load_json_any(path: Path) -> Union[Dict[str, Any], List[Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_master(data: Union[Dict[str, Any], List[Any]], fallback_prefix: str) -> Dict[str, Dict[str, Any]]:
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}

    if isinstance(data, list):
        out: Dict[str, Dict[str, Any]] = {}
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            key = (
                item.get("path")
                or item.get("relative_path")
                or item.get("img_path")
                or item.get("image_path")
                or item.get("file")
                or f"{fallback_prefix}/idx_{i:06d}"
            )
            out[str(key)] = item
        return out

    raise ValueError("Format JSON non supporté (attendu dict ou list).")


def write_summary_json(out_path: Path, summary: Dict[str, Any]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_people_csv(out_path: Path, people: List[PersonSummary]) -> None:
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
        "top_emotions",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ps in people:
            row = asdict(ps)
            row["top_emotions"] = "; ".join([f"{e}:{r:.3f}" for e, r in ps.top_emotions])
            w.writerow(row)


def write_report_txt(out_path: Path, people: List[PersonSummary], session_label: str) -> None:
    lines: List[str] = []
    lines.append(f"=== Rapport Résumé Émotions (Étape 1) — Session: {session_label} ===\n")
    for ps in sorted(people, key=lambda x: x.person_id):
        lines.append(f"[{ps.person_id}]")
        lines.append(f"- Frames analysées : {ps.n_frames}")
        lines.append(f"- Émotion dominante : {ps.dominant_emotion} ({ps.dominant_ratio*100:.1f}%)")
        lines.append(f"- Confiance moyenne : {ps.avg_confidence:.3f}")
        lines.append(f"- Stabilité : {ps.stability_score:.3f} (change_rate={ps.change_rate:.3f}, transitions={ps.n_transitions})")
        lines.append(
            f"- Moment le plus intense : {ps.most_intense_emotion} "
            f"(conf={ps.most_intense_confidence:.3f}) à {ps.most_intense_time} "
            f"(frame={ps.most_intense_frame})"
        )
        lines.append("- Distribution (top) : " + ", ".join([f"{e} {r*100:.1f}%" for e, r in ps.top_emotions]))
        lines.append("")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# Grouping: base key (ignore run dir + ignore person_XXXX) + master global handling
# =============================================================================

def video_base_key(input_root: Path, json_path: Path) -> Tuple[str, Path]:
    """
    Retourne une base de session SANS timestamp:
      output/emotion_results/<...>/person_0000/<run>/file.json
    -> base = <...> (sans person_XXXX, sans <run>, sans file)

    Cas spécial:
      output/emotion_results/emotion_results_master.json
    -> base vide => ALL_SESSIONS_MASTER
    """
    rel = json_path.relative_to(input_root)
    parts = list(rel.parts)

    # remove file
    if len(parts) >= 1:
        parts = parts[:-1]
    # remove run dir
    if len(parts) >= 1:
        parts = parts[:-1]

    parts_no_person = [p for p in parts if not PERSON_DIR_RE.match(p)]

    if not parts_no_person:
        rel_out = Path("ALL_SESSIONS_MASTER")
        label = "ALL_SESSIONS_MASTER"
        return label, rel_out

    rel_out = Path(*parts_no_person)
    label = str(rel_out).replace("\\", "/")
    return label, rel_out


def any_existing_summary(out_base_dir: Path) -> bool:
    """SKIP si on trouve déjà un summary.json dans un sous-dossier timestamp."""
    if not out_base_dir.exists():
        return False
    for p in out_base_dir.rglob("summary.json"):
        if p.is_file():
            return True
    return False


# =============================================================================
# Pipeline
# =============================================================================

def build_enriched_records(master: Dict[str, Dict[str, Any]], prefer_smoothed: bool) -> Dict[str, List[Dict[str, Any]]]:
    per_person: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

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


def run_one_video(group_label: str, files: List[Path], input_root: Path, out_root: Path, prefer_smoothed: bool, summary_ts: str) -> str:
    """
    Fusionne tous les JSON de la même base.
    Sortie: output/reports/<base>/<summary_ts>/
    SKIP si un résumé existe déjà pour cette base.
    """
    _, rel_base_dir = video_base_key(input_root, files[0])
    out_base_dir = out_root / rel_base_dir

    # ✅ SKIP si résumé existe déjà (quelque soit timestamp)
    if any_existing_summary(out_base_dir):
        return "skipped"

    out_dir = out_base_dir / summary_ts
    ensure_dir(out_dir)

    merged_master: Dict[str, Dict[str, Any]] = {}

    for fpath in files:
        data = load_json_any(fpath)
        master = normalize_master(data, fallback_prefix=str(fpath.parent))

        for k, v in master.items():
            kk = k
            if kk in merged_master:
                kk = f"{k}__{fpath.parent.name}"
                n = 2
                while kk in merged_master:
                    kk = f"{k}__{fpath.parent.name}__{n}"
                    n += 1
            merged_master[kk] = v

    per_person = build_enriched_records(merged_master, prefer_smoothed=prefer_smoothed)

    people_summaries: List[PersonSummary] = []
    for pid, items in per_person.items():
        if items:
            people_summaries.append(summarize_person(items, pid))

    total_frames = sum(ps.n_frames for ps in people_summaries)

    global_counter = Counter()
    for items in per_person.values():
        for it in items:
            global_counter[it["emotion"]] += 1
    global_dominant = global_counter.most_common(1)[0][0] if global_counter else "Unknown"

    summary = {
        "session": group_label,
        "summary_timestamp": summary_ts,
        "inputs": [str(p.resolve()) for p in files],
        "prefer_smoothed": prefer_smoothed,
        "n_people": len(people_summaries),
        "total_frames": total_frames,
        "global_dominant_emotion": global_dominant,
        "global_distribution": {k: (v / total_frames if total_frames else 0.0) for k, v in global_counter.items()},
        "people": [
            {
                **asdict(ps),
                "top_emotions": [{"emotion": e, "ratio": r} for e, r in ps.top_emotions],
            }
            for ps in sorted(people_summaries, key=lambda x: x.person_id)
        ],
    }

    write_summary_json(out_dir / "summary.json", summary)
    write_people_csv(out_dir / "summary_people.csv", people_summaries)
    write_report_txt(out_dir / "report.txt", people_summaries, session_label=group_label)

    return "ok"


# =============================================================================
# Main (AUTO)
# =============================================================================

def main() -> None:
    input_root = Path("output/emotion_results")
    out_root = Path("output/reports")

    # On scanne les outputs par dossier (latest/...) + le master global éventuel
    candidates = [
        "analyzed_emotions_final.json",
        "analyzed_emotions.json",
        "emotion_results_master.json",
    ]

    prefer_smoothed = True
    ensure_dir(out_root)

    if not input_root.exists():
        print("[ERREUR] Dossier introuvable :", input_root.resolve())
        return

    summary_ts = now_ts()

    # collect tous les fichiers candidats
    all_files: List[Path] = []
    for name in candidates:
        all_files.extend(list(input_root.rglob(name)))
    all_files = [p for p in set(all_files) if p.is_file()]
    all_files.sort(key=lambda p: str(p).lower())

    if not all_files:
        print("[ERREUR] Aucune session trouvée.")
        print(" - Dossier scanné :", input_root.resolve())
        print(" - Fichiers cherchés :", ", ".join(candidates))
        return

    # group par base (ignore run dir + ignore person_XXXX)
    groups: Dict[str, List[Path]] = defaultdict(list)
    for f in all_files:
        label, _ = video_base_key(input_root, f)
        groups[label].append(f)

    ok = skipped = failed = 0

    for label in sorted(groups.keys()):
        files = sorted(groups[label], key=lambda p: str(p).lower())
        try:
            status = run_one_video(
                group_label=label,
                files=files,
                input_root=input_root,
                out_root=out_root,
                prefer_smoothed=prefer_smoothed,
                summary_ts=summary_ts
            )
            if status == "skipped":
                skipped += 1
                print(f"[SKIP] {label} (résumé déjà existant)")
            else:
                ok += 1
                print(f"[OK] {label} -> timestamp={summary_ts} (n_files={len(files)})")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {label} -> {type(e).__name__}: {e}")

    print("\n=== Résumé (auto) ===")
    print(f"Groupes trouvés  : {len(groups)}")
    print(f"OK              : {ok}")
    print(f"SKIP            : {skipped}")
    print(f"Échecs          : {failed}")
    print(f"Sortie          : {out_root.resolve()}")


if __name__ == "__main__":
    main()
