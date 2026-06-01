from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple


DEFAULT_ROOT = Path("output/realtime")
DEFAULT_SESSION_JSON = "realtime_emotions.json"
DEFAULT_SUMMARY_NAME = "summary.json"
DEFAULT_MASTER_NAME = "summary_master.json"


# =========================
# Utils JSON
# =========================
def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_session_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("session_")]
    )


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# =========================
# Summary "per session"
# (we rely on your realtime_analysis.py format)
# =========================
UNKNOWN_LABELS = {"unknown", "none", ""}


def _normalize_label(v: Any) -> str:
    if v is None:
        return "unknown"
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    return v if v else "unknown"


def _load_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return [r for r in data["records"] if isinstance(r, dict)]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [r for r in data.values() if isinstance(r, dict)]
    return []


def summarize_realtime_json(json_path: Path) -> Dict[str, Any]:
    data = _load_json(json_path)
    records = _load_records(data)

    if not records:
        return {
            "source_json": json_path.name,
            "nb_frames": 0,
            "emotion_dominante": None,
            "pourcentages_par_emotion": {},
            "duree_estimee_sec": 0.0,
            "taux_uncertain": 0.0,
        }

    emotions: List[str] = []
    uncertain_count = 0
    times_sec: List[float] = []

    for r in records:
        emo = _normalize_label(r.get("emotion"))
        emotions.append(emo)

        if r.get("is_uncertain") is True or emo.lower() == "uncertain":
            uncertain_count += 1

        # duration: t_rel_ms priority, otherwise time_ms
        if "t_rel_ms" in r and r["t_rel_ms"] is not None:
            try:
                times_sec.append(float(r["t_rel_ms"]) / 1000.0)
            except Exception:
                pass
        elif "time_ms" in r and r["time_ms"] is not None:
            try:
                times_sec.append(float(r["time_ms"]) / 1000.0)
            except Exception:
                pass

    nb = len(emotions)
    counts = Counter(emotions)

    # dominant: ignore unknown + Uncertain if possible
    counts_for_dom = Counter(
        {
            k: v
            for k, v in counts.items()
            if (k.lower() != "uncertain") and (k.lower() not in UNKNOWN_LABELS)
        }
    )
    dominant = (
        counts_for_dom.most_common(1)[0][0]
        if counts_for_dom
        else counts.most_common(1)[0][0]
    )

    perc = {k: round((v / nb) * 100.0, 2) for k, v in counts.items()} if nb else {}

    duration = 0.0
    if times_sec:
        tmin, tmax = min(times_sec), max(times_sec)
        duration = tmax if tmin <= 0.001 else (tmax - tmin)
        if duration < 0:
            duration = 0.0

    uncertain_rate = round((uncertain_count / nb) * 100.0, 2) if nb else 0.0

    return {
        "source_json": json_path.name,
        "nb_frames": nb,
        "emotion_dominante": dominant,
        "pourcentages_par_emotion": perc,
        "duree_estimee_sec": round(duration, 3),
        "taux_uncertain": uncertain_rate,
    }


def ensure_session_summary(
    session_dir: Path, session_json_name: str, summary_name: str, force: bool
) -> Tuple[bool, str]:
    summary_path = session_dir / summary_name
    json_path = session_dir / session_json_name

    if not json_path.exists():
        return False, "no_session_json"

    if summary_path.exists() and not force:
        return False, "already_exists"

    try:
        summary = summarize_realtime_json(json_path)
        _write_json(summary_path, summary)
        return True, "written"
    except Exception:
        return False, "error"


# =========================
# Rankings + Quality
# =========================
def _get_emotion_percent(summary: Dict[str, Any], emotion: str) -> float:
    percs = summary.get("pourcentages_par_emotion", {}) or {}
    if not isinstance(percs, dict):
        return 0.0

    if emotion in percs:
        return _safe_float(percs[emotion], 0.0)

    # case fallback
    for k, v in percs.items():
        if str(k).lower() == emotion.lower():
            return _safe_float(v, 0.0)
    return 0.0


def _quality_breakdown(summaries: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, int]:
    c = Counter()
    for _, s in summaries:
        q = s.get("quality_score", None)
        if q is None:
            c["MISSING"] += 1
        else:
            c[str(q)] += 1
    return {
        "A": int(c.get("A", 0)),
        "B": int(c.get("B", 0)),
        "C": int(c.get("C", 0)),
        "MISSING": int(c.get("MISSING", 0)),
    }


def _build_rankings(
    summaries: List[Tuple[str, Dict[str, Any]]], emotion_focus: str, top_k: int
) -> Dict[str, Any]:
    items = []
    for session_name, s in summaries:
        nb = _safe_int(s.get("nb_frames", 0), 0)
        dur = _safe_float(s.get("duree_estimee_sec", 0.0), 0.0)
        unc = _safe_float(s.get("taux_uncertain", 0.0), 0.0)
        dom = s.get("emotion_dominante", None)
        q = s.get("quality_score", None)
        emo_pct = _get_emotion_percent(s, emotion_focus)

        items.append(
            {
                "session": session_name,
                "nb_frames": nb,
                "duree_estimee_sec": dur,
                "taux_uncertain": unc,
                "emotion_dominante": dom,
                "quality_score": q,
                f"{emotion_focus}_percent": emo_pct,
            }
        )

    return {
        "params": {"emotion_focus": emotion_focus, "top_k": top_k},
        "top_by_duration": sorted(
            items, key=lambda x: x["duree_estimee_sec"], reverse=True
        )[:top_k],
        "top_by_frames": sorted(items, key=lambda x: x["nb_frames"], reverse=True)[
            :top_k
        ],
        "best_by_uncertain": sorted(items, key=lambda x: x["taux_uncertain"])[:top_k],
        "worst_by_uncertain": sorted(
            items, key=lambda x: x["taux_uncertain"], reverse=True
        )[:top_k],
        f"top_by_{emotion_focus.lower()}_percent": sorted(
            items, key=lambda x: x[f"{emotion_focus}_percent"], reverse=True
        )[:top_k],
    }


# =========================
# Master aggregation
# =========================
def aggregate_master(
    root: Path, summary_name: str, emotion_focus: str, top_k: int
) -> Dict[str, Any]:
    session_dirs = _iter_session_dirs(root)

    summaries_ok: List[Tuple[str, Dict[str, Any]]] = []
    sessions_sans_summary: List[str] = []

    for d in session_dirs:
        sp = d / summary_name
        if not sp.exists():
            sessions_sans_summary.append(d.name)
            continue
        try:
            summaries_ok.append((d.name, _load_json(sp)))
        except Exception:
            sessions_sans_summary.append(d.name)

    total_frames = 0
    total_duration = 0.0
    uncertain_weighted_sum = 0.0
    emotion_counts = Counter()

    for session_name, s in summaries_ok:
        nb = _safe_int(s.get("nb_frames", 0), 0)
        total_frames += nb
        total_duration += _safe_float(s.get("duree_estimee_sec", 0.0), 0.0)

        taux_unc = _safe_float(s.get("taux_uncertain", 0.0), 0.0)  # %
        uncertain_weighted_sum += nb * taux_unc

        percs = s.get("pourcentages_par_emotion", {}) or {}
        if isinstance(percs, dict):
            for emo, pct in percs.items():
                emotion_counts[str(emo)] += nb * (_safe_float(pct, 0.0) / 100.0)

    if total_frames <= 0:
        master = {
            "nb_sessions": len(summaries_ok),
            "sessions_incluses": [name for name, _ in summaries_ok],
            "sessions_sans_summary": sessions_sans_summary,
            "nb_frames_total": 0,
            "duree_totale_sec": 0.0,
            "emotion_dominante_globale": None,
            "pourcentages_par_emotion_global": {},
            "taux_uncertain_moyen": 0.0,
        }
    else:
        global_perc = {
            emo: round((cnt / total_frames) * 100.0, 2)
            for emo, cnt in emotion_counts.items()
        }
        dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else None
        uncertain_avg = round((uncertain_weighted_sum / total_frames), 2)

        master = {
            "nb_sessions": len(summaries_ok),
            "sessions_incluses": [name for name, _ in summaries_ok],
            "sessions_sans_summary": sessions_sans_summary,
            "nb_frames_total": total_frames,
            "duree_totale_sec": round(total_duration, 3),
            "emotion_dominante_globale": dominant,
            "pourcentages_par_emotion_global": global_perc,
            "taux_uncertain_moyen": uncertain_avg,
        }

    master["root"] = str(root)
    master["summary_name"] = summary_name

    # ✅ Requested additions
    master["quality_breakdown"] = _quality_breakdown(summaries_ok)
    master["rankings"] = _build_rankings(
        summaries_ok, emotion_focus=emotion_focus, top_k=top_k
    )

    return master


# =========================
# Main (all-in-one)
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="All-in-one: generates missing summary.json then summary_master.json (+rankings)"
    )
    ap.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Root sessions folder (default: output/realtime)",
    )
    ap.add_argument(
        "--session-json",
        default=DEFAULT_SESSION_JSON,
        help="Session JSON name (default: realtime_emotions.json)",
    )
    ap.add_argument(
        "--summary-name",
        default=DEFAULT_SUMMARY_NAME,
        help="Summary name (default: summary.json)",
    )
    ap.add_argument(
        "--master-name",
        default=DEFAULT_MASTER_NAME,
        help="Master name (default: summary_master.json)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Regenerate summary.json even if they exist",
    )
    ap.add_argument(
        "--emotion-focus",
        default="Happiness",
        help="Emotion used for top_by_<emotion>_percent ranking",
    )
    ap.add_argument(
        "--top-k", type=int, default=5, help="Number of items per ranking (default: 5)"
    )
    args = ap.parse_args()

    root = Path(args.root)

    # 1) Ensure summaries
    created = 0
    skipped = 0
    no_json = 0
    errors = 0

    for d in _iter_session_dirs(root):
        _, reason = ensure_session_summary(
            d, args.session_json, args.summary_name, args.force
        )
        if reason == "written":
            created += 1
        elif reason == "already_exists":
            skipped += 1
        elif reason == "no_session_json":
            no_json += 1
        else:
            errors += 1

    print("=== SUMMARY PER SESSION ===")
    print(f"Created/UPDATED : {created}")
    print(f"Skipped         : {skipped}")
    print(f"No JSON         : {no_json}")
    print(f"Errors          : {errors}")

    # 2) Master (+ rankings)
    master = aggregate_master(root, args.summary_name, args.emotion_focus, args.top_k)
    out_path = root / args.master_name
    _write_json(out_path, master)
    print(f"\n[OK] master written: {out_path}")


if __name__ == "__main__":
    main()
