from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Optional, Iterable, Tuple


# =========================
# CONFIG (pipeline)
# =========================
DEFAULT_ROOT = Path("output/realtime")
DEFAULT_SESSION_JSON = "realtime_emotions.json"
DEFAULT_SUMMARY_NAME = "summary.json"

# labels considérés comme "pas une émotion exploitable"
UNKNOWN_LABELS = {"unknown", "none", ""}


# =========================
# CORE (SRP: calcul résumé)
# =========================
def _normalize_label(v: Any) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    if not v:
        return None
    return v


def _pick_emotion(rec: Dict[str, Any]) -> Optional[str]:
    """
    IMPORTANT (Realtime):
    realtime_analysis.py enregistre la clé 'emotion'.

    On garde aussi les clés "offline" pour compatibilité.
    """
    for k in (
        "emotion",  # realtime
        "smoothed_final_emotion",  # offline
        "final_emotion",  # offline
        "hse_emotion",  # offline
        "deepface_emotion",  # offline
    ):
        if k in rec:
            return _normalize_label(rec.get(k))
    return None


def _pick_confidence(rec: Dict[str, Any]) -> Optional[float]:
    for k in (
        "confidence",
        "final_confidence",
        "hse_confidence",
        "deepface_confidence",
    ):
        if k in rec and rec.get(k) is not None:
            try:
                return float(rec[k])
            except Exception:
                return None
    return None


def _is_uncertain(rec: Dict[str, Any], emotion: Optional[str]) -> bool:
    """
    Realtime:
    - on se base en priorité sur le flag 'is_uncertain'
    - sinon fallback: émotion explicitement 'Uncertain'
    - sinon fallback: confiance <= 0
    """
    if rec.get("is_uncertain") is True:
        return True

    if isinstance(emotion, str) and emotion.strip().lower() == "uncertain":
        return True

    conf = _pick_confidence(rec)
    if conf is not None and conf <= 0.0:
        return True

    return False


def _extract_time_seconds(rec: Dict[str, Any]) -> Optional[float]:
    """
    Realtime:
    - 't_rel_ms' (meilleur, temps relatif depuis début)
    - sinon 'time_ms'
    Offline éventuel:
    - timestamp_sec, t_sec, time_sec, ts
    """
    if "t_rel_ms" in rec:
        try:
            return float(rec["t_rel_ms"]) / 1000.0
        except Exception:
            return None

    if "time_ms" in rec:
        try:
            return float(rec["time_ms"]) / 1000.0
        except Exception:
            return None

    for k in ("timestamp_sec", "t_sec", "time_sec", "ts"):
        if k in rec:
            try:
                return float(rec[k])
            except Exception:
                return None

    return None


def _load_records(data: Any) -> list[Dict[str, Any]]:
    """
    Supporte plusieurs formats:
    - list[record, ...]
    - dict avec 'records' (realtime actuel)
    - dict avec 'frames'
    - dict type {"0": rec, "1": rec, ...}
    """
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]

    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            return [r for r in data["records"] if isinstance(r, dict)]
        if "frames" in data and isinstance(data["frames"], list):
            return [r for r in data["frames"] if isinstance(r, dict)]
        # dict classique
        return [r for r in data.values() if isinstance(r, dict)]

    return []


def summarize_json(json_path: Path) -> Dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
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

    emotions: list[str] = []
    uncertain_count = 0
    times: list[float] = []

    for rec in records:
        emo = _pick_emotion(rec)

        # si pas d'émotion, on considère unknown (mais on ne veut pas que ça devienne dominant)
        emo_norm = emo if emo is not None else "unknown"
        emotions.append(emo_norm)

        if _is_uncertain(rec, emo):
            uncertain_count += 1

        t = _extract_time_seconds(rec)
        if t is not None:
            times.append(t)

    nb = len(emotions)
    counts = Counter(emotions)

    # dominante: on ignore unknown + Uncertain si possible
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
        else (counts.most_common(1)[0][0] if counts else None)
    )

    # % par émotion (on garde tout, même uncertain/unknown, c'est informatif)
    perc = {emo: round((c / nb) * 100.0, 2) for emo, c in counts.items()} if nb else {}

    # durée:
    # - si on a des temps relatifs, la durée = max(times)
    # - sinon fallback = max-min
    duration = 0.0
    if len(times) >= 1:
        tmin, tmax = min(times), max(times)
        # si on est en relatif (t_rel_ms), tmin devrait être proche de 0
        duration = tmax if tmin <= 0.001 else (tmax - tmin)
        if duration < 0:
            duration = 0.0

    uncertain_rate = round((uncertain_count / nb) * 100.0, 2) if nb > 0 else 0.0

    return {
        "source_json": json_path.name,
        "nb_frames": nb,
        "emotion_dominante": dominant,
        "pourcentages_par_emotion": perc,
        "duree_estimee_sec": round(duration, 3),
        "taux_uncertain": uncertain_rate,
    }


def write_summary(json_path: Path, output_name: str = DEFAULT_SUMMARY_NAME) -> Path:
    summary = summarize_json(json_path)
    out_path = json_path.parent / output_name
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


# =========================
# PIPELINE HELPERS
# =========================
def _iter_sessions(root: Path, session_json_name: str) -> Iterable[Tuple[Path, Path]]:
    if not root.exists():
        return
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        json_path = session_dir / session_json_name
        if json_path.exists():
            yield session_dir, json_path


def run_one_session(
    root: Path, session_name: str, session_json_name: str, out_name: str
) -> None:
    session_dir = root / session_name
    if not session_dir.exists():
        raise FileNotFoundError(f"Session inexistante: {session_dir}")

    json_path = session_dir / session_json_name
    if not json_path.exists():
        raise FileNotFoundError(f"Fichier manquant: {json_path}")

    out = write_summary(json_path, out_name)
    print(f"[OK] summary écrit: {out}")


def run_all_sessions(
    root: Path, session_json_name: str, out_name: str, force: bool
) -> None:
    done = 0
    skipped = 0
    errors = 0

    for session_dir, json_path in _iter_sessions(root, session_json_name):
        summary_path = session_dir / out_name

        if summary_path.exists() and not force:
            print(f"[SKIP] {session_dir.name} (déjà présent)")
            skipped += 1
            continue

        try:
            write_summary(json_path, out_name)
            print(f"[OK] {session_dir.name}")
            done += 1
        except Exception as e:
            print(f"[ERROR] {session_dir.name}: {e}")
            errors += 1

    print("\n=== RÉCAP ===")
    print(f"Créés    : {done}")
    print(f"Skippés  : {skipped}")
    print(f"Erreurs  : {errors}")


# =========================
# CLI
# =========================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline summarize: résume une session (--session) ou toutes (--all)."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--session", help="Nom du dossier session (ex: session_2026-01-10_22-41-06)"
    )
    mode.add_argument(
        "--all", action="store_true", help="Résumer toutes les sessions sous le root"
    )

    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Dossier racine des sessions (default: output/realtime)",
    )
    parser.add_argument(
        "--session-json",
        default=DEFAULT_SESSION_JSON,
        help="Nom du fichier JSON dans chaque session (default: realtime_emotions.json)",
    )
    parser.add_argument(
        "--out-name",
        default=DEFAULT_SUMMARY_NAME,
        help="Nom du fichier summary (default: summary.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Régénérer même si summary existe (utile avec --all)",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.root)

    if args.session:
        run_one_session(root, args.session, args.session_json, args.out_name)
    else:
        run_all_sessions(root, args.session_json, args.out_name, args.force)
