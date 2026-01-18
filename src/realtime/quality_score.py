from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List


DEFAULT_ROOT = Path("output/realtime")
SUMMARY_NAME = "summary.json"


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def score_quality(summary: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    nb_frames = int(summary.get("nb_frames", 0) or 0)
    duration = float(summary.get("duree_estimee_sec", 0.0) or 0.0)
    uncertain = float(summary.get("taux_uncertain", 0.0) or 0.0)

    # Règles A/B/C
    is_A = (uncertain <= 5.0) and (duration >= 5.0) and (nb_frames >= 50)
    is_B = (uncertain <= 15.0) and (duration >= 2.0) and (nb_frames >= 20)

    if is_A:
        score = "A"
    elif is_B:
        score = "B"
    else:
        score = "C"

    details = {
        "metrics": {
            "nb_frames": nb_frames,
            "duree_estimee_sec": duration,
            "taux_uncertain": uncertain,
        },
        "rules": {
            "A": {"taux_uncertain_max": 5.0, "duree_min": 5.0, "nb_frames_min": 50},
            "B": {"taux_uncertain_max": 15.0, "duree_min": 2.0, "nb_frames_min": 20},
        },
        "passed": {
            "A": bool(is_A),
            "B": bool(is_B),
        },
    }
    return score, details


def iter_sessions(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("session_")]
    )


def main():
    ap = argparse.ArgumentParser(
        description="Ajoute un quality_score (A/B/C) dans chaque summary.json de session."
    )
    ap.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Racine des sessions (default: output/realtime)",
    )
    ap.add_argument(
        "--summary-name",
        default=SUMMARY_NAME,
        help="Nom du fichier summary (default: summary.json)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Réécrire même si quality_score existe déjà",
    )
    args = ap.parse_args()

    root = Path(args.root)

    updated = 0
    skipped = 0
    missing = 0
    errors = 0

    for session_dir in iter_sessions(root):
        sp = session_dir / args.summary_name
        if not sp.exists():
            missing += 1
            continue

        try:
            s = load_json(sp)

            if ("quality_score" in s) and not args.force:
                skipped += 1
                continue

            score, details = score_quality(s)
            s["quality_score"] = score
            s["quality_details"] = details

            save_json(sp, s)
            updated += 1
            print(f"[OK] {session_dir.name}: {score}")

        except Exception as e:
            errors += 1
            print(f"[ERROR] {session_dir.name}: {e}")

    print("\n=== RÉCAP ===")
    print(f"MAJ      : {updated}")
    print(f"Skippés  : {skipped}")
    print(f"Manquants: {missing}")
    print(f"Erreurs  : {errors}")


if __name__ == "__main__":
    main()
