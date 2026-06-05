"""
workflows/config.py — shared configuration for the unified TeleLogs CLI.

One place for everything the per-model scripts used to each redeclare: data
paths, the feature-set vocabulary, the target/label space, and the
split/seed/validation-ratio defaults. Edit here, not in the runners.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (TeleLogs-specific). Override on the CLI with --train-file / --test-file
# / --output-root if you move the data or want a different run folder.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]

TRAIN_FILE = Path(os.getenv(
    "DSSP_TRAIN_FILE",
    _REPO_ROOT / "project/data/preprocessed/TeleLogs/train.parquet"))
TEST_FILE = Path(os.getenv(
    "DSSP_TEST_FILE",
    _REPO_ROOT / "project/data/preprocessed/TeleLogs/test.parquet"))
OUTPUT_ROOT = Path(os.getenv(
    "DSSP_OUTPUT_ROOT",
    _REPO_ROOT / "project/curriculum/TeleLogs/outputs_workflow"))

# Study artifact database (one append-only SQLite file across runs). Override
# with DSSP_STUDY_DB or the CLI's --study-db. Defaults under OUTPUT_ROOT.
STUDY_DB = Path(os.getenv("DSSP_STUDY_DB", OUTPUT_ROOT / "study.db"))

# ---------------------------------------------------------------------------
# Target / label space
# ---------------------------------------------------------------------------
TARGET = "answer"
CLASS_LABELS = [f"C{i}" for i in range(1, 9)]   # fixed 8-class order for matrices

# ---------------------------------------------------------------------------
# Split / reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VALIDATION_RATIO = 0.30     # fraction of TRAIN carved off as validation when
                            # the CLI runs in the default (no test file) mode
CV_SPLITS = 5               # k for the within-train CV that tunes each family
AVERAGE = "macro"           # averaging for precision/recall/F1 and the CV scorer
SCORING = "f1_macro"

# ---------------------------------------------------------------------------
# Feature sets — the candidate input groups. The CLI's --feature-sets argument
# selects among these keys; default is all of them.
# ---------------------------------------------------------------------------
FLAG_FEATURES = [
    "c1_downtilt_too_large", 
    "c2_overshoot", 
    "c3_neighbor_stronger",
    "c4_noncolocated_neighbor", 
    "c5_frequent_handover", 
    "c6_pci_mod30_collision",
    "c7_speed_over_40", 
    "c8_rb_below_160", 
    "has_close_nbr",
]

NUMERIC_FEATURES = [
    # Symptom (not cause-specific): overall degradation shape
    "frac_degraded", "mean_throughput_mbps", "min_throughput_mbps",

    # C1 — downtilt too large / weak far-end coverage: how far past the
    # half-power beam edge the UE sits (graded form of c1_downtilt_too_large)
    "max_beam_ratio", "beam_ratio_mean", "beam_ratio_median", "beam_ratio_p90",
    "frac_beam_over_1", "frac_beam_over_1p3",

    # C2 — over-shoot: UE-to-serving-cell distance
    "max_dist_to_serving_km",

    # C3 — neighbor would serve better: serving-vs-strongest-neighbor gap and
    # how many neighbors are competitively close (graded form of has_close_nbr)
    "min_best_nbr_gap_db", "best_nbr_gap_mean", "best_nbr_gap_median",
    "frac_close_nbr", "mean_nbr_within_close",
    # serving RF stays healthy when degradation is competition- not coverage-limited
    "min_sinr_deg", "mean_sinr_deg", "mean_rsrp_deg", "min_rsrp_deg",

    # C4 — non-colocated overlapping neighbor: gap to strongest off-site neighbor
    # and crowding within the tighter overlap margin
    "min_noncolo_nbr_gap_db", "noncolo_nbr_gap_mean", "mean_nbr_within_overlap",

    # C5 — frequent handovers
    "n_handovers",

    # C6 — PCI mod 30 collision: fraction of degraded samples colliding
    "frac_mod30_collision",

    # C7 — vehicle speed over 40 km/h
    "mean_speed_kmh", "max_speed_kmh",

    # C8 — too few scheduled RBs
    "mean_rb_num", "min_rb_num",

    # General RF context (not tied to one cause)
    "mean_rsrp", "mean_sinr", "min_sinr",

    # Uknown if they are helpful
    "offaxis_deg_mean", "offaxis_deg_median", "offaxis_deg_max", "offaxis_deg_min", "frac_offaxis_over_60"

]
FEATURE_SETS = {
    "flags": FLAG_FEATURES,
    "numeric": NUMERIC_FEATURES,
    "both": FLAG_FEATURES + NUMERIC_FEATURES,
}

# Plot config
IMPORTANCE_TOP_N = 12
