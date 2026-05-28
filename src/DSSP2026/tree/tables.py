"""
tree/tables.py — presentation-quality tables for tree models.

Currently thin: the depth/metrics table styler lives in core/tables.py as
`style_tree_metrics_table` because it is reused across families. Add
tree-specific tables (feature importance, leaf summaries, etc.) here as they
come up.
"""

from DSSP2026.core.tables import style_tree_metrics_table  # re-export for convenience

__all__ = ["style_tree_metrics_table"]