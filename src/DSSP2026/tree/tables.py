from DSSP2026.core.style import att_table_styles
from DSSP2026.core.tables import _CAPTION_STYLE


def style_tree_metrics_table(metrics_df, *, context: str = "report"):
    fmt = {"MSPE": "{:,.3f}", "R2": "{:,.3f}", "R²": "{:,.3f}"}
    return (
        metrics_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(fmt, na_rep="")
            .set_caption("Tree model performance")
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


__all__ = ["style_tree_metrics_table"]
