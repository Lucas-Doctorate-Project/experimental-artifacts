"""Shared typography and semantic colors for analysis figures."""

from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.text import Text

FONT_DIR = Path(__file__).resolve().parents[1] / "fonts"
for weight in ("Regular", "Bold"):
    font_manager.fontManager.addfont(str(FONT_DIR / f"RobotoCondensed-{weight}.ttf"))

POLICY_COLORS = {
    "Greenfilling": "#737373",
    "CATS": "#00765e",
    "Head": "#2677ad",
    "Offline": "#b85d18",
    "Bound": "#a17940",
}
CONDITION_COLORS = {
    "carbon slack": "#b49bc8",
    "carbon stress": "#6f428c",
    "water slack": "#df9aa9",
    "water stress": "#a83e59",
}

RESULT_STYLE = {
    "font.family": "Roboto Condensed",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": ".7",
    "axes.grid": True,
    "axes.axisbelow": True,
    "text.color": "#292929",
    "axes.labelcolor": "#292929",
    "xtick.color": "#505050",
    "ytick.color": "#292929",
    "grid.alpha": 0.15,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}


def pale(color, amount=0.28):
    """Mix with white without transparency so guides cannot show through boxes."""
    return tuple(amount * np.array(to_rgb(color)) + (1 - amount))


def typography(fig):
    for text in fig.findobj(Text):
        text.set_fontfamily("Roboto Condensed")
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(labelsize=11)
    if fig._suptitle is not None:
        fig._suptitle.set(fontsize=20, fontweight="bold", x=0.025, ha="left")
