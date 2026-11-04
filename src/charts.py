"""Figures for the report. Each function saves a PNG to figures/ and returns the figure."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from mplsoccer import VerticalPitch

FIG = Path(__file__).resolve().parents[1] / "figures"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REST, ELITE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
LABELS = {False: "Rest of La Liga", True: "Elite (Real Madrid / Atlético)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    return fig


def xg_split(df: pd.DataFrame):
    """Grouped bars: xG for / against per game, rest vs elite."""
    g = df.groupby("elite")[["xg_for", "xg_against"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    for i, elite in enumerate([False, True]):
        vals = g.loc[elite].values
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=[REST, ELITE][i], label=LABELS[elite])
        ax.bar_label(bars, fmt="%.2f", padding=3, color=INK)
    ax.set_xticks(x, ["xG created per game", "xG conceded per game"])
    ax.set_ylim(0, 2.4)
    ax.grid(axis="x", visible=False)
    ax.set_title("Barça create a third less against elite sides")
    ax.legend(loc="upper right")
    return _save(fig, "xg_split")


def season_trend(df: pd.DataFrame):
    """xG difference per game by season: rest vs elite."""
    s = df.groupby(["season", "elite"]).xg_diff.mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4))
    for elite, color in [(False, REST), (True, ELITE)]:
        ax.plot(s.index, s[elite], color=color, lw=2, marker="o", ms=6, label=LABELS[elite])
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_ylabel("xG difference per game")
    ax.set_xticks(range(len(s.index)), [x[2:4] + "/" + x[7:] for x in s.index], rotation=0, fontsize=9)
    above = int((s[True] < s[False]).sum())
    ax.set_title(f"Barça's margin is smaller against elite sides in {above} of {int(s[True].notna().sum())} seasons")
    ax.text(0, -0.16, "Elite line uses only 1–4 games per season, so single seasons are noisy.",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(loc="lower left")
    return _save(fig, "season_trend")


def style_dumbbell(df: pd.DataFrame):
    """How Barça's style metrics shift against elite opposition."""
    metrics = [("pass_share", "Pass share (%)", 100), ("field_tilt", "Field tilt (%)", 100),
               ("def_action_height", "Defensive action height (m)", 1), ("ppda", "PPDA (lower = more pressing)", 1),
               ("high_recoveries", "High recoveries per game", 1)]
    g = df.groupby("elite")[[m for m, _, _ in metrics]].mean()
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 5), sharex=False)
    for ax, (col, label, k) in zip(axes, metrics):
        a, b = g.loc[False, col] * k, g.loc[True, col] * k
        ax.plot([a, b], [0, 0], color=GRID, lw=3, zorder=1)
        ax.scatter([a], [0], s=70, color=REST, zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.scatter([b], [0], s=70, color=ELITE, zorder=2, edgecolor=SURFACE, linewidth=2)
        lo, hi = sorted([a, b])
        pad = max(hi - lo, hi * 0.04)
        ax.set_xlim(lo - pad * 1.5, hi + pad * 1.5)
        ax.set_yticks([])
        ax.set_xticks([])       # each row has its own scale, so axis ticks would invite false comparisons
        ax.grid(False)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.text(0, 0.62, label, transform=ax.transAxes, fontsize=10, color=INK_2)
        ax.annotate(f"{a:.1f}", (a, 0), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{b:.1f}", (b, 0), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9)
    fig.suptitle("Press stays the same; territory doesn't", x=0.03, y=1.04, ha="left", fontweight="bold", fontsize=13)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=c, label=LABELS[k])
               for k, c in [(False, REST), (True, ELITE)]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.0), ncol=2, fontsize=9.5)
    fig.text(0.03, -0.02, "Barça averages per game. Each row has its own scale.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return _save(fig, "style_dumbbell")


def shot_maps(shots: pd.DataFrame, matches: pd.DataFrame):
    """Barça's non-penalty shots per game by pitch zone, rest vs elite, on one shared colour scale."""
    s = shots[(shots.team == "Barcelona") & (shots.shot_type != "Penalty")].merge(matches[["match_id", "elite"]])
    cmap = LinearSegmentedColormap.from_list("blues", ["#f1f6fd", "#9cc2ef", "#2a78d6", "#123f78"])
    pitch = VerticalPitch(half=True, pitch_color=SURFACE, line_color="#8a8984", linewidth=1, line_zorder=2)
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(10, 5.6))
    stats = {}
    for elite in (False, True):
        d = s[s.elite == elite]
        st = pitch.bin_statistic(d.x, d.y, statistic="count", bins=(6, 5))
        st["statistic"] = st["statistic"] / d.match_id.nunique()        # shots per game in each zone
        stats[elite] = (d, st)
    vmax = max(st["statistic"].max() for _, st in stats.values())
    for ax, elite in zip(axes, (False, True)):
        d, st = stats[elite]
        n = d.match_id.nunique()
        in_box = ((d.x >= 102) & d.y.between(18, 62)).mean()
        pitch.heatmap(st, ax=ax, cmap=cmap, vmin=0, vmax=vmax, edgecolor=SURFACE, linewidth=2)
        st["statistic"] = np.where(st["statistic"] >= 0.05, st["statistic"], np.nan)
        pitch.label_heatmap(st, ax=ax, str_format="{:.1f}", fontsize=10, ha="center", va="center",
                            color=INK, exclude_nan=True,
                            path_effects=[patheffects.withStroke(linewidth=2.5, foreground=SURFACE)])
        ax.set_title(f"{LABELS[elite]}  ({n} games)\n{len(d) / n:.1f} shots/game · {in_box:.0%} inside the box · "
                     f"{d.xg_statsbomb.mean():.3f} xG per shot", fontsize=10.5, loc="center")
    fig.suptitle("Barça non-penalty shots per game, by zone: fewer, and fewer in the box, against elite sides",
                 x=0.06, ha="left", fontweight="bold", fontsize=13)
    return _save(fig, "shot_maps")


def calibration(shots_xg: pd.DataFrame):
    """Reliability curve: predicted xG vs observed goal rate, by decile of prediction."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 0.5], [0, 0.5], color=INK_2, lw=1, ls="--", label="Perfect calibration")
    for col, color, label in [("xg_statsbomb", INK_2, "StatsBomb xG"), ("xg_logistic", REST, "My logistic model"),
                              ("xg_gradient_boosting", ELITE, "My gradient boosting")]:
        bins = pd.qcut(shots_xg[col], 10, duplicates="drop")
        g = shots_xg.groupby(bins, observed=True).agg(pred=(col, "mean"), obs=("goal", "mean"))
        ax.plot(g.pred, g.obs, marker="o", ms=6, lw=2, color=color, label=label,
                alpha=0.8 if color == INK_2 else 1)
    ax.set_xlabel("Predicted probability (decile mean)")
    ax.set_ylabel("Actual goal rate")
    ax.set_title("xG model calibration")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, "xg_calibration")
