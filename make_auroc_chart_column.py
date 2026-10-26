# make_auroc_chart_column.py: one-column version (panels stacked). AUROC with 95% bootstrap CIs per evidence condition (numbers from results_eval.txt)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDS = ["Gold evidence", "BM25", "Similarity", "Verifier-guided"]
FULL = [
    (0.715, 0.589, 0.827),
    (0.594, 0.456, 0.730),
    (0.587, 0.473, 0.699),
    (0.599, 0.483, 0.719),
]
STMT = [
    (0.679, 0.562, 0.795),
    (0.595, 0.471, 0.718),
    (0.565, 0.475, 0.652),
    (0.609, 0.511, 0.711),
]
GOLD, RETR, INK, MUTED, GRID = "#2a78d6", "#6f6e69", "#0b0b0b", "#52514e", "#d9d8d3"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.2), sharex=True)
for ax, data, title in zip(
    axes,
    [FULL, STMT],
    ["Full score (statements + claim check)", "Statements only (no claim check)"],
):
    for i, (m, lo, hi) in enumerate(data):
        y = len(CONDS) - 1 - i
        c = GOLD if i == 0 else RETR
        ax.plot([lo, hi], [y, y], color=c, lw=2.5, solid_capstyle="round", zorder=2)
        ax.plot(m, y, "o", ms=11, color=c, mec="white", mew=2, zorder=3)
        ax.text(
            m,
            y + 0.24,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            color=INK,
            fontweight="bold" if i == 0 else "normal",
        )
    ax.axvline(0.5, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(0.505, -0.75, "chance (0.5)", color=MUTED, fontsize=13, va="bottom")
    ax.set_xlim(0.40, 0.88)
    ax.set_ylim(-0.8, len(CONDS) - 0.35)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED)
    pass
    ax.set_title(title, loc="left", fontsize=16, color=INK, pad=10)
for ax in axes:
    ax.set_yticks(range(len(CONDS)))
    ax.set_yticklabels(CONDS[::-1])
    for t in ax.get_yticklabels():
        t.set_color(GOLD if t.get_text() == "Gold evidence" else INK)
        t.set_fontweight("bold" if t.get_text() == "Gold evidence" else "normal")
axes[1].set_xlabel(
    "AUROC (dot) with 95% bootstrap CI over claims (line)", color=MUTED, fontsize=13
)
fig.text(
    0.01,
    0.005,
    "300 LLM outputs, 32 wrong. Gold evidence is the only\ncondition clear of chance on both scores.",
    color=MUTED,
    fontsize=13,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("png", "svg", "pdf"):
    fig.savefig(f"auroc_chart_column.{ext}", dpi=300, facecolor="white")
print("ok")
