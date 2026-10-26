# make_pipeline_column_svg.py: top-to-bottom pipeline diagram for a one-column poster layout
INK, MUTED, LINE, GOLD, GOLDBG, GREY, GREYBG = (
    "#0b0b0b",
    "#52514e",
    "#9a9993",
    "#2a78d6",
    "#e7f1fb",
    "#6f6e69",
    "#f0efec",
)
F = "DejaVu Sans, Arial, Helvetica, sans-serif"
W = 900
BX, BW = 120, 660
out = []
a = out.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def text(x, y, s, size=23, color=INK, weight="normal", anchor="middle", rot=None):
    t = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    a(
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{t}>{esc(s)}</text>'
    )


boxes = [
    (
        "Claim",
        ["100 AVeriTeC claims (50 Supported, 50 Refuted)", "claim, date and speaker"],
    ),
    (
        "LLM",
        [
            "closed-book, temperature 0: Gemini, Claude, GPT",
            "reasoning + label · 300 outputs, 32 wrong",
        ],
    ),
    (
        "Decompose",
        [
            "GPT-6.1 splits the reasoning into facts",
            "1,612 facts + 100 claims to check",
        ],
    ),
    ("Evidence", None),
    (
        "Verify",
        [
            "DeBERTa-v3 NLI, same in every condition",
            "entailment / neutral / contradiction",
        ],
    ),
    (
        "Flag",
        [
            "flag if any fact is refuted, or if the claim",
            "verdict opposes the LLM label",
        ],
    ),
    (
        "Evaluate",
        ["flags vs. wrong outputs", "AUROC with 95% bootstrap CI over claims"],
    ),
]
y, GAP, BH, EH = 20, 52, 128, 316
pos = []
body = []
for i, (title, lines) in enumerate(boxes):
    h = EH if lines is None else BH
    pos.append((y, h))
    y += h + GAP
H = y - GAP + 20
a(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{F}">'
)
a(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
a(
    "<defs>"
    + "".join(
        f'<marker id="{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for n, c in (("ar", LINE), ("ard", MUTED), ("arg", GOLD))
    )
    + "</defs>"
)
for i, (title, lines) in enumerate(boxes):
    by, h = pos[i]
    a(
        f'<rect x="{BX}" y="{by}" width="{BW}" height="{h}" rx="16" fill="#ffffff" stroke="{LINE}" stroke-width="2.5"/>'
    )
    a(f'<circle cx="{BX+36}" cy="{by+38}" r="19" fill="{INK}"/>')
    text(BX + 36, by + 46, str(i + 1), 21, "#ffffff", "bold")
    text(BX + 68, by + 48, title, 30, INK, "bold", "start")
    if lines:
        for k, ln in enumerate(lines):
            text(BX + BW / 2, by + 88 + k * 30, ln, 23, MUTED)
    else:
        text(
            BX + BW - 24, by + 46, "per fact and per claim", 21, MUTED, "normal", "end"
        )
        opts = [
            ("Gold", "annotators' QA pairs", GOLD, GOLDBG),
            ("BM25", "top 5, word match", GREY, GREYBG),
            ("Similarity", "top 5, embeddings", GREY, GREYBG),
            ("Verifier-guided", "NLI picks top 5 of 20", GREY, GREYBG),
        ]
        cw, chh = (BW - 60) / 2, 84
        for k, (nm, sub, col, bg) in enumerate(opts):
            cx = BX + 20 + (k % 2) * (cw + 20)
            cy = by + 76 + (k // 2) * (chh + 16)
            a(
                f'<rect x="{cx}" y="{cy}" width="{cw}" height="{chh}" rx="10" fill="{bg}" stroke="{col}" stroke-width="2.5"/>'
            )
            text(cx + cw / 2, cy + 36, nm, 24, col if k == 0 else INK, "bold")
            text(cx + cw / 2, cy + 66, sub, 19, MUTED)
        text(
            BX + BW / 2, by + h - 22, "retrieval pool: 10,498 web sentences", 20, MUTED
        )
    if i < len(boxes) - 1:
        a(
            f'<path d="M{BX+BW/2} {by+h+4} V{by+h+GAP-6}" stroke="{LINE}" stroke-width="4" fill="none" marker-end="url(#ar)"/>'
        )

mid = lambda i: pos[i][0] + pos[i][1] / 2
# LLM label: right side, to Flag and Evaluate
rx = BX + BW + 52
a(
    f'<path d="M{BX+BW} {mid(1)} H{rx} V{mid(6)} H{BX+BW+8}" stroke="{MUTED}" stroke-width="2.5" stroke-dasharray="9 7" fill="none" marker-end="url(#ard)"/>'
)
a(
    f'<path d="M{rx} {mid(5)} H{BX+BW+8}" stroke="{MUTED}" stroke-width="2.5" stroke-dasharray="9 7" fill="none" marker-end="url(#ard)"/>'
)
text(
    rx + 30,
    (mid(1) + mid(5)) / 2,
    "LLM label (saved in step 2)",
    21,
    MUTED,
    "normal",
    "middle",
    90,
)
# gold label: left side, Claim to Evaluate
lx = BX - 52
a(
    f'<path d="M{BX} {mid(0)} H{lx} V{mid(6)} H{BX-8}" stroke="{GOLD}" stroke-width="2.5" stroke-dasharray="9 7" fill="none" marker-end="url(#arg)"/>'
)
text(
    lx - 16,
    (mid(0) + mid(6)) / 2,
    "gold label: used only in step 7",
    21,
    GOLD,
    "normal",
    "middle",
    -90,
)
a("</svg>")
open("pipeline_column.svg", "w").write("\n".join(out))
print(W, H)
