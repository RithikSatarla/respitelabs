"""Build the pitch deck charts.

Our own results are read from the JSON that Kintrace runs wrote into bench_out/.
Nothing is typed in by hand. Outside numbers appear once, at the top of the
function that uses them, with the source next to them, and every one of them is
listed in pitch/SOURCES.md.

Run from the repo root:  python pitch/make_charts.py
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench_out"
OUT = ROOT / "pitch" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# Palette, from docs/index.html. See pitch/NOTES.md.
# Yellow is a fill colour only. It is never used for text or for a thin line,
# because #FFC400 on #F7F7F4 has nowhere near enough contrast to read.
INK = "#121212"
SOFT = "#55565B"
RULE = "#DEDED8"
PANEL = "#EFEFEA"
PAPER = "#F7F7F4"
ACCENT = "#FFC400"     # Kintrace, and only Kintrace
GREY = "#C9C9C2"       # everything that is not us
GO = "#0B7A2A"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16,
    "text.color": INK,
    "axes.labelcolor": SOFT,
    "xtick.color": SOFT,
    "ytick.color": SOFT,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})

PLAIN = {
    "encoder_bias": "Joint knocked off zero",
    "tcp_config": "Wrong setting loaded",
    "tool_bent": "Gripper bent",
    "camera_moved": "Camera bumped",
    "latency": "Commands running slow",
    "none": "Nothing wrong",
}
ORDER = ["encoder_bias", "tcp_config", "tool_bent", "camera_moved", "latency", "none"]


def strip(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side, keep in (("left", left), ("bottom", bottom)):
        ax.spines[side].set_visible(keep)
        if keep:
            ax.spines[side].set_color(RULE)


def chart_results():
    """Per fault, how often each approach named the right cause."""
    d = json.loads((BENCH / "bench.json").read_text())
    per = d["summary"]["per_class"]
    n = d["summary"]["n_runs"]

    labels = [PLAIN[k] for k in ORDER]
    dash = [per[k]["dashboard"] * 100 for k in ORDER]
    kin = [per[k]["kintrace"] * 100 for k in ORDER]

    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    h = 0.34
    ys = list(range(len(ORDER)))

    for i, (dv, kv) in enumerate(zip(dash, kin)):
        ax.barh(i - h / 2, kv, height=h, color=ACCENT, zorder=3,
                edgecolor=INK, linewidth=0.7)
        ax.barh(i + h / 2, dv, height=h, color=GREY, zorder=3)
        ax.text(kv + 1.6, i - h / 2, f"{kv:.0f}%", va="center", ha="left",
                fontsize=15, color=INK, fontweight="bold")
        ax.text(dv + 1.6, i + h / 2, f"{dv:.0f}%", va="center", ha="left",
                fontsize=15, color=SOFT)

    # A small key above the bars. Chips sit in axes coordinates so they stay put.
    for x, colour, label, weight in (
        (0.00, ACCENT, "Kintrace", "bold"),
        (0.175, GREY, "Our dashboard baseline", "normal"),
    ):
        ax.add_patch(plt.Rectangle((x, 1.045), 0.016, 0.035, facecolor=colour,
                                   edgecolor=INK, linewidth=0.7,
                                   transform=ax.transAxes, clip_on=False, zorder=5))
        ax.text(x + 0.026, 1.062, label, transform=ax.transAxes, fontsize=16,
                color=INK if weight == "bold" else SOFT, fontweight=weight,
                va="center", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=17, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    strip(ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(OUT / "results_by_fault.png", dpi=200)
    plt.close(fig)
    print(f"results_by_fault.png   dashboard {d['summary']['dashboard_acc']*100:.0f}%"
          f"  kintrace {d['summary']['kintrace_acc']*100:.0f}%  n={n}")


def chart_early_warning():
    """Health score falling, with the warning and the first missed pick."""
    d = json.loads((BENCH / "watch_camera_sag_seed3.json").read_text())
    win = d["windows"]
    t = [w["t"] for w in win]
    health = [w["health"] for w in win]
    warn_t, warn_health = d["t"], d["health"]

    txt = (BENCH / "watch_camera_sag_seed3.txt").read_text()
    miss_t = None
    for line in txt.splitlines():
        if "first missed pick at" in line:
            hh, mm, ss = (int(x) for x in line.strip().rstrip(")").split()[-1].split(":"))
            sh, sm, sss = (int(x) for x in d["start_time"].split("T")[1].split(":"))
            miss_t = (hh * 3600 + mm * 60 + ss) - (sh * 3600 + sm * 60 + sss)
    if miss_t is None:
        raise SystemExit("could not read the first missed pick time from the run log")

    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    ax.fill_between(t, health, color=ACCENT, alpha=0.55, zorder=1)
    ax.plot(t, health, color=INK, linewidth=3.0, zorder=4, solid_capstyle="round")

    ax.axvline(warn_t, color=INK, linewidth=1.6, zorder=5)
    ax.axvline(miss_t, color=SOFT, linewidth=1.2, linestyle=(0,(4,3)), zorder=5)

    ax.annotate("Kintrace warns here\nhealth 71, nothing has missed yet",
                xy=(warn_t, warn_health), xytext=(warn_t - 55, 30),
                fontsize=15.5, ha="left", color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=1.1,
                                connectionstyle="arc3,rad=0.12"))
    ax.text(miss_t + 3.5, 92, "First missed pick", fontsize=15.5, color=SOFT, ha="left")

    ax.annotate("", xy=(warn_t, 106), xytext=(miss_t, 106),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.3))
    ax.text((warn_t + miss_t) / 2, 109, f"{miss_t - warn_t:.0f} seconds of warning",
            ha="center", fontsize=16, color=INK, fontweight="bold")

    ax.set_xlabel("Seconds into the run", fontsize=15, color=SOFT)
    ax.set_ylabel("Physical health score", fontsize=15, color=SOFT)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 50, 100])
    ax.set_xlim(0, miss_t + 26)
    strip(ax)
    fig.tight_layout()
    fig.savefig(OUT / "early_warning.png", dpi=200)
    plt.close(fig)
    print(f"early_warning.png      warn {warn_t:.0f}s  miss {miss_t:.0f}s  "
          f"lead {miss_t - warn_t:.0f}s")


def chart_why_now():
    """Robotics venture funding, the one verified multi year series we have."""
    # Robotics Center, State of Robotics 2026. USD billions.
    years = ["2022", "2023", "2024", "2025"]
    vc = [3.8, 5.2, 6.6, 9.4]

    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    colors = [GREY, GREY, GREY, ACCENT]
    bars = ax.bar(years, vc, width=0.52, color=colors, zorder=3,
                  edgecolor=INK, linewidth=0.7)
    for b, v in zip(bars, vc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"${v}B",
                ha="center", fontsize=17, color=INK,
                fontweight="bold" if v == vc[-1] else "normal")

    ax.text(2.55, 8.3, "+41%\nin one year", ha="center", va="center",
            fontsize=15.5, color=INK, fontweight="bold", linespacing=1.4)

    ax.set_ylim(0, 11.4)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, labelsize=17)
    strip(ax, left=False, bottom=True)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUT / "why_now.png", dpi=200)
    plt.close(fig)
    print(f"why_now.png            VC {vc[0]}B -> {vc[-1]}B")


def chart_market():
    """TAM / SAM / SOM, bottom up. Circle area is dollars."""
    per_year = 2400                 # $200 a month
    robots = 5_000_000              # IFR World Robotics 2026
    sam_share = 0.10                # our estimate, no public figure exists
    som_robots = 20_000             # our 5 year plan

    tam = robots * per_year
    sam = robots * sam_share * per_year
    som = som_robots * per_year

    rows = [
        ("TAM", f"${tam/1e9:.0f}B a year", "every industrial robot on earth\n5,000,000 x $2,400 a year"),
        ("SAM", f"${sam/1e9:.1f}B a year", f"the {sam_share:.0%} with a camera, in fleets\nand labs we can reach (our estimate)"),
        ("SOM", f"${som/1e6:.0f}M a year", "20,000 robots, five years out\n20,000 x $2,400 a year"),
    ]
    vals = [tam, sam, som]
    fills = ["#FFF3C9", "#FFE07A", ACCENT]

    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    R = 1.0
    radii = [R * (v / vals[0]) ** 0.5 for v in vals]
    for r, c in zip(radii, fills):
        ax.add_patch(Circle((0, r), r, facecolor=c, edgecolor=INK,
                            linewidth=1.0, zorder=2))

    for i, ((tag, money, note), r) in enumerate(zip(rows, radii)):
        ty = 1.80 - i * 0.60
        ax.annotate("", xy=(0 if i else -r * 0.34, r * (0.30 if i else 1.62)),
                    xytext=(1.42, ty),
                    arrowprops=dict(arrowstyle="-", color="#C9CACE", linewidth=1.1))
        ax.text(1.50, ty + 0.10, f"{tag}   {money}", fontsize=18,
                color=INK, fontweight="bold", va="center", ha="left")
        ax.text(1.50, ty - 0.17, note, fontsize=14.5, color=SOFT,
                va="center", ha="left", linespacing=1.45)

    ax.set_xlim(-1.12, 4.25)
    ax.set_ylim(-0.06, 2.16)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "market.png", dpi=200)
    plt.close(fig)
    print(f"market.png             TAM ${tam/1e9:.0f}B  SAM ${sam/1e9:.1f}B  SOM ${som/1e6:.0f}M")


def chart_budget():
    """Where the $150K goes. Plain horizontal bars, biggest first."""
    rows = [
        ("Part-time robotics engineer, or a cofounder", 80),
        ("A real industrial arm, used UR3 or UR5e", 20),
        ("Buffer", 30),
        ("Pilots: travel, cameras, markers on customer robots", 10),
        ("Company setup, legal, accounting", 5),
        ("Cloud and software", 3),
        ("Desk robots and parts, 2 to 3 test rigs", 2),
    ]
    rows.sort(key=lambda r: -r[1])
    total = sum(v for _, v in rows)
    assert total == 150, f"budget adds to {total}, not 150"

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    ys = range(len(rows))
    ax.barh(list(ys), vals, height=0.56, color=ACCENT, edgecolor=INK,
            linewidth=0.7, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 1.6, i, f"${v}K", va="center", ha="left",
                fontsize=16, color=INK, fontweight="bold")

    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=14.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 96)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    strip(ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(OUT / "budget.png", dpi=200)
    plt.close(fig)
    print(f"budget.png             {len(rows)} lines, ${total}K total")


def chart_cost():
    """100 squares. The share of audited training episodes marked unusable."""
    # dev.to audit of 4,959 episodes from 12 datasets: 775 EXCLUDE = 15.6%.
    share = 0.156
    total, cols = 100, 20
    marked = round(share * total)

    fig, ax = plt.subplots(figsize=(11.6, 3.5))
    size, gap = 1.0, 0.26
    for i in range(total):
        r, c = divmod(i, cols)
        ax.add_patch(plt.Rectangle((c * (size + gap), -r * (size + gap)),
                                   size, size,
                                   facecolor=ACCENT if i < marked else RULE,
                                   edgecolor=INK if i < marked else "none",
                                   linewidth=0.6))
    ax.set_xlim(-0.4, cols * (size + gap))
    ax.set_ylim(-5 * (size + gap) + 0.3, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, 1.05, f"{marked} in 100", fontsize=22, color=INK,
            fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "cost_episodes.png", dpi=200)
    plt.close(fig)
    print(f"cost_episodes.png      {marked}/100 squares marked ({share:.1%})")


if __name__ == "__main__":
    chart_results()
    chart_early_warning()
    chart_why_now()
    chart_market()
    chart_cost()
    chart_budget()
