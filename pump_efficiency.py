"""Pump efficiency vs. flow rate at 45 Hz. Requires matplotlib.

Usage: python pump_efficiency.py [pump_chart_60Hz.csv]
"""
import csv
import logging
import math
import sys
from pathlib import Path

import matplotlib
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RHO = 998.0           # kg m^-3, water at 70 °F
MU = 9.78e-4          # Pa s
G = 9.81
IN = 0.0254
FT = 0.3048
GPM = 3.785411784
HP = 745.7
F_RATIO = 45 / 60
PO_BRAKE_60_HP = 0.75
PO_BRAKE_HP = 0.32
ETA_MOTOR = 0.72
Q_TABLE = [0, 2, 4, 6, 8, 10, 12]

PUMP_CHART_60HZ = [(10, 72), (15, 68), (20, 63), (30, 53), (40, 41), (50, 22), (55, 0)]  # head / ft, flow / GPM

PIPES = [  # name, ID / in, length / in, roughness / in
    ("Copper, small", 0.310, 95.31, 6e-6),
    ("Copper, medium", 0.785, 53.38, 6e-6),
    ("Copper, large", 1.025, 44.25, 6e-6),
    ("Flex tubing", 0.750, 35.47, 4e-6),
]

FITTINGS = [  # name, ID / in, K, n
    ("Ball valve", 1.025, 0.07, 2),
    ("Globe valve", 0.785, 8.5, 2),
    ("3-way valve", 0.785, 2.25, 1),
    ("Bend, 90°", 0.310, 0.81, 2),
    ("Bend, 90°", 0.785, 0.75, 2),
    ("Bend, 90°", 1.025, 0.69, 1),
    ("Tee, 90°", 0.785, 1.50, 1),
    ("Tee, 90°", 1.025, 1.38, 1),
    ("Tee branch, 45°", 0.785, 0.40, 2),
]

_B2 = (0.310 / 0.750) ** 2
TRANSITIONS = [  # name, ID / in, K
    ('0.750" → 0.310"', 0.310, 0.5 * (1 - _B2)),
    ('0.310" → 0.750"', 0.310, (1 - _B2) ** 2),
    ('0.750" → Tank', 0.750, 1.0),
    ('Tank → 1.025"', 1.025, 0.5),
]


def velocity(q, d_in):
    return 4 * q / (math.pi * (d_in * IN) ** 2)


def velocity_head(q, d_in):
    return RHO * velocity(q, d_in) ** 2 / 2


def reynolds(q, d_in):
    return 4 * RHO * q / (math.pi * MU * d_in * IN)


def haaland_fanning(re, rel_rough):
    return (-3.6 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / re)) ** -2


def system_losses(q_lpm):
    q = q_lpm / 60000
    if q == 0:
        return {"Q_m3_s": 0.0, "dP_pipe_Pa": 0.0, "dP_fittings_Pa": 0.0, "dP_exp_cont_Pa": 0.0, "dP_total_Pa": 0.0}
    pipe = sum(4 * haaland_fanning(reynolds(q, d), e / d) * (L / d) * velocity_head(q, d) for _, d, L, e in PIPES)
    fitt = sum(n * k * velocity_head(q, d) for _, d, k, n in FITTINGS)
    trans = sum(k * velocity_head(q, d) for _, d, k in TRANSITIONS)
    return {"Q_m3_s": q, "dP_pipe_Pa": pipe, "dP_fittings_Pa": fitt, "dP_exp_cont_Pa": trans,
            "dP_total_Pa": pipe + fitt + trans}


def load_pump_chart(path=None):
    """60 Hz chart scaled to 45 Hz: [(Q / LPM, dP / Pa, efficiency / %)]."""
    if path is None:
        raw = PUMP_CHART_60HZ
    else:
        with open(path, newline="") as fh:
            raw = [(float(r["head_ft"]), float(r["flow_GPM"])) for r in csv.DictReader(fh)]
    pts = []
    for h60, gpm in raw:
        q60, dp60 = gpm * GPM, RHO * G * h60 * FT
        eta = 100 * dp60 * q60 / 60000 / (PO_BRAKE_60_HP * HP)
        pts.append((q60 * F_RATIO, dp60 * F_RATIO ** 2, eta))
    return sorted(pts)


def interp(pts, q, col):
    for a, b in zip(pts, pts[1:]):
        if a[0] <= q <= b[0]:
            return a[col] + (b[col] - a[col]) * (q - a[0]) / (b[0] - a[0])
    raise ValueError(f"{q} L/min is outside the pump chart")


def bisect(f, lo, hi, n=80):
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return mid


def efficiency(q_lpm, dp):
    return 100 * (q_lpm / 60000) * dp / (PO_BRAKE_HP * HP)


def pick_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Arial", "Liberation Sans", "Helvetica") if f in installed), "DejaVu Sans")


def main(chart_path=None):
    out_dir = Path(__file__).resolve().parent
    chart = load_pump_chart(chart_path)
    q_op = bisect(lambda q: system_losses(q)["dP_total_Pa"] - interp(chart, q, 1), 1e-6, chart[-1][0])
    dp_op = interp(chart, q_op, 1)

    rows = []
    for q in Q_TABLE:
        s = system_losses(q)
        rows.append({
            "flow_rate_LPM": round(q, 3),
            "dP_pipe_tube_Pa": s["dP_pipe_Pa"], "dP_fitt_bend_Pa": s["dP_fittings_Pa"],
            "dP_exp_cont_Pa": s["dP_exp_cont_Pa"], "dP_total_Pa": s["dP_total_Pa"],
            "Po_hyd_W": s["Q_m3_s"] * s["dP_total_Pa"],
            "eta_pump_calc_pct": efficiency(q, s["dP_total_Pa"]),
            "dP_pump_chart_Pa": interp(chart, q, 1),
            "eta_pump_chart_pct": interp(chart, q, 2),
        })
    out_csv = out_dir / "system_curve_results.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    font = pick_font()
    plt.rcParams.update({
        "font.family": font, "font.size": 10,
        "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": f"{font}:italic",
        "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.labelcolor": "black",
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.color": "black", "ytick.color": "black", "legend.frameon": False,
    })
    eta_calc = lambda q: efficiency(q, system_losses(q)["dP_total_Pa"])
    q_max = max(q for q in Q_TABLE if q <= q_op)
    qs = [i * q_max / 240 for i in range(241)]
    pts = [q for q in Q_TABLE if q <= q_max]
    q_x = bisect(lambda q: eta_calc(q) - interp(chart, q, 2), 0.5, q_max)
    eta_x = eta_calc(q_x)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(qs, [interp(chart, q, 2) for q in qs], color="black", lw=1.2, ls=(0, (5, 3)), label="Pump chart, 45 Hz")
    ax.plot(qs, [eta_calc(q) for q in qs], color="black", lw=1.2,
            label=r"Calculated, $Q\,\Delta P_{\mathrm{total}}/\mathrm{Po}_{\mathrm{brake}}$")
    ax.plot(pts, [eta_calc(q) for q in pts], "o", ms=5, mfc="white", mec="black", mew=1.0, zorder=3)
    ax.plot([q_x], [eta_x], "o", ms=6, mfc="black", mec="black", zorder=4)
    ax.annotate(f"{q_x:.1f} L min$^{{-1}}$, {eta_x:.1f} %", (q_x, eta_x), textcoords="offset points",
                xytext=(-10, 6), ha="right", va="bottom", fontsize=9)
    ax.set_xlim(0, q_max + 0.5)
    ax.set_ylim(0, 8)
    ax.set_xlabel("Flow rate / L min$^{-1}$")
    ax.set_ylabel("Pump efficiency / %")
    ax.legend(loc="upper left", fontsize=9, handlelength=2.6)
    out_png = out_dir / "pump_curve.png"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")

    print(f"Electrical power: {PO_BRAKE_HP * HP / ETA_MOTOR:.1f} W")
    print(f"Operating point: {q_op:.2f} L/min, {dp_op:.0f} Pa")
    print(f"Efficiency curves intersect at {q_x:.2f} L/min, {eta_x:.2f} %")
    print()
    print(f'{"Q":>7} {"dP_pipe":>9} {"dP_fitt":>9} {"dP_exp":>9} {"dP_total":>9} {"Po_hyd":>7} {"eta_calc":>8} {"eta_chart":>9}')
    print(f'{"L/min":>7} {"Pa":>9} {"Pa":>9} {"Pa":>9} {"Pa":>9} {"W":>7} {"%":>8} {"%":>9}')
    for r in rows:
        print(f'{r["flow_rate_LPM"]:7.3f} {r["dP_pipe_tube_Pa"]:9.0f} {r["dP_fitt_bend_Pa"]:9.0f} '
              f'{r["dP_exp_cont_Pa"]:9.0f} {r["dP_total_Pa"]:9.0f} {r["Po_hyd_W"]:7.2f} '
              f'{r["eta_pump_calc_pct"]:8.2f} {r["eta_pump_chart_pct"]:9.2f}')
    print()
    print(f"Wrote {out_csv}\n      {out_png}")


if __name__ == "__main__":
    main(*sys.argv[1:])
