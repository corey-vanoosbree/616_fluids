import math
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from PIL import Image

import create_plots as cp

OUT = Path("errorbar_figures")
PSI = cp.PSI_TO_PA
INCH = 0.0254
FILES = [("FLU-prelab-Slowflow", "TA", "A"), ("FLU_Prelab_Highflow", "TB", "B"),
         ("FLU-lab-Slowflow", "A", "A"), ("FLU_Lab_Highflow", "B", "B")]
DP_FULL_SCALE_PSID = {"A": 1.0, "B": 15.0}
U_D = math.hypot(0.03e-3, 0.01e-3)
U_L = math.hypot(0.5 / 16 * INCH, 1 / 16 * INCH)
U_T = math.hypot(0.03, 0.2)
MEAN_EPS_OVER_D = statistics.mean(s["roughness_in"] / s["diameter_in"] for s in cp.SPECS_BY_DIAMETER.values())
ERRORBAR_STYLE = dict(color="black", markerfacecolor="black", ecolor="black", elinewidth=0.8, capsize=2.5,
                      capthick=0.8)


def read_runs():
    runs = []
    for fname, prefix, side in FILES:
        for line in open(fname, encoding="utf-8").read().splitlines()[4:]:
            clock, q, p, note = [x.strip() for x in line.split("\t")]
            if not runs or runs[-1]["file"] != fname or runs[-1]["note"] != note:
                number = sum(r["file"] == fname for r in runs) + 1
                runs.append(dict(id=f"{prefix}{number:02d}", file=fname, side=side, note=note, rows=[],
                                 start=datetime.strptime(clock, "%I:%M:%S %p")))
            runs[-1]["rows"].append((clock, float(q), float(p)))
    return runs


def describe(run):
    note = run["note"]
    temperature = cp.TEMP_PATTERN.search(note).group(1)
    run["T"] = float(temperature)
    run["T_cut_off"] = temperature.endswith(".")
    if "globe" in note:
        opening = cp.OPENING_LABELS[cp.OPENING_PATTERN.search(note).group(1).lower()]
        run.update(kind="valve", opening=opening, D_in=cp.VALVE_DIAMETER_IN, L_in=float("nan"))
    else:
        material = cp.MATERIAL_NAMES[cp.MATERIAL_PATTERN.match(note).group(1).lower()]
        spec = cp.PIPE_SPECS[(material, float(cp.DIAMETER_PATTERN.search(note).group(1)))]
        run.update(kind="pipe", material=material, D_in=spec["diameter_in"], L_in=spec["length_in"])


def interpolate_cut_off_temperatures(runs):
    tour = [r for r in runs if r["id"].startswith("T")]
    anchors = sorted((r["start"], r["T"]) for r in tour if not r["T_cut_off"])
    seconds = [(t - anchors[0][0]).total_seconds() for t, _ in anchors]
    temperatures = [T for _, T in anchors]
    for r in tour:
        if r["T_cut_off"]:
            r["T"] = float(np.interp((r["start"] - anchors[0][0]).total_seconds(), seconds, temperatures))
            print(f'{r["id"]} temperature "{r["note"]}" was cut off; interpolated to {r["T"]:.2f} C')


def clean(run):
    q_median = statistics.median(q for _, q, _ in run["rows"])
    run["kept"] = [(clock, q, p) for clock, q, p in run["rows"]
                   if (run["file"], run["note"], clock) not in cp.EXCLUDED_READINGS
                   and q > 0 and p > 0 and q >= 0.85 * q_median]
    mean_dp_psid = statistics.mean(p for _, _, p in run["kept"])
    run["dropped"] = None
    if run["id"] == "B16":
        run["dropped"] = "fully open 26.5 L/min run 1, run 2 (B17) used instead"
    elif run["kind"] == "pipe" and mean_dp_psid > 0.98 * DP_FULL_SCALE_PSID[run["side"]]:
        run["dropped"] = f"mean dP {mean_dp_psid:.4f} psid is past the 0-{DP_FULL_SCALE_PSID[run['side']]:g} psid range"


def u_a(values):
    return stats.t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))


def u_b_flow(side, q):
    return 0.03 * q if side == "A" else 0.02 * q


def u_b_dp(side):
    return 0.0008 * DP_FULL_SCALE_PSID[side] * PSI


def half_spread(fn, T):
    return abs(fn(T + U_T) - fn(T - U_T)) / 2


def propagate(fn, values, uncertainties):
    total = 0.0
    for name, u in uncertainties.items():
        h = 1e-6 * values[name]
        up = dict(values, **{name: values[name] + h})
        down = dict(values, **{name: values[name] - h})
        total += ((fn(**up) - fn(**down)) / (2 * h) * u) ** 2
    return fn(**values), math.sqrt(total)


def velocity(Q, D):
    return Q / 60000 / (math.pi * D**2 / 4)


def reynolds(Q, D, rho, mu, **_):
    return rho * velocity(Q, D) * D / mu


def friction_factor(Q, P, D, L, rho, **_):
    return P * D / (2 * L * rho * velocity(Q, D)**2)


def loss_coefficient(Q, P, D, rho, **_):
    return P / (0.5 * rho * velocity(Q, D)**2)


def equivalent_length(Q, P, D, rho, mu, **_):
    return loss_coefficient(Q, P, D, rho) / (4 * cp.churchill_fanning(reynolds(Q, D, rho, mu), MEAN_EPS_OVER_D))


def summarize(run, rho, T_rho):
    q = [q for _, q, _ in run["kept"]]
    p = [p * PSI for _, _, p in run["kept"]]
    run["Q"], run["P"] = statistics.mean(q), statistics.mean(p)
    run["U_Q"] = math.hypot(u_a(q), u_b_flow(run["side"], run["Q"]))
    run["U_P"] = math.hypot(u_a(p), u_b_dp(run["side"]))
    values = dict(Q=run["Q"], P=run["P"], D=run["D_in"] * INCH, L=run["L_in"] * INCH, rho=rho,
                  mu=cp.viscosity_kestin(run["T"]))
    uncertainties = dict(Q=run["U_Q"], P=run["U_P"], D=U_D, rho=half_spread(cp.density_kell, T_rho),
                         mu=half_spread(cp.viscosity_kestin, run["T"]))
    run["Re"], run["U_Re"] = propagate(reynolds, values, uncertainties)
    if run["kind"] == "pipe":
        run["f"], run["U_f"] = propagate(friction_factor, values, dict(uncertainties, L=U_L))
    else:
        run["K"], run["U_K"] = propagate(loss_coefficient, values, uncertainties)
        run["Leq"], run["U_Leq"] = propagate(equivalent_length, values, uncertainties)


def r_squared(measured, predicted):
    mean = statistics.mean(measured)
    ss_res = sum((m - p)**2 for m, p in zip(measured, predicted))
    ss_tot = sum((m - mean)**2 for m in measured)
    return 1 - ss_res / ss_tot


def save(fig, ax, filename, tight_bbox=False):
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight" if tight_bbox else None)
    plt.close(fig)


def pipes_in(points):
    return sorted({(p["material"], p["D_in"]) for p in points})


def plot_pipes_by_side(ax, points, x_key, y_key, label_suffix="", **marker_kwargs):
    """Plot each pipe's points split by side: open markers for low-flow (side
    "A") data, filled for high-flow (side "B") data. Adds one legend entry per
    pipe (keyed to its filled series) plus two generic entries explaining the
    open/filled convention."""
    handles = []
    for material, d in pipes_in(points):
        sub = [p for p in points if (p["material"], p["D_in"]) == (material, d)]
        marker = cp.PIPE_MARKERS[d]

        low = [p for p in sub if p["side"] == "A"]
        if low:
            # not appended to handles: unlabeled, and explicit `handles=` legend
            # calls (unlike the implicit ax.legend()) don't auto-skip those
            ax.errorbar([p[x_key] for p in low], [p[y_key] for p in low],
                        xerr=[p["U_" + x_key] for p in low], yerr=[p["U_" + y_key] for p in low],
                        fmt=marker, **{**ERRORBAR_STYLE, "markerfacecolor": "none"}, **marker_kwargs)

        high = [p for p in sub if p["side"] == "B"]
        if high:
            handles.append(ax.errorbar([p[x_key] for p in high], [p[y_key] for p in high],
                                        xerr=[p["U_" + x_key] for p in high], yerr=[p["U_" + y_key] for p in high],
                                        fmt=marker, label=f"{cp.pipe_label(material, d)}{label_suffix}",
                                        **ERRORBAR_STYLE, **marker_kwargs))

    handles.append(ax.errorbar([], [], fmt='o', label="high-flow data", **ERRORBAR_STYLE))
    handles.append(ax.errorbar([], [], fmt='o', label="low-flow data",
                                **{**ERRORBAR_STYLE, "markerfacecolor": "none"}))
    return handles


def flow_vs_pressure_plot(points, title, filename):
    """Plot pressure drop vs. flow rate across both the low- and high-flow-side
    datasets (side "A"/slowflow vs "B"/highflow, different transducers/ranges),
    but fit the power law (deltaP = a * Q^n) to each pipe's high-flow-side
    points only. High-flow points are filled; low-flow points are open."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fits = []
    for material, d in pipes_in(points):
        sub = sorted([p for p in points if (p["material"], p["D_in"]) == (material, d)], key=lambda p: p["Q"])
        marker = cp.PIPE_MARKERS[d]

        low = [p for p in sub if p["side"] == "A"]
        if low:
            ax.errorbar([p["Q"] for p in low], [p["P"] for p in low], xerr=[p["U_Q"] for p in low],
                        yerr=[p["U_P"] for p in low], fmt=marker,
                        **{**ERRORBAR_STYLE, "markerfacecolor": "none"})

        high = [p for p in sub if p["side"] == "B"]
        if high:
            ax.errorbar([p["Q"] for p in high], [p["P"] for p in high], xerr=[p["U_Q"] for p in high],
                        yerr=[p["U_P"] for p in high], fmt=marker, label=cp.pipe_label(material, d),
                        **ERRORBAR_STYLE)

        if not high:
            continue
        fit_q, fit_dp = [p["Q"] for p in high], [p["P"] for p in high]
        a, n = cp.power_law_fit(fit_q, fit_dp)
        q_curve = [min(fit_q) * (max(fit_q) / min(fit_q))**(i / 99) for i in range(100)]
        ax.plot(q_curve, [a * x**n for x in q_curve], "--", color="black")
        fits.append((material, d, n, r_squared(fit_dp, [a * x**n for x in fit_q])))
    ax.errorbar([], [], fmt='o', label="high-flow data (fit)", **ERRORBAR_STYLE)
    ax.errorbar([], [], fmt='o', label="low-flow data", **{**ERRORBAR_STYLE, "markerfacecolor": "none"})
    ax.text(0.97, 0.03, "\n".join(f"{cp.pipe_label(m, d)}: ΔP ~ Q^{n:.2f} (R² = {r2:.4f})"
                                   for m, d, n, r2 in fits),
            transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))
    ax.set(xscale="log", yscale="log", xlabel="Flow Rate (LPM)", ylabel="Pressure Drop (Pa)", title=title)
    ax.legend(loc="upper left")
    save(fig, ax, filename)
    print(f"Saved {OUT / filename} (" + ", ".join(f"{cp.pipe_label(m, d)}: n={n:.3f}, R^2={r2:.4f}"
                                                  for m, d, n, r2 in fits) + ")")


def friction_plot(points, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    re_min, re_max = min(p["Re"] for p in points), max(p["Re"] for p in points)
    ax.axvspan(re_min, cp.LAMINAR_RE, color="black", alpha=0.06)
    ax.axvspan(cp.LAMINAR_RE, cp.TURBULENT_RE, color="black", alpha=0.16)
    ax.axvline(cp.LAMINAR_RE, color="black", linestyle="--", linewidth=1)
    ax.axvline(cp.TURBULENT_RE, color="black", linestyle="--", linewidth=1)
    plot_pipes_by_side(ax, points, "Re", "f")
    ax.set(xscale="log", yscale="log", xlabel="Reynolds Number", ylabel="Fanning Friction Factor", title=title)
    for label, x in [("Laminar", (re_min * cp.LAMINAR_RE)**0.5),
                     ("Transitional", (cp.LAMINAR_RE * cp.TURBULENT_RE)**0.5),
                     ("Turbulent", (cp.TURBULENT_RE * re_max)**0.5)]:
        ax.text(x, 0.97, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
    ax.legend(loc="lower left", fontsize=9)
    save(fig, ax, filename)
    print(f"Saved {OUT / filename}")


def regime_comparison_plot(points, in_regime, correlation, correlation_label, title, filename):
    points = [p for p in points if in_regime(p["Re"])]
    fig, ax = plt.subplots(figsize=(8, 6))
    handles = plot_pipes_by_side(ax, points, "Re", "f", label_suffix=" (measured)")
    re_min, re_max = min(p["Re"] for p in points), max(p["Re"] for p in points)
    re_curve = [re_min * (re_max / re_min)**(i / 99) for i in range(100)]
    handles += ax.plot(re_curve, [correlation(x) for x in re_curve], "--", color="black", label=correlation_label)
    r2 = r_squared([p["f"] for p in points], [correlation(p["Re"]) for p in points])
    ax.text(0.03, 0.03, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=10, va="bottom", ha="left",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))
    ax.set(xscale="log", yscale="log", xlabel="Reynolds Number", ylabel="Fanning Friction Factor", title=title)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    save(fig, ax, filename, tight_bbox=True)
    print(f"Saved {OUT / filename} (R^2 = {r2:.4f})")


def all_correlations_plot(points, eps_over_d, filename):
    """Overlay all 5 correlations (each evaluated over its own applicable Re
    range, against all pipes) on one friction-factor-vs-Reynolds-number plot,
    shaded by flow regime, with each correlation's own R^2 labeled."""
    fig, ax = plt.subplots(figsize=(9, 7))
    re_min, re_max = min(p["Re"] for p in points), max(p["Re"] for p in points)
    ax.axvspan(re_min, cp.LAMINAR_RE, color="black", alpha=0.06)
    ax.axvspan(cp.LAMINAR_RE, cp.TURBULENT_RE, color="black", alpha=0.16)
    ax.axvline(cp.LAMINAR_RE, color="black", linestyle="--", linewidth=1)
    ax.axvline(cp.TURBULENT_RE, color="black", linestyle="--", linewidth=1)

    handles = plot_pipes_by_side(ax, points, "Re", "f", markersize=5, alpha=0.5)

    for label, x in [("Laminar", (re_min * cp.LAMINAR_RE)**0.5),
                     ("Transitional", (cp.LAMINAR_RE * cp.TURBULENT_RE)**0.5),
                     ("Turbulent", (cp.TURBULENT_RE * re_max)**0.5)]:
        ax.text(x, 0.97, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)

    correlations = [
        ("f = 16/Re", lambda re: 16 / re, lambda re: re < cp.LAMINAR_RE, "-."),
        ("f = 0.079 Re^-0.25 (Blasius)", lambda re: 0.079 * re**-0.25, lambda re: re > cp.TURBULENT_RE, ":"),
        ("Haaland", lambda re: cp.haaland_fanning(re, eps_over_d), lambda re: re > cp.TURBULENT_RE, "--"),
        ("Churchill", lambda re: cp.churchill_fanning(re, eps_over_d), lambda re: True, "-"),
        ("Romeo, Royo & Monzon", lambda re: cp.romeo_royo_monzon_fanning(re, eps_over_d),
         lambda re: re > cp.TURBULENT_RE, (0, (3, 1, 1, 1))),
    ]

    r2_lines = []
    for label, correlation, in_regime, linestyle in correlations:
        subset = [p for p in points if in_regime(p["Re"])]
        if not subset:
            continue
        sub_re_min, sub_re_max = min(p["Re"] for p in subset), max(p["Re"] for p in subset)
        re_curve = [sub_re_min * (sub_re_max / sub_re_min)**(i / 99) for i in range(100)]
        handles += ax.plot(re_curve, [correlation(x) for x in re_curve], linestyle=linestyle, color="black",
                           linewidth=1.6, label=label)
        r2 = r_squared([p["f"] for p in subset], [correlation(p["Re"]) for p in subset])
        r2_lines.append(f"{label}: R² = {r2:.4f}")

    ax.text(0.015, 0.015, "\n".join(r2_lines), transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))
    ax.set(xscale="log", yscale="log", xlabel="Reynolds Number", ylabel="Fanning Friction Factor",
           title="Friction Factor vs. Reynolds Number: All Correlations (All Pipes)")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    save(fig, ax, filename, tight_bbox=True)
    print(f"Saved {OUT / filename}")
    for line in r2_lines:
        print(f"  {line}")


def valve_flow_pressure_plot(points, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    for opening in cp.VALVE_OPENINGS:
        sub = sorted([p for p in points if p["opening"] == opening], key=lambda p: p["Q"])
        ax.errorbar([p["Q"] for p in sub], [p["P"] for p in sub], xerr=[p["U_Q"] for p in sub],
                    yerr=[p["U_P"] for p in sub], fmt=cp.VALVE_OPENING_MARKERS[opening], label=opening,
                    **ERRORBAR_STYLE)
    ax.set(xscale="log", yscale="log", xlabel="Flow Rate (LPM)", ylabel="Pressure Drop (Pa)", title=title)
    ax.legend()
    save(fig, ax, filename)
    print(f"Saved {OUT / filename}")


def valve_vs_reynolds_plot(points, key, literature, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    handles = []
    for opening in cp.ALL_VALVE_OPENINGS:
        sub = sorted([p for p in points if p["opening"] == opening], key=lambda p: p["Re"])
        re = [p["Re"] for p in sub]
        marker = cp.ALL_VALVE_OPENING_MARKERS[opening]
        handles.append(ax.errorbar(re, [p[key] for p in sub], xerr=[p["U_Re"] for p in sub],
                                   yerr=[p["U_" + key] for p in sub], fmt=marker, label=f"{opening} (measured)",
                                   **ERRORBAR_STYLE))
        handles += ax.plot([min(re), max(re)], [literature[opening]] * 2, "--", marker=marker, color="black",
                           markerfacecolor="none", label=f"{opening} (literature)")
    ax.set(xscale="log", yscale="log", xlabel="Reynolds Number", ylabel=ylabel, title=title)
    ticks = [2.2e4, 2.6e4, 3.0e4, 3.4e4]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([rf"${t / 1e4:.1f}\times10^{{4}}$" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    save(fig, ax, filename, tight_bbox=True)
    print(f"Saved {OUT / filename}")


def print_points(pipes, valves):
    print("\nPipe points (95% uncertainty)")
    print(f"{'run':5} {'pipe':14} {'n kept':>8} {'T (C)':>6} {'Q (L/min)':>17} {'dP (Pa)':>19} "
          f"{'Re':>17} {'f':>19}")
    for p in sorted(pipes, key=lambda p: (p["material"], p["D_in"], p["Q"])):
        print(f"{p['id']:5} {cp.pipe_label(p['material'], p['D_in']):14} {len(p['kept']):>3}/{len(p['rows']):<4} "
              f"{p['T']:6.2f} {p['Q']:8.3f} ± {p['U_Q']:6.3f} {p['P']:9.1f} ± {p['U_P']:7.1f} "
              f"{p['Re']:7.0f} ± {p['U_Re']:7.0f} {p['f']:.5f} ± {p['U_f']:.5f}")
    print("\nValve points (95% uncertainty)")
    print(f"{'run':5} {'opening':11} {'n kept':>8} {'T (C)':>6} {'Q (L/min)':>17} {'dP (Pa)':>19} "
          f"{'Re':>17} {'K':>13} {'Leq/D':>15}")
    for p in sorted(valves, key=lambda p: (cp.ALL_VALVE_OPENINGS.index(p["opening"]), p["Q"])):
        print(f"{p['id']:5} {p['opening']:11} {len(p['kept']):>3}/{len(p['rows']):<4} {p['T']:6.2f} "
              f"{p['Q']:8.3f} ± {p['U_Q']:6.3f} {p['P']:9.1f} ± {p['U_P']:7.1f} {p['Re']:7.0f} ± {p['U_Re']:7.0f} "
              f"{p['K']:5.2f} ± {p['U_K']:5.2f} {p['Leq']:6.1f} ± {p['U_Leq']:5.1f}")
    print()


def main():
    OUT.mkdir(exist_ok=True)
    runs = read_runs()
    for run in runs:
        describe(run)
    interpolate_cut_off_temperatures(runs)
    for run in runs:
        clean(run)
    print("\nDropped runs")
    for run in runs:
        if run["dropped"]:
            print(f"  {run['id']:5} {run['note']:36} {run['dropped']}")
    used = [r for r in runs if not r["dropped"]]
    print(f"Dropped readings in kept runs: {sum(len(r['rows']) - len(r['kept']) for r in used)}")

    T_rho = statistics.mean(r["T"] for r in used if r["kind"] == "pipe" and not r["id"].startswith("T"))
    rho = cp.density_kell(T_rho)
    print(f"Density (Kell) at the mean lab-day pipe temperature {T_rho:.2f} C: {rho:.3f} kg/m^3")
    for run in used:
        summarize(run, rho, T_rho)

    pipes = [r for r in used if r["kind"] == "pipe"]
    valves = [r for r in used if r["kind"] == "valve"]
    print_points(pipes, valves)

    pvc_diameters = {s["diameter_in"] for (m, _), s in cp.PIPE_SPECS.items() if m == "pvc"}
    small_diameters = {cp.PIPE_SPECS[("pvc", 0.285)]["diameter_in"], cp.PIPE_SPECS[("steel", 0.31)]["diameter_in"],
                       cp.PIPE_SPECS[("copper", 0.31)]["diameter_in"]}
    pvc = [p for p in pipes if p["D_in"] in pvc_diameters]
    small = [p for p in pipes if p["D_in"] in small_diameters]

    flow_vs_pressure_plot(pvc, "Pressure Drop vs. Flow Rate (PVC Pipes)", "Figure_1.png")
    flow_vs_pressure_plot(small, "Pressure Drop vs. Flow Rate (0.285\" PVC, Steel, Copper)", "Figure_2.png")
    friction_plot(small, "Friction Factor vs. Reynolds Number (0.285\" PVC, Steel, Copper)", "Figure_3.png")
    friction_plot(pvc, "Friction Factor vs. Reynolds Number (PVC Pipes)", "Figure_4.png")
    regime_comparison_plot(pipes, lambda re: re < cp.LAMINAR_RE, lambda re: 16 / re, "f = 16/Re",
                           "Laminar Friction Factor vs. f = 16/Re Correlation (All Pipes)", "Figure_5.png")
    regime_comparison_plot(pipes, lambda re: re > cp.TURBULENT_RE, lambda re: 0.079 * re**-0.25,
                           "f = 0.079 Re^-0.25 (Blasius)",
                           "Turbulent Friction Factor vs. Blasius Correlation (All Pipes)", "Figure_6.png")
    regime_comparison_plot(pipes, lambda re: re > cp.TURBULENT_RE,
                           lambda re: cp.haaland_fanning(re, MEAN_EPS_OVER_D), "Haaland correlation",
                           "Turbulent Friction Factor vs. Haaland Correlation (All Pipes)", "Figure_7.png")
    regime_comparison_plot(pipes, lambda re: True, lambda re: cp.churchill_fanning(re, MEAN_EPS_OVER_D),
                           "Churchill correlation",
                           "Friction Factor vs. Churchill Correlation (All Pipes, All Flow Regimes)", "Figure_8.png")
    regime_comparison_plot(pipes, lambda re: re > cp.TURBULENT_RE,
                           lambda re: cp.romeo_royo_monzon_fanning(re, MEAN_EPS_OVER_D),
                           "Romeo, Royo & Monzon correlation",
                           "Turbulent Friction Factor vs. Romeo, Royo & Monzon Correlation (All Pipes)",
                           "Figure_9.png")
    valve_flow_pressure_plot(valves, "Pressure Drop vs. Flow Rate (Globe Valve, 1/2 / 3/4 / Fully Open)",
                             "Figure_10.png")
    f_t, literature_k = cp.literature_valve_k(cp.PIPE_SPECS[("copper", 0.31)]["roughness_in"], cp.VALVE_DIAMETER_IN)
    literature_leq = {opening: k / f_t for opening, k in literature_k.items()}
    valve_vs_reynolds_plot(valves, "K", literature_k, "Resistance Coefficient K",
                           "Resistance Coefficient vs. Reynolds Number (All Valve Openings)", "Figure_11.png")
    valve_vs_reynolds_plot(valves, "Leq", literature_leq, "Equivalent Length, Leq/D",
                           "Equivalent Length vs. Reynolds Number (All Valve Openings)", "Figure_12.png")
    all_correlations_plot(pipes, MEAN_EPS_OVER_D, "Figure_13.png")

    images = [Image.open(OUT / f"Figure_{n}.png").convert("RGB") for n in range(1, 14)]
    images[0].save(OUT / "Figures_1-13_errorbars.pdf", save_all=True, append_images=images[1:], resolution=200)
    print(f"Saved {OUT / 'Figures_1-13_errorbars.pdf'}")


if __name__ == "__main__":
    main()
