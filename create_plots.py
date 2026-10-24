import re
import statistics
import math
import matplotlib.pyplot as plt
import plot_style
from pipe_data import pipes as PIPE_MEASUREMENTS

MARKERS = ["o", "^", "s", "D", "v"]  
DISPLAY_MATERIAL = {"pvc": "PVC", "steel": "steel", "copper": "copper"}

PSI_TO_PA = 6894.757293168
DIAMETER_PATTERN = re.compile(r'(\d+\.\d+)\s*(?:"|inches)')
TEMP_PATTERN = re.compile(r'LPM[^\d]*(\d+\.?\d*)')
MATERIAL_PATTERN = re.compile(r'^(\S+)')
MATERIAL_NAMES = {"pvc": "pvc", "ss": "steel", "cu": "copper"}

_NOMINAL_DIAMETERS_IN = [0.41, 0.285, 0.47, 0.31, 0.31]
PIPE_SPECS = {
    (material, nominal): {"diameter_in": diameter_in, "length_in": length_in,
                           "roughness_in": roughness_microin * 1e-6}
    for (material, diameter_in, length_in, roughness_microin), nominal
    in zip(PIPE_MEASUREMENTS, _NOMINAL_DIAMETERS_IN)}

SPECS_BY_DIAMETER = {spec["diameter_in"]: spec for spec in PIPE_SPECS.values()}
PIPE_MARKERS = {d: m for d, m in zip(sorted(SPECS_BY_DIAMETER), MARKERS)}
#thresholds
LAMINAR_RE = 2300
TURBULENT_RE = 4000

# note misclick started logging before the SS 0.31" pipe reached steady state
# at 1.3 LPM on the lab slowflow run, drop those pre-steady-state samples
EXCLUDED_READINGS = {
    ("FLU-lab-Slowflow", "SS 0.31 inches 1.3 LPM 23.52", "1:06:07 PM"),
    ("FLU-lab-Slowflow", "SS 0.31 inches 1.3 LPM 23.52", "1:06:09 PM"),
}

def pipe_label(material, diameter_in):
    return f'{diameter_in:.3f}" {DISPLAY_MATERIAL[material]}'

def density_kell(t):
    numerator = (999.83952 + 16.945176 * t - 7.9870401e-3 * t**2
                 - 46.170461e-6 * t**3 + 105.56302e-9 * t**4 - 280.54253e-12 * t**5)
    denominator = 1 + 16.879850e-3 * t
    return numerator / denominator

def viscosity_kestin(t):
    mu20 = 1.002e-3  # Pa.s at 20C
    dt = 20 - t
    log_ratio = (dt * (1.2364 - 1.37e-3 * dt + 5.7e-6 * dt**2)) / (t + 96)
    return mu20 * 10**log_ratio

def parse_flow_pressure_groups(filename):
    """`side` is "high" for a *Highflow file, "slow" otherwise; the two use
    different pressure transducers ("FLU Rack: Side A"/"Side B") and ranges,
    so it's tracked per group alongside material/diameter/note."""
    side = "high" if "highflow" in filename.lower() else "slow"
    groups = {}
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines[4:]:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 4:
            continue
        try:
            flow = float(fields[1])
            pressure_psid = float(fields[2])
        except ValueError:
            continue
        note = fields[3].strip()
        if not note or "globe" in note.lower():
            continue
        if (filename, note, fields[0].strip()) in EXCLUDED_READINGS:
            continue
        diameter_match = DIAMETER_PATTERN.search(note)
        if not diameter_match:
            continue
        nominal_diameter = float(diameter_match.group(1))
        material_match = MATERIAL_PATTERN.match(note)
        material = MATERIAL_NAMES.get(material_match.group(1).lower()) if material_match else None
        if (material, nominal_diameter) not in PIPE_SPECS:
            continue
        groups.setdefault((material, nominal_diameter, note, side), []).append((flow, pressure_psid))

    return list(groups.items())

def parse_globe_valve_groups(filename):
    groups = {}
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines[4:]:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 4:
            continue
        try:
            flow = float(fields[1])
            pressure_psid = float(fields[2])
        except ValueError:
            continue
        note = fields[3].strip()
        if "globe" not in note.lower():
            continue
        groups.setdefault(note, []).append((flow, pressure_psid))

    return list(groups.items())

OPENING_PATTERN = re.compile(r'(\d/\d|fully)\s+open', re.IGNORECASE)
OPENING_LABELS = {"1/4": "1/4 open", "1/2": "1/2 open", "3/4": "3/4 open", "fully": "fully open"}
VALVE_OPENINGS = ["1/2 open", "3/4 open", "fully open"]  # openings shown in Figure 10
VALVE_OPENING_MARKERS = {opening: m for opening, m in zip(VALVE_OPENINGS, MARKERS)}

def globe_valve_flow_pressure_rows(groups):
    rows = []
    for note, readings in groups:
        readings = [(flow, psid) for flow, psid in readings if flow > 0 and psid > 0]
        if not readings:
            continue
        opening_match = OPENING_PATTERN.search(note)
        opening = OPENING_LABELS.get(opening_match.group(1).lower()) if opening_match else None
        if opening is None:
            continue
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        rows.append((opening, avg_flow, avg_dp))
    return sorted(rows)

VALVE_DIAMETER_IN = 0.791  # diameter of the pipe the half-open valve is installed in

def valve_resistance_rows(groups, rho):
    d = VALVE_DIAMETER_IN * 0.0254
    area = math.pi / 4 * d**2
    rows = []
    for note, readings in groups:
        readings = [(flow, psid) for flow, psid in readings if flow > 0 and psid > 0]
        if not readings:
            continue
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        temp = float(TEMP_PATTERN.search(note).group(1))
        mu = viscosity_kestin(temp)
        q = avg_flow / 60000  # LPM to m^3/s
        v = q / area
        re = rho * v * d / mu
        k = avg_dp / (0.5 * rho * v**2)
        rows.append((re, k))
    return sorted(rows)

ALL_VALVE_OPENINGS = ["1/4 open", "1/2 open", "3/4 open", "fully open"]  # openings shown in Figures 11-12
ALL_VALVE_OPENING_MARKERS = {opening: m for opening, m in zip(ALL_VALVE_OPENINGS, MARKERS)}
# restricted diameter as a fraction of the connecting pipe diameter, per opening
OPENING_FRACTIONS = {"1/4 open": 0.25, "1/2 open": 0.5, "3/4 open": 0.75, "fully open": 1.0}

def crane_fully_turbulent_friction_factor(roughness_in, diameter_in):
    eps_over_d = roughness_in / diameter_in
    return 0.25 / (math.log10(eps_over_d / 3.7))**2

def crane_valve_k(k_fully_open, beta):
    return (k_fully_open + beta * (0.5 * (1 - beta**2) + (1 - beta**2)**2)) / beta**4

def literature_valve_k(roughness_in, diameter_in):
    f_t = crane_fully_turbulent_friction_factor(roughness_in, diameter_in)
    k_fully_open = 340 * f_t
    k_by_opening = {}
    for opening, frac in OPENING_FRACTIONS.items():
        restricted_area = math.pi / 4 * (frac * diameter_in)**2
        pipe_area = math.pi / 4 * diameter_in**2
        beta = math.sqrt(restricted_area / pipe_area)
        k_by_opening[opening] = crane_valve_k(k_fully_open, beta)
    return f_t, k_by_opening

def valve_resistance_rows_by_opening(groups, rho):
    d = VALVE_DIAMETER_IN * 0.0254
    area = math.pi / 4 * d**2
    rows = []
    for note, readings in groups:
        readings = [(flow, psid) for flow, psid in readings if flow > 0 and psid > 0]
        if not readings:
            continue
        opening_match = OPENING_PATTERN.search(note)
        opening = OPENING_LABELS.get(opening_match.group(1).lower()) if opening_match else None
        if opening is None:
            continue
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        temp = float(TEMP_PATTERN.search(note).group(1))
        mu = viscosity_kestin(temp)
        q = avg_flow / 60000  # LPM to m^3/s
        v = q / area
        re = rho * v * d / mu
        k = avg_dp / (0.5 * rho * v**2)
        rows.append((opening, re, k))
    return sorted(rows)

def globe_valve_rows(groups, rho):
    rows = []
    for note, readings in groups:
        readings = [(flow, psid) for flow, psid in readings if flow > 0 and psid > 0]
        if not readings:
            continue
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        friction_loss = avg_dp / rho  
        rows.append((avg_flow, avg_dp, friction_loss))
    return sorted(rows)

def averages(groups):
    """Return material, measured diameter (in), avg_flow (LPM), avg_delta_p (Pa),
    temp (C), and side ("slow"/"high", the source file) per group."""
    rows = []
    for (material, nominal_diameter, note, side), readings in groups:
        readings = [(flow, psid) for flow, psid in readings if flow > 0 and psid > 0]
        if not readings:
            continue
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        temp = float(TEMP_PATTERN.search(note).group(1))
        diameter_in = PIPE_SPECS[(material, nominal_diameter)]["diameter_in"]
        rows.append((material, diameter_in, avg_flow, avg_dp, temp, side))
    return sorted(rows)

def reynolds_number(diameter_in, flow_rate_lpm, rho, mu):
    d = diameter_in * 0.0254
    area = math.pi / 4 * d**2
    q = flow_rate_lpm / 60000  
    v = q / area
    return rho * v * d / mu

def friction_factor(diameter_in, flow_rate_lpm, delta_p, rho):
    d = diameter_in * 0.0254
    length = SPECS_BY_DIAMETER[diameter_in]["length_in"] * 0.0254
    area = math.pi / 4 * d**2
    q = flow_rate_lpm / 60000  
    v = q / area
    return delta_p * d / (length * 2 * rho * v**2)

def haaland_fanning(re, eps_over_d):
    f_darcy = (-1.8 * math.log10((eps_over_d / 3.7)**1.11 + 6.9 / re))**-2
    return f_darcy / 4

def churchill_fanning(re, eps_over_d):
    a = (2.457 * math.log(1 / ((7 / re)**0.9 + 0.27 * eps_over_d)))**16
    b = (37530 / re)**16
    return 2 * ((8 / re)**12 + (a + b)**-1.5)**(1 / 12)

def romeo_royo_monzon_fanning(re, eps_over_d):
    inner = math.log10((eps_over_d / 7.7918)**0.9924 + (5.3326 / re)**0.9345)
    mid = math.log10((eps_over_d / 3.827) - (4.567 / re) * inner)
    outer = math.log10((eps_over_d / 3.7065) - (5.0272 / re) * mid)
    return (-4 * outer)**-2

def friction_reynolds(rows, rho):
    """Returns (material, diameter_in, re, f, side) per row."""
    results = []
    for material, diameter_in, avg_flow, avg_dp, temp, side in rows:
        mu = viscosity_kestin(temp)
        re = reynolds_number(diameter_in, avg_flow, rho, mu)
        f = friction_factor(diameter_in, avg_flow, avg_dp, rho)
        results.append((material, diameter_in, re, f, side))
    return results

def pipes_present(rows):
    return sorted(set((r[0], r[1]) for r in rows))

def plot_pipes_by_side(ax, rows, label_suffix="", **marker_kwargs):
    """Plot each (material, diameter) pipe's points, split by side: open markers
    for low-flow data, filled markers for high-flow data. Adds one legend entry
    per pipe (keyed to its filled series) plus two generic entries explaining
    the open/filled convention. `rows` entries have x, y at indices 2, 3 and
    end with a "slow"/"high" side field."""
    for material, diameter_in in pipes_present(rows):
        subset = [r for r in rows if r[0] == material and r[1] == diameter_in]
        marker = PIPE_MARKERS[diameter_in]

        low = [r for r in subset if r[-1] == "slow"]
        if low:
            ax.plot([r[2] for r in low], [r[3] for r in low], marker, color="black",
                    markerfacecolor="none", **marker_kwargs)

        high = [r for r in subset if r[-1] == "high"]
        if high:
            ax.plot([r[2] for r in high], [r[3] for r in high], marker, color="black",
                    markerfacecolor="black", label=f'{pipe_label(material, diameter_in)}{label_suffix}',
                    **marker_kwargs)

    ax.plot([], [], 'o', color="black", markerfacecolor="black", label="high-flow data")
    ax.plot([], [], 'o', color="black", markerfacecolor="none", label="low-flow data")

def create_flow_vs_pressure_plot(rows, title, filename):
    """Plot pressure drop vs. flow rate across both the low- and high-flow
    datasets (`side` is "slow"/"high", the source file -- they use different
    pressure transducers/ranges), but fit the power law (deltaP = a * Q^n) to
    each pipe's high-flow-side points only. High-flow points are filled;
    low-flow points are open. `rows` entries are
    (material, diameter_in, avg_flow, avg_dp, side)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    fit_info = []
    for material, diameter_in in pipes_present(rows):
        subset = [r for r in rows if r[0] == material and r[1] == diameter_in]
        marker = PIPE_MARKERS[diameter_in]

        low_subset = [r for r in subset if r[4] == "slow"]
        if low_subset:
            ax.plot([r[2] for r in low_subset], [r[3] for r in low_subset], marker,
                    color="black", markerfacecolor="none")

        high_subset = [r for r in subset if r[4] == "high"]
        if high_subset:
            ax.plot([r[2] for r in high_subset], [r[3] for r in high_subset], marker,
                    label=pipe_label(material, diameter_in), color="black", markerfacecolor="black")

        if not high_subset:
            continue
        fit_flow = [r[2] for r in high_subset]
        fit_dp = [r[3] for r in high_subset]
        a, n = power_law_fit(fit_flow, fit_dp)
        q_min, q_max = min(fit_flow), max(fit_flow)
        q_curve = [q_min * (q_max / q_min)**(i / 99) for i in range(100)]
        dp_curve = [a * qc**n for qc in q_curve]
        ax.plot(q_curve, dp_curve, '--', color="black")

        predicted = [a * qi**n for qi in fit_flow]
        mean_dp = statistics.mean(fit_dp)
        ss_res = sum((m - p)**2 for m, p in zip(fit_dp, predicted))
        ss_tot = sum((m - mean_dp)**2 for m in fit_dp)
        r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")
        fit_info.append((material, diameter_in, n, r_squared))

    ax.plot([], [], 'o', color="black", markerfacecolor="black", label="high-flow data (fit)")
    ax.plot([], [], 'o', color="black", markerfacecolor="none", label="low-flow data")

    annotation = "\n".join(
        f'{pipe_label(m, d)}: ΔP ~ Q^{n:.2f} (R² = {r2:.4f})' for m, d, n, r2 in fit_info)
    ax.text(0.97, 0.03, annotation, transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow Rate (LPM)")
    ax.set_ylabel("Pressure Drop (Pa)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    fit_summary = ", ".join(f'{pipe_label(m, d)}: n={n:.3f}, R^2={r2:.4f}' for m, d, n, r2 in fit_info)
    print(f"Saved plot to {filename} ({fit_summary})")

def create_globe_valve_plot(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, 'o', color="black", markerfacecolor="black")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def create_valve_flow_pressure_plot(rows, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))

    for opening in VALVE_OPENINGS:
        subset = [r for r in rows if r[0] == opening]
        if not subset:
            continue
        avg_flow = [r[1] for r in subset]
        avg_dp = [r[2] for r in subset]
        ax.plot(avg_flow, avg_dp, VALVE_OPENING_MARKERS[opening], label=opening,
                color="black", markerfacecolor="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow Rate (LPM)")
    ax.set_ylabel("Pressure Drop (Pa)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def create_valve_resistance_by_opening_plot(rows, literature, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))

    for opening in ALL_VALVE_OPENINGS:
        subset = [r for r in rows if r[0] == opening]
        if not subset:
            continue
        re = [r[1] for r in subset]
        k = [r[2] for r in subset]
        ax.plot(re, k, ALL_VALVE_OPENING_MARKERS[opening], label=f'{opening} (measured)',
                color="black", markerfacecolor="black")
        ax.plot([min(re), max(re)], [literature[opening]] * 2, '--',
                marker=ALL_VALVE_OPENING_MARKERS[opening], color="black", markerfacecolor="none",
                label=f'{opening} (literature)')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Resistance Coefficient K")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {filename}")

def create_equivalent_length_plot(rows, literature, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))

    for opening in ALL_VALVE_OPENINGS:
        subset = [r for r in rows if r[0] == opening]
        if not subset:
            continue
        re = [r[1] for r in subset]
        leq = [r[2] for r in subset]
        ax.plot(re, leq, ALL_VALVE_OPENING_MARKERS[opening], label=f'{opening} (measured)',
                color="black", markerfacecolor="black")
        ax.plot([min(re), max(re)], [literature[opening]] * 2, '--',
                marker=ALL_VALVE_OPENING_MARKERS[opening], color="black", markerfacecolor="none",
                label=f'{opening} (literature)')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Equivalent Length, Leq/D")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {filename}")

def create_friction_plot(results, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))

    re_min = min(r[2] for r in results)
    re_max = max(r[2] for r in results)
    ax.axvspan(re_min, LAMINAR_RE, color="black", alpha=0.06)
    ax.axvspan(LAMINAR_RE, TURBULENT_RE, color="black", alpha=0.16)
    ax.axvspan(TURBULENT_RE, re_max, color="black", alpha=0.0)
    ax.axvline(LAMINAR_RE, color="black", linestyle="--", linewidth=1)
    ax.axvline(TURBULENT_RE, color="black", linestyle="--", linewidth=1)

    plot_pipes_by_side(ax, results)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Fanning Friction Factor")
    ax.set_title(title)

    for label, x in [("Laminar", (re_min * LAMINAR_RE)**0.5),
                      ("Transitional", (LAMINAR_RE * TURBULENT_RE)**0.5),
                      ("Turbulent", (TURBULENT_RE * re_max)**0.5)]:
        ax.text(x, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="black")

    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def create_valve_resistance_plot(rows, k_reference, filename="half_open_valve_resistance_vs_reynolds.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    re = [r[0] for r in rows]
    k = [r[1] for r in rows]
    ax.plot(re, k, 'o', label="measured", color="black", markerfacecolor="black")

    re_min, re_max = min(re), max(re)
    ax.plot([re_min, re_max], [k_reference, k_reference], '--', color="black",
            label=f"K = {k_reference}")

    ax.set_xscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Resistance Coefficient K")
    ax.set_title("Resistance Coefficient vs. Reynolds Number (Half-Open Valve)")
    ax.legend()
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def power_law_fit(x, y):
    log_x = [math.log(v) for v in x]
    log_y = [math.log(v) for v in y]
    mean_x = statistics.mean(log_x)
    mean_y = statistics.mean(log_y)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(log_x, log_y))
    den = sum((xi - mean_x)**2 for xi in log_x)
    n = num / den
    a = math.exp(mean_y - n * mean_x)
    return a, n

def create_powerlaw_plot(rows, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))

    diameters = sorted(set(r[0] for r in rows))
    markers = {d: m for d, m in zip(diameters, MARKERS)}

    fit_info = []
    for diameter in diameters:
        subset = [r for r in rows if r[0] == diameter]
        q = [r[1] for r in subset]
        dp = [r[2] for r in subset]
        ax.plot(q, dp, markers[diameter], label=f'{diameter}" pipe (measured)',
                color="black", markerfacecolor="black")

        a, n = power_law_fit(q, dp)
        q_min, q_max = min(q), max(q)
        q_curve = [q_min * (q_max / q_min)**(i / 99) for i in range(100)]
        dp_curve = [a * qc**n for qc in q_curve]
        ax.plot(q_curve, dp_curve, '--', color="black")

        predicted = [a * qi**n for qi in q]
        mean_dp = statistics.mean(dp)
        ss_res = sum((m - p)**2 for m, p in zip(dp, predicted))
        ss_tot = sum((m - mean_dp)**2 for m in dp)
        r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")
        fit_info.append((diameter, n, r_squared))

    annotation = "\n".join(
        f'{d}" pipe: ΔP ~ Q^{n:.2f} (R² = {r2:.4f})' for d, n, r2 in fit_info)
    ax.text(0.03, 0.97, annotation, transform=ax.transAxes, fontsize=10, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow Rate (LPM)")
    ax.set_ylabel("Pressure Drop (Pa)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    fit_summary = ", ".join(f'{d}": n={n:.3f}, R^2={r2:.4f}' for d, n, r2 in fit_info)
    print(f"Saved plot to {filename} ({fit_summary})")

def create_regime_comparison_plot(results, regime_filter, correlation, correlation_label, title, filename):
    subset_all = [r for r in results if regime_filter(r[2])]
    if not subset_all:
        print(f"No data in range for {filename}.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_pipes_by_side(ax, subset_all, label_suffix=" (measured)")

    re_min = min(r[2] for r in subset_all)
    re_max = max(r[2] for r in subset_all)
    re_curve = [re_min * (re_max / re_min)**(i / 99) for i in range(100)]
    f_curve = [correlation(re) for re in re_curve]
    ax.plot(re_curve, f_curve, '--', color="black", label=correlation_label)

    measured_f = [r[3] for r in subset_all]
    predicted_f = [correlation(r[2]) for r in subset_all]
    mean_f = statistics.mean(measured_f)
    ss_res = sum((mf - pf)**2 for mf, pf in zip(measured_f, predicted_f))
    ss_tot = sum((mf - mean_f)**2 for mf in measured_f)
    r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")
    ax.text(0.03, 0.03, f"R² = {r_squared:.4f}", transform=ax.transAxes,
            fontsize=10, va="bottom", ha="left",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Fanning Friction Factor")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {filename} (R^2 = {r_squared:.4f})")

def create_all_correlations_plot(results, eps_over_d, filename):
    """Overlay all 5 correlations (each evaluated over its own applicable Re
    range, against all pipes) on one friction-factor-vs-Reynolds-number plot,
    shaded by flow regime, with each correlation's own R^2 labeled."""
    fig, ax = plt.subplots(figsize=(9, 7))

    re_min = min(r[2] for r in results)
    re_max = max(r[2] for r in results)
    ax.axvspan(re_min, LAMINAR_RE, color="black", alpha=0.06)
    ax.axvspan(LAMINAR_RE, TURBULENT_RE, color="black", alpha=0.16)
    ax.axvspan(TURBULENT_RE, re_max, color="black", alpha=0.0)
    ax.axvline(LAMINAR_RE, color="black", linestyle="--", linewidth=1)
    ax.axvline(TURBULENT_RE, color="black", linestyle="--", linewidth=1)

    plot_pipes_by_side(ax, results, markersize=5, alpha=0.5)

    for label, x in [("Laminar", (re_min * LAMINAR_RE)**0.5),
                      ("Transitional", (LAMINAR_RE * TURBULENT_RE)**0.5),
                      ("Turbulent", (TURBULENT_RE * re_max)**0.5)]:
        ax.text(x, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="black")

    correlations = [
        ("f = 16/Re", lambda re: 16 / re, lambda re: re < LAMINAR_RE, "-."),
        ("f = 0.079 Re^-0.25 (Blasius)", lambda re: 0.079 * re**-0.25, lambda re: re > TURBULENT_RE, ":"),
        ("Haaland", lambda re: haaland_fanning(re, eps_over_d), lambda re: re > TURBULENT_RE, "--"),
        ("Churchill", lambda re: churchill_fanning(re, eps_over_d), lambda re: True, "-"),
        ("Romeo, Royo & Monzon", lambda re: romeo_royo_monzon_fanning(re, eps_over_d),
         lambda re: re > TURBULENT_RE, (0, (3, 1, 1, 1))),
    ]

    r2_lines = []
    for label, correlation, regime_filter, linestyle in correlations:
        subset = [r for r in results if regime_filter(r[2])]
        if not subset:
            continue
        sub_re_min = min(r[2] for r in subset)
        sub_re_max = max(r[2] for r in subset)
        re_curve = [sub_re_min * (sub_re_max / sub_re_min)**(i / 99) for i in range(100)]
        f_curve = [correlation(re) for re in re_curve]
        ax.plot(re_curve, f_curve, linestyle=linestyle, color="black", linewidth=1.6, label=label)

        measured_f = [r[3] for r in subset]
        predicted_f = [correlation(r[2]) for r in subset]
        mean_f = statistics.mean(measured_f)
        ss_res = sum((mf - pf)**2 for mf, pf in zip(measured_f, predicted_f))
        ss_tot = sum((mf - mean_f)**2 for mf in measured_f)
        r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")
        r2_lines.append(f"{label}: R² = {r_squared:.4f}")

    annotation = "\n".join(r2_lines)
    ax.text(0.015, 0.015, annotation, transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Fanning Friction Factor")
    ax.set_title("Friction Factor vs. Reynolds Number: All Correlations (All Pipes)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {filename}")
    for line in r2_lines:
        print(f"  {line}")

if __name__ == "__main__":
    # prelab stuff
    slowflow_groups = parse_flow_pressure_groups("FLU-prelab-Slowflow")
    highflow_groups = parse_flow_pressure_groups("FLU_Prelab_Highflow")

    rows = averages(slowflow_groups + highflow_groups)

    for material, diameter_in, avg_flow, avg_dp, temp, side in rows:
        print(f'{pipe_label(material, diameter_in)} pipe: Q={avg_flow:.2f} LPM, dP={avg_dp:.1f} Pa, T={temp:.2f} C')

    # density from the Kell correlation at the average temperature across all
    # prelab runs; viscosity is recalculated per point (temperature per run)
    # from the Kestin correlation
    avg_temp = statistics.mean(r[4] for r in rows)
    rho = density_kell(avg_temp)
    print(f"Average temp across prelab runs: {avg_temp:.2f} C -> rho (Kell) = {rho:.3f} kg/m^3")

    friction_results = friction_reynolds(rows, rho)
    for material, diameter_in, re, f, side in friction_results:
        print(f'{pipe_label(material, diameter_in)} pipe: Re={re:.0f}, f={f:.4f}')

    # (diameter, avg_flow, avg_dp) split by flow regime, using each row's own Reynolds
    # number from friction_results (rows and friction_results share index order)
    laminar_flow_rows = [(d, q, dp) for (_, d, q, dp, _, _), (_, _, re, _, _) in zip(rows, friction_results) if re < LAMINAR_RE]
    turbulent_flow_rows = [(d, q, dp) for (_, d, q, dp, _, _), (_, _, re, _, _) in zip(rows, friction_results) if re > TURBULENT_RE]

    create_powerlaw_plot(
        laminar_flow_rows, "Laminar Pressure Drop vs. Flow Rate (Power-Law Fit)",
        "laminar_powerlaw_fit.png")

    create_powerlaw_plot(
        turbulent_flow_rows, "Turbulent Pressure Drop vs. Flow Rate (Power-Law Fit)",
        "turbulent_powerlaw_fit.png")

    globe_groups = parse_globe_valve_groups("FLU_Prelab_Highflow")
    globe_rows = globe_valve_rows(globe_groups, rho)

    for avg_flow, avg_dp, friction_loss in globe_rows:
        print(f'Globe valve: Q={avg_flow:.2f} LPM, dP={avg_dp:.1f} Pa, friction loss={friction_loss:.2f} J/kg')

    create_globe_valve_plot(
        [r[0] for r in globe_rows], [r[2] for r in globe_rows],
        "Flow Rate (LPM)", "Friction Loss (J/kg)", "Friction Loss vs. Flow Rate (Globe Valve)",
        "globe_valve_friction_loss_vs_flow.png")

    create_globe_valve_plot(
        [r[1] for r in globe_rows], [r[2] for r in globe_rows],
        "Pressure Drop (Pa)", "Friction Loss (J/kg)", "Friction Loss vs. Pressure Drop (Globe Valve)",
        "globe_valve_friction_loss_vs_pressure.png")

    K_REFERENCE = 8.5  # Perry's handbook K for a half-open globe valve

    valve_resistance = valve_resistance_rows(globe_groups, rho)
    for re, k in valve_resistance:
        print(f'1/2 open valve: Re={re:.0f}, K={k:.2f}')

    create_valve_resistance_plot(valve_resistance, K_REFERENCE)


    # New combined prelab + lab analysis 
  
    all_groups = (
        parse_flow_pressure_groups("FLU-prelab-Slowflow")
        + parse_flow_pressure_groups("FLU_Prelab_Highflow")
        + parse_flow_pressure_groups("FLU-lab-Slowflow")
        + parse_flow_pressure_groups("FLU_Lab_Highflow"))
    all_rows = averages(all_groups)

    for material, diameter_in, avg_flow, avg_dp, temp, side in all_rows:
        print(f'{pipe_label(material, diameter_in)} pipe: Q={avg_flow:.2f} LPM, dP={avg_dp:.1f} Pa, T={temp:.2f} C ({side}-flow)')


    lab_day_rows = averages(
        parse_flow_pressure_groups("FLU-lab-Slowflow") + parse_flow_pressure_groups("FLU_Lab_Highflow"))
    avg_temp_lab_day = statistics.mean(r[4] for r in lab_day_rows)
    rho_all = density_kell(avg_temp_lab_day)
    print(f"Average temp over the lab day: {avg_temp_lab_day:.2f} C -> rho (Kell) = {rho_all:.3f} kg/m^3")

    friction_results_all = friction_reynolds(all_rows, rho_all)
    for material, diameter_in, re, f, side in friction_results_all:
        print(f'{pipe_label(material, diameter_in)} pipe: Re={re:.0f}, f={f:.4f}')

    pvc_diameters = {spec["diameter_in"] for (m, _), spec in PIPE_SPECS.items() if m == "pvc"}
    small_diameters = {PIPE_SPECS[("pvc", 0.285)]["diameter_in"],
                        PIPE_SPECS[("steel", 0.31)]["diameter_in"],
                        PIPE_SPECS[("copper", 0.31)]["diameter_in"]}

    # (material, diameter_in, avg_flow, avg_dp, side), dropping temp since the
    # flow-vs-pressure plots key their fit (and open/filled marker) on side
    all_rows_with_side = [(material, diameter_in, avg_flow, avg_dp, side)
                           for material, diameter_in, avg_flow, avg_dp, temp, side in all_rows]

    pvc_flow_rows = [r for r in all_rows_with_side if r[1] in pvc_diameters]
    small_flow_rows = [r for r in all_rows_with_side if r[1] in small_diameters]
    pvc_friction_results = [r for r in friction_results_all if r[1] in pvc_diameters]
    small_friction_results = [r for r in friction_results_all if r[1] in small_diameters]

    create_flow_vs_pressure_plot(
        pvc_flow_rows, "Pressure Drop vs. Flow Rate (PVC Pipes)", "Figure_1.png")

    create_flow_vs_pressure_plot(
        small_flow_rows, "Pressure Drop vs. Flow Rate (0.285\" PVC, Steel, Copper)", "Figure_2.png")

    create_friction_plot(
        small_friction_results, "Friction Factor vs. Reynolds Number (0.285\" PVC, Steel, Copper)",
        "Figure_3.png")

    create_friction_plot(
        pvc_friction_results, "Friction Factor vs. Reynolds Number (PVC Pipes)", "Figure_4.png")

    # all 5 pipes are hydraulically smooth (eps/D ~1e-5), so their correlation
    # curves are visually nearly identical. use the mean eps/D across all pipes for correlations
    # for a single representative curve
    mean_eps_over_d = statistics.mean(spec["roughness_in"] / spec["diameter_in"]
                                       for spec in SPECS_BY_DIAMETER.values())

    create_regime_comparison_plot(
        friction_results_all, lambda re: re < LAMINAR_RE, lambda re: 16 / re,
        "f = 16/Re", "Laminar Friction Factor vs. f = 16/Re Correlation (All Pipes)",
        "Figure_5.png")

    create_regime_comparison_plot(
        friction_results_all, lambda re: re > TURBULENT_RE, lambda re: 0.079 * re**-0.25,
        "f = 0.079 Re^-0.25 (Blasius)", "Turbulent Friction Factor vs. Blasius Correlation (All Pipes)",
        "Figure_6.png")

    create_regime_comparison_plot(
        friction_results_all, lambda re: re > TURBULENT_RE,
        lambda re: haaland_fanning(re, mean_eps_over_d),
        "Haaland correlation", "Turbulent Friction Factor vs. Haaland Correlation (All Pipes)",
        "Figure_7.png")

    create_regime_comparison_plot(
        friction_results_all, lambda re: True,
        lambda re: churchill_fanning(re, mean_eps_over_d),
        "Churchill correlation", "Friction Factor vs. Churchill Correlation (All Pipes, All Flow Regimes)",
        "Figure_8.png")

    create_regime_comparison_plot(
        friction_results_all, lambda re: re > TURBULENT_RE,
        lambda re: romeo_royo_monzon_fanning(re, mean_eps_over_d),
        "Romeo, Royo & Monzon correlation",
        "Turbulent Friction Factor vs. Romeo, Royo & Monzon Correlation (All Pipes)",
        "Figure_9.png")

    globe_groups_all = (
        parse_globe_valve_groups("FLU_Prelab_Highflow")
        + parse_globe_valve_groups("FLU_Lab_Highflow"))
    valve_opening_rows = [r for r in globe_valve_flow_pressure_rows(globe_groups_all) if r[0] in VALVE_OPENINGS]

    create_valve_flow_pressure_plot(
        valve_opening_rows, "Pressure Drop vs. Flow Rate (Globe Valve, 1/2 / 3/4 / Fully Open)",
        "Figure_10.png")

    valve_k_rows = valve_resistance_rows_by_opening(globe_groups_all, rho_all)
    for opening, re, k in valve_k_rows:
        print(f'{opening} valve: Re={re:.0f}, K={k:.2f}')

    # Crane TP-410 literature comparison: K for the fully open valve is 340 * f_t,
    # the fully-turbulent (Darcy) friction factor of the connecting pipe (using
    # copper's roughness, per the assignment). other openings follow Crane's
    # valve-in-reducer beta correction, assuming the restricted diameter is
    # the stated fraction of the connecting pipe diameter
    copper_roughness_in = PIPE_SPECS[("copper", 0.31)]["roughness_in"]
    f_t, literature_k = literature_valve_k(copper_roughness_in, VALVE_DIAMETER_IN)
    literature_leq = {opening: k / f_t for opening, k in literature_k.items()}
    for opening in ALL_VALVE_OPENINGS:
        print(f'{opening} valve (Crane literature): K={literature_k[opening]:.2f}, Leq/D={literature_leq[opening]:.1f}')

    create_valve_resistance_by_opening_plot(
        valve_k_rows, literature_k, "Resistance Coefficient vs. Reynolds Number (All Valve Openings)",
        "Figure_11.png")

    # deltaP/rho = 4f * Leq * (v^2/2) = K * (v^2/2), so Leq (in pipe diameters) = K / (4f);
    # f is the Fanning friction factor at the valve's own Re, from the Churchill
    # correlation using the same representative smooth-pipe eps/D as Figures 5-9
    valve_leq_rows = [(opening, re, k / (4 * churchill_fanning(re, mean_eps_over_d)))
                       for opening, re, k in valve_k_rows]
    for opening, re, leq in valve_leq_rows:
        print(f'{opening} valve: Re={re:.0f}, Leq/D={leq:.1f}')

    create_equivalent_length_plot(
        valve_leq_rows, literature_leq, "Equivalent Length vs. Reynolds Number (All Valve Openings)",
        "Figure_12.png")

    create_all_correlations_plot(friction_results_all, mean_eps_over_d, "Figure_13.png")
