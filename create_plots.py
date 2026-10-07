import re
import statistics
import math
import matplotlib.pyplot as plt
import plot_style
from pipe_data import pipes as PIPE_MEASUREMENTS

MARKERS = ["o", "^", "s", "D", "v"]  # distinct marker per pipe, black-and-white friendly
DISPLAY_MATERIAL = {"pvc": "PVC", "steel": "steel", "copper": "copper"}

PSI_TO_PA = 6894.757293168
DIAMETER_PATTERN = re.compile(r'(\d+\.\d+)\s*(?:"|inches)')
TEMP_PATTERN = re.compile(r'LPM[^\d]*(\d+\.?\d*)')
MATERIAL_PATTERN = re.compile(r'^(\S+)')
MATERIAL_NAMES = {"pvc": "pvc", "ss": "steel", "cu": "copper"}  # note token -> pipe_data.py material

# nominal diameter as labeled in the logging notes, in the same pipe order as
# PIPE_MEASUREMENTS (material, measured diameter_in, length_in, roughness_microin)
_NOMINAL_DIAMETERS_IN = [0.41, 0.285, 0.47, 0.31, 0.31]
PIPE_SPECS = {
    (material, nominal): {"diameter_in": diameter_in, "length_in": length_in,
                           "roughness_in": roughness_microin * 1e-6}
    for (material, diameter_in, length_in, roughness_microin), nominal
    in zip(PIPE_MEASUREMENTS, _NOMINAL_DIAMETERS_IN)
}
# same specs keyed by measured diameter alone, for lookups once a reading has
# been matched to its pipe (measured diameters are unique across all 5 pipes)
SPECS_BY_DIAMETER = {spec["diameter_in"]: spec for spec in PIPE_SPECS.values()}
PIPE_MARKERS = {d: m for d, m in zip(sorted(SPECS_BY_DIAMETER), MARKERS)}

LAMINAR_RE = 2300
TURBULENT_RE = 4000

# a brief misclick started logging before the SS 0.31" pipe reached steady state
# at 1.3 LPM on the lab slowflow run; drop those pre-steady-state samples
EXCLUDED_READINGS = {
    ("FLU-lab-Slowflow", "SS 0.31 inches 1.3 LPM 23.52", "1:06:07 PM"),
    ("FLU-lab-Slowflow", "SS 0.31 inches 1.3 LPM 23.52", "1:06:09 PM"),
}

def pipe_label(material, diameter_in):
    return f'{diameter_in:.3f}" {DISPLAY_MATERIAL[material]}'

def density_kell(t):
    """Kell (1975) equation for the density of water at 1 atm. t in C, returns kg/m^3."""
    numerator = (999.83952 + 16.945176 * t - 7.9870401e-3 * t**2
                 - 46.170461e-6 * t**3 + 105.56302e-9 * t**4 - 280.54253e-12 * t**5)
    denominator = 1 + 16.879850e-3 * t
    return numerator / denominator

def viscosity_kestin(t):
    """Kestin, Sokolov & Wakeham (1978) equation for the viscosity of water at 1 atm.
    t in C, returns viscosity in Pa.s."""
    mu20 = 1.002e-3  # Pa.s at 20C
    dt = 20 - t
    log_ratio = (dt * (1.2364 - 1.37e-3 * dt + 5.7e-6 * dt**2)) / (t + 96)
    return mu20 * 10**log_ratio

def parse_flow_pressure_groups(filename):
    """Group rows by logging note (one pipe/flow-rate setting) and return each
    group's material, nominal diameter (as labeled in the note), and its flow
    rate / pressure drop readings, in file order."""
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
        groups.setdefault((material, nominal_diameter, note), []).append((flow, pressure_psid))

    return list(groups.items())

def parse_globe_valve_groups(filename):
    """Group globe-valve rows by logging note (one valve-opening/flow-rate setting)
    and return per-group flow rate and pressure drop readings, in file order."""
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
    """Return valve opening label, avg_flow (LPM), and avg_delta_p (Pa) per group."""
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
    """Return Reynolds number and resistance coefficient K per group, based on
    the pipe diameter at the valve (VALVE_DIAMETER_IN). `rho` (kg/m^3) is the
    fixed density used for the whole dataset; viscosity is recalculated per
    group from the Kestin correlation using that group's own temperature."""
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
        q = avg_flow / 60000  # LPM -> m^3/s
        v = q / area
        re = rho * v * d / mu
        k = avg_dp / (0.5 * rho * v**2)
        rows.append((re, k))
    return sorted(rows)

ALL_VALVE_OPENINGS = ["1/4 open", "1/2 open", "3/4 open", "fully open"]  # openings shown in Figures 11-12
ALL_VALVE_OPENING_MARKERS = {opening: m for opening, m in zip(ALL_VALVE_OPENINGS, MARKERS)}

def valve_resistance_rows_by_opening(groups, rho):
    """Return valve opening label, Reynolds number, and resistance coefficient K per
    group, based on the pipe diameter at the valve (VALVE_DIAMETER_IN). `rho` (kg/m^3)
    is the fixed density used for the whole dataset; viscosity is recalculated per
    group from the Kestin correlation using that group's own temperature."""
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
        q = avg_flow / 60000  # LPM -> m^3/s
        v = q / area
        re = rho * v * d / mu
        k = avg_dp / (0.5 * rho * v**2)
        rows.append((opening, re, k))
    return sorted(rows)

def globe_valve_rows(groups, rho):
    """Return avg_flow (LPM), avg_delta_p (Pa), and friction loss as specific energy
    (J/kg) per group. `rho` (kg/m^3) is the fixed density used for the whole dataset."""
    rows = []
    for note, readings in groups:
        readings = [(flow, psid) for flow, psid in readings if flow > 0 and psid > 0]
        if not readings:
            continue
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        friction_loss = avg_dp / rho  # specific energy loss, J/kg
        rows.append((avg_flow, avg_dp, friction_loss))
    return sorted(rows)

def averages(groups):
    """Return material, measured diameter (in), avg_flow (LPM), avg_delta_p (Pa),
    and temp (C) per group. Temp is parsed from the logging note; flow/pressure
    readings that are zero or negative are dropped before averaging."""
    rows = []
    for (material, nominal_diameter, note), readings in groups:
        readings = [(flow, psid) for flow, psid in readings if flow > 0 and psid > 0]
        if not readings:
            continue
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        temp = float(TEMP_PATTERN.search(note).group(1))
        diameter_in = PIPE_SPECS[(material, nominal_diameter)]["diameter_in"]
        rows.append((material, diameter_in, avg_flow, avg_dp, temp))
    return sorted(rows)

def reynolds_number(diameter_in, flow_rate_lpm, rho, mu):
    """Reynolds number for a pipe (measured diameter in inches) at a given flow rate (LPM)."""
    d = diameter_in * 0.0254
    area = math.pi / 4 * d**2
    q = flow_rate_lpm / 60000  # LPM -> m^3/s
    v = q / area
    return rho * v * d / mu

def friction_factor(diameter_in, flow_rate_lpm, delta_p, rho):
    """Fanning friction factor for a pipe (measured diameter in inches), flow rate
    (LPM), and pressure drop (Pa)."""
    d = diameter_in * 0.0254
    length = SPECS_BY_DIAMETER[diameter_in]["length_in"] * 0.0254
    area = math.pi / 4 * d**2
    q = flow_rate_lpm / 60000  # LPM -> m^3/s
    v = q / area
    return delta_p * d / (length * 2 * rho * v**2)

def haaland_fanning(re, eps_over_d):
    """Fanning friction factor from the Haaland correlation for a given Reynolds
    number and relative roughness (eps/D)."""
    f_darcy = (-1.8 * math.log10((eps_over_d / 3.7)**1.11 + 6.9 / re))**-2
    return f_darcy / 4

def churchill_fanning(re, eps_over_d):
    """Fanning friction factor from the Churchill correlation (valid across the
    laminar, transitional, and turbulent regimes) for a given Reynolds number
    and relative roughness (eps/D)."""
    a = (2.457 * math.log(1 / ((7 / re)**0.9 + 0.27 * eps_over_d)))**16
    b = (37530 / re)**16
    return 2 * ((8 / re)**12 + (a + b)**-1.5)**(1 / 12)

def romeo_royo_monzon_fanning(re, eps_over_d):
    """Fanning friction factor from the Romeo, Royo & Monzon correlation for a
    given Reynolds number and relative roughness (eps/D)."""
    inner = math.log10((eps_over_d / 7.7918)**0.9924 + (5.3326 / re)**0.9345)
    mid = math.log10((eps_over_d / 3.827) - (4.567 / re) * inner)
    outer = math.log10((eps_over_d / 3.7065) - (5.0272 / re) * mid)
    return (-4 * outer)**-2

def friction_reynolds(rows, rho):
    """Compute Reynolds number and friction factor for each averaged group.
    `rho` (kg/m^3) is the fixed density used for the whole dataset; viscosity is
    recalculated per point (temperature per run) from the Kestin correlation."""
    results = []
    for material, diameter_in, avg_flow, avg_dp, temp in rows:
        mu = viscosity_kestin(temp)
        re = reynolds_number(diameter_in, avg_flow, rho, mu)
        f = friction_factor(diameter_in, avg_flow, avg_dp, rho)
        results.append((material, diameter_in, re, f))
    return results

def pipes_present(rows):
    """Distinct (material, diameter_in) pairs in a rows/results list, sorted for
    stable plotting order."""
    return sorted(set((r[0], r[1]) for r in rows))

def create_flow_vs_pressure_plot(rows, title, filename):
    """Plot pressure drop vs. flow rate by pipe, with a power-law fit
    (deltaP = a * Q^n, fit separately per pipe) overlaid on each series."""
    fig, ax = plt.subplots(figsize=(8, 6))

    fit_info = []
    for material, diameter_in in pipes_present(rows):
        subset = [r for r in rows if r[0] == material and r[1] == diameter_in]
        avg_flow = [r[2] for r in subset]
        avg_dp = [r[3] for r in subset]
        ax.plot(avg_flow, avg_dp, PIPE_MARKERS[diameter_in], label=pipe_label(material, diameter_in),
                color="black", markerfacecolor="black")

        a, n = power_law_fit(avg_flow, avg_dp)
        q_min, q_max = min(avg_flow), max(avg_flow)
        q_curve = [q_min * (q_max / q_min)**(i / 99) for i in range(100)]
        dp_curve = [a * qc**n for qc in q_curve]
        ax.plot(q_curve, dp_curve, '--', color="black")

        predicted = [a * qi**n for qi in avg_flow]
        mean_dp = statistics.mean(avg_dp)
        ss_res = sum((m - p)**2 for m, p in zip(avg_dp, predicted))
        ss_tot = sum((m - mean_dp)**2 for m in avg_dp)
        r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")
        fit_info.append((material, diameter_in, n, r_squared))

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
    """Plot globe-valve pressure drop vs. flow rate, grouped by valve opening."""
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

def create_valve_resistance_by_opening_plot(rows, title, filename):
    """Plot resistance coefficient K vs. Reynolds number, grouped by valve opening."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for opening in ALL_VALVE_OPENINGS:
        subset = [r for r in rows if r[0] == opening]
        if not subset:
            continue
        re = [r[1] for r in subset]
        k = [r[2] for r in subset]
        ax.plot(re, k, ALL_VALVE_OPENING_MARKERS[opening], label=opening,
                color="black", markerfacecolor="black")

    ax.set_xscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Resistance Coefficient K")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def create_equivalent_length_plot(rows, title, filename):
    """Plot equivalent length (in pipe diameters) vs. Reynolds number, grouped by
    valve opening."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for opening in ALL_VALVE_OPENINGS:
        subset = [r for r in rows if r[0] == opening]
        if not subset:
            continue
        re = [r[1] for r in subset]
        leq = [r[2] for r in subset]
        ax.plot(re, leq, ALL_VALVE_OPENING_MARKERS[opening], label=opening,
                color="black", markerfacecolor="black")

    ax.set_xscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Equivalent Length, Leq/D")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
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

    for material, diameter_in in pipes_present(results):
        subset = [r for r in results if r[0] == material and r[1] == diameter_in]
        re = [r[2] for r in subset]
        f = [r[3] for r in subset]
        ax.plot(re, f, PIPE_MARKERS[diameter_in], label=pipe_label(material, diameter_in),
                color="black", markerfacecolor="black")

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

    ax.legend(loc="lower left")
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def create_valve_resistance_plot(rows, k_reference, filename="half_open_valve_resistance_vs_reynolds.png"):
    """Plot resistance coefficient K vs. Reynolds number for the half-open valve,
    compared to a constant reference K value."""
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
    """Least-squares fit of y = a * x^n via linear regression in log-log space.
    Returns (a, n)."""
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
    """Fit deltaP = a * Q^n (power law) separately for each pipe diameter present
    in `rows` (the proportionality constant a depends on diameter, so diameters
    are not pooled into one fit), plot the measured points and fitted curves on
    log-log axes, and report each pipe's fitted exponent n alongside R^2."""
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
    """Compare friction factor data within a Re range to a theoretical correlation,
    across all pipes present in `results`."""
    subset_all = [r for r in results if regime_filter(r[2])]
    if not subset_all:
        print(f"No data in range for {filename}.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for material, diameter_in in pipes_present(subset_all):
        subset = [r for r in subset_all if r[0] == material and r[1] == diameter_in]
        re = [r[2] for r in subset]
        f = [r[3] for r in subset]
        ax.plot(re, f, PIPE_MARKERS[diameter_in], label=f'{pipe_label(material, diameter_in)} (measured)',
                color="black", markerfacecolor="black")

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
    ax.legend()
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename} (R^2 = {r_squared:.4f})")

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Old prelab-only analysis (power-law fits, globe valve, half-open valve):
    # unchanged data source and rho definition, left as-is other than the
    # flow/pressure filtering and the shared-helper signature updates above.
    # ------------------------------------------------------------------
    slowflow_groups = parse_flow_pressure_groups("FLU-prelab-Slowflow")
    highflow_groups = parse_flow_pressure_groups("FLU_Prelab_Highflow")

    rows = averages(slowflow_groups + highflow_groups)

    for material, diameter_in, avg_flow, avg_dp, temp in rows:
        print(f'{pipe_label(material, diameter_in)} pipe: Q={avg_flow:.2f} LPM, dP={avg_dp:.1f} Pa, T={temp:.2f} C')

    # density from the Kell correlation at the average temperature across all
    # prelab runs; viscosity is recalculated per point (temperature per run)
    # from the Kestin correlation
    avg_temp = statistics.mean(r[4] for r in rows)
    rho = density_kell(avg_temp)
    print(f"Average temp across prelab runs: {avg_temp:.2f} C -> rho (Kell) = {rho:.3f} kg/m^3")

    friction_results = friction_reynolds(rows, rho)
    for material, diameter_in, re, f in friction_results:
        print(f'{pipe_label(material, diameter_in)} pipe: Re={re:.0f}, f={f:.4f}')

    # (diameter, avg_flow, avg_dp) split by flow regime, using each row's own Reynolds
    # number from friction_results (rows and friction_results share index order)
    laminar_flow_rows = [(d, q, dp) for (_, d, q, dp, _), (_, _, re, _) in zip(rows, friction_results) if re < LAMINAR_RE]
    turbulent_flow_rows = [(d, q, dp) for (_, d, q, dp, _), (_, _, re, _) in zip(rows, friction_results) if re > TURBULENT_RE]

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

    # ------------------------------------------------------------------
    # New combined prelab + lab analysis (Figures 1-4 and the correlation
    # comparisons), covering all 5 pipes (3 PVC, steel, copper).
    # ------------------------------------------------------------------
    all_groups = (
        parse_flow_pressure_groups("FLU-prelab-Slowflow")
        + parse_flow_pressure_groups("FLU_Prelab_Highflow")
        + parse_flow_pressure_groups("FLU-lab-Slowflow")
        + parse_flow_pressure_groups("FLU_Lab_Highflow"))
    all_rows = averages(all_groups)

    for material, diameter_in, avg_flow, avg_dp, temp in all_rows:
        print(f'{pipe_label(material, diameter_in)} pipe: Q={avg_flow:.2f} LPM, dP={avg_dp:.1f} Pa, T={temp:.2f} C')

    # density from the Kell correlation at the average temperature over the lab
    # day only (FLU-lab-Slowflow / FLU_Lab_Highflow); viscosity is recalculated
    # per point (temperature per run) from the Kestin correlation
    lab_day_rows = averages(
        parse_flow_pressure_groups("FLU-lab-Slowflow") + parse_flow_pressure_groups("FLU_Lab_Highflow"))
    avg_temp_lab_day = statistics.mean(r[4] for r in lab_day_rows)
    rho_all = density_kell(avg_temp_lab_day)
    print(f"Average temp over the lab day: {avg_temp_lab_day:.2f} C -> rho (Kell) = {rho_all:.3f} kg/m^3")

    friction_results_all = friction_reynolds(all_rows, rho_all)
    for material, diameter_in, re, f in friction_results_all:
        print(f'{pipe_label(material, diameter_in)} pipe: Re={re:.0f}, f={f:.4f}')

    pvc_diameters = {spec["diameter_in"] for (m, _), spec in PIPE_SPECS.items() if m == "pvc"}
    small_diameters = {PIPE_SPECS[("pvc", 0.285)]["diameter_in"],
                        PIPE_SPECS[("steel", 0.31)]["diameter_in"],
                        PIPE_SPECS[("copper", 0.31)]["diameter_in"]}

    pvc_flow_rows = [r for r in all_rows if r[1] in pvc_diameters]
    small_flow_rows = [r for r in all_rows if r[1] in small_diameters]
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
    # curves are visually nearly identical; use the mean eps/D across all pipes
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

    create_valve_resistance_by_opening_plot(
        valve_k_rows, "Resistance Coefficient vs. Reynolds Number (All Valve Openings)",
        "Figure_11.png")

    # deltaP/rho = 4f * Leq * (v^2/2) = K * (v^2/2), so Leq (in pipe diameters) = K / (4f);
    # f is the Fanning friction factor at the valve's own Re, from the Churchill
    # correlation using the same representative smooth-pipe eps/D as Figures 5-9
    valve_leq_rows = [(opening, re, k / (4 * churchill_fanning(re, mean_eps_over_d)))
                       for opening, re, k in valve_k_rows]
    for opening, re, leq in valve_leq_rows:
        print(f'{opening} valve: Re={re:.0f}, Leq/D={leq:.1f}')

    create_equivalent_length_plot(
        valve_leq_rows, "Equivalent Length vs. Reynolds Number (All Valve Openings)",
        "Figure_12.png")
