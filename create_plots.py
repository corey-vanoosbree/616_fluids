import re
import statistics
import math
import matplotlib.pyplot as plt
import plot_style  # noqa: F401 (sets sans-serif font rcParams on import)

MARKERS = ["o", "^"]  # distinct marker per pipe diameter, black-and-white friendly

PSI_TO_PA = 6894.757293168
DIAMETER_PATTERN = re.compile(r'(\d+\.\d+)\s*(?:"|inches)')
TEMP_PATTERN = re.compile(r'LPM[^\d]*(\d+\.?\d*)')

pipes = [("pvc", 0.408, 76, 4), ("pvc", 0.282, 70.5, 4), ("pvc", 0.47, 66, 4), ("steel", 0.31, 79.5, 8), ("copper", 0.312, 72.5, 8)]

# nominal diameter as labeled in the logging notes -> measured pipe diameter/length from `pipes`
PIPE_SPECS = {
    0.41: {"diameter_in": 0.408, "length_in": 76},
    0.285: {"diameter_in": 0.282, "length_in": 70.5},
}
ROUGHNESS_IN = 4e-6  # pvc roughness, inches (4 microinches, from `pipes`)

def density_kell(t):
    """Kell (1975) equation for the density of water at 1 atm. t in C, returns kg/m^3."""
    numerator = (999.83952 + 16.945176 * t - 7.9870401e-3 * t**2
                 - 46.170461e-6 * t**3 + 105.56302e-9 * t**4 - 280.54253e-12 * t**5)
    denominator = 1 + 16.879850e-3 * t
    return numerator / denominator

def viscosity_kestin(t):
    """Kestin, Sokolov & Wakeham (1978) equation for the viscosity of water at 1 atm.
    t in C, returns dynamic viscosity in Pa.s."""
    mu20 = 1.002e-3  # Pa.s at 20C
    dt = 20 - t
    log_ratio = (dt * (1.2364 - 1.37e-3 * dt + 5.7e-6 * dt**2)) / (t + 96)
    return mu20 * 10**log_ratio

def parse_flow_pressure_groups(filename):
    """Group rows by logging note (one flow-rate setting) and return per-group
    pipe diameter plus flow rate and pressure drop readings, in file order."""
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
        diameter_match = DIAMETER_PATTERN.search(note)
        if not diameter_match:
            continue
        diameter = float(diameter_match.group(1))
        if diameter not in PIPE_SPECS:
            continue
        groups.setdefault((diameter, note), []).append((flow, pressure_psid))

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

def globe_valve_rows(groups, rho):
    """Return avg_flow (LPM), avg_delta_p (Pa), and friction loss as specific energy
    (J/kg) per group. `rho` (kg/m^3) is the fixed density used for the whole dataset."""
    rows = []
    for note, readings in groups:
        readings = [(flow, psid) for flow, psid in readings if psid > 0]
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        friction_loss = avg_dp / rho  # specific energy loss, J/kg
        rows.append((avg_flow, avg_dp, friction_loss))
    return sorted(rows)

def averages_with_error(groups):
    """Return diameter, avg_flow, std_flow, avg_delta_p (Pa), std_delta_p (Pa), temp (C)
    per group. Temp is parsed from the logging note."""
    rows = []
    for (diameter, note), readings in groups:
        readings = [(flow, psid) for flow, psid in readings if psid > 0]
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        std_flow = statistics.pstdev(flows)
        std_dp = statistics.pstdev(pressures_pa)
        temp = float(TEMP_PATTERN.search(note).group(1))
        rows.append((diameter, avg_flow, std_flow, avg_dp, std_dp, temp))
    return sorted(rows)

def reynolds_number(diameter, flow_rate_lpm, rho, mu):
    """Reynolds number for a pipe (nominal diameter in inches) at a given flow rate (LPM)."""
    spec = PIPE_SPECS[diameter]
    d = spec["diameter_in"] * 0.0254
    area = math.pi / 4 * d**2
    q = flow_rate_lpm / 60000  # LPM -> m^3/s
    v = q / area
    return rho * v * d / mu

def friction_factor(diameter, flow_rate_lpm, delta_p, rho):
    """Fanning friction factor for a pipe (nominal diameter in inches), flow rate (LPM),
    and pressure drop (Pa)."""
    spec = PIPE_SPECS[diameter]
    d = spec["diameter_in"] * 0.0254
    length = spec["length_in"] * 0.0254
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
    """Compute Reynolds number and friction factor, with error bars, for each averaged group.
    `rho` (kg/m^3) is the fixed density used for the whole dataset; viscosity is
    recalculated per point from the Kestin correlation using that point's own temperature."""
    results = []
    for diameter, avg_flow, std_flow, avg_dp, std_dp, temp in rows:
        mu = viscosity_kestin(temp)
        re = reynolds_number(diameter, avg_flow, rho, mu)
        f = friction_factor(diameter, avg_flow, avg_dp, rho)

        rel_flow = std_flow / avg_flow if avg_flow else 0
        rel_dp = std_dp / avg_dp if avg_dp else 0
        std_re = re * rel_flow
        std_f = f * math.sqrt(rel_dp**2 + (2 * rel_flow)**2)

        results.append((diameter, re, std_re, f, std_f))
    return results

def create_plot(rows, filename="flow_vs_pressure.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    diameters = sorted(set(r[0] for r in rows))
    markers = {d: m for d, m in zip(diameters, MARKERS)}

    for diameter in diameters:
        subset = [r for r in rows if r[0] == diameter]
        avg_flow = [r[1] for r in subset]
        avg_dp = [r[3] for r in subset]
        ax.plot(avg_flow, avg_dp, markers[diameter], label=f'{diameter}" pipe',
                color="black", markerfacecolor="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow Rate (LPM)")
    ax.set_ylabel("Pressure Drop (Pa)")
    ax.set_title("Pressure Drop vs. Flow Rate by Pipe Diameter")
    ax.legend()
    ax.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

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

LAMINAR_RE = 2300
TURBULENT_RE = 4000

def create_friction_plot(results, filename="friction_factor_vs_reynolds.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    diameters = sorted(set(r[0] for r in results))
    markers = {d: m for d, m in zip(diameters, MARKERS)}

    re_min = min(r[1] for r in results)
    re_max = max(r[1] for r in results)
    ax.axvspan(re_min, LAMINAR_RE, color="black", alpha=0.06)
    ax.axvspan(LAMINAR_RE, TURBULENT_RE, color="black", alpha=0.16)
    ax.axvspan(TURBULENT_RE, re_max, color="black", alpha=0.0)
    ax.axvline(LAMINAR_RE, color="black", linestyle="--", linewidth=1)
    ax.axvline(TURBULENT_RE, color="black", linestyle="--", linewidth=1)

    for diameter in diameters:
        subset = [r for r in results if r[0] == diameter]
        re = [r[1] for r in subset]
        f = [r[3] for r in subset]
        ax.plot(re, f, markers[diameter], label=f'{diameter}" pipe',
                color="black", markerfacecolor="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Fanning Friction Factor")
    ax.set_title("Friction Factor vs. Reynolds Number by Pipe Diameter")

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

def create_regime_comparison_plot(results, regime_filter, correlation, correlation_label, title, filename):
    """Compare friction factor data within a Re range to a theoretical correlation."""
    subset_all = [r for r in results if regime_filter(r[1])]
    if not subset_all:
        print(f"No data in range for {filename}.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    diameters = sorted(set(r[0] for r in subset_all))
    markers = {d: m for d, m in zip(diameters, MARKERS)}

    for diameter in diameters:
        subset = [r for r in subset_all if r[0] == diameter]
        re = [r[1] for r in subset]
        f = [r[3] for r in subset]
        ax.plot(re, f, markers[diameter], label=f'{diameter}" pipe (measured)',
                color="black", markerfacecolor="black")

    re_min = min(r[1] for r in subset_all)
    re_max = max(r[1] for r in subset_all)
    re_curve = [re_min * (re_max / re_min)**(i / 99) for i in range(100)]
    f_curve = [correlation(re) for re in re_curve]
    ax.plot(re_curve, f_curve, '--', color="black", label=correlation_label)

    measured_f = [r[3] for r in subset_all]
    predicted_f = [correlation(r[1]) for r in subset_all]
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

slowflow_groups = parse_flow_pressure_groups("FLU-prelab-Slowflow")
highflow_groups = parse_flow_pressure_groups("FLU_Prelab_Highflow")

rows = averages_with_error(slowflow_groups + highflow_groups)

for diameter, avg_flow, std_flow, avg_dp, std_dp, temp in rows:
    print(f'{diameter}" pipe: Q={avg_flow:.2f}+/-{std_flow:.2f} LPM, dP={avg_dp:.1f}+/-{std_dp:.1f} Pa, T={temp:.2f} C')

create_plot(rows)

# density from the Kell correlation at the average temperature across all runs;
# viscosity is recalculated per point (in friction_reynolds) from the Kestin
# correlation using that point's own temperature
avg_temp = statistics.mean(r[5] for r in rows)
rho = density_kell(avg_temp)
print(f"Average temp across all runs: {avg_temp:.2f} C -> rho (Kell) = {rho:.3f} kg/m^3")

friction_results = friction_reynolds(rows, rho)
for diameter, re, std_re, f, std_f in friction_results:
    print(f'{diameter}" pipe: Re={re:.0f}+/-{std_re:.0f}, f={f:.4f}+/-{std_f:.4f}')

create_friction_plot(friction_results)

create_regime_comparison_plot(
    friction_results, lambda re: re < LAMINAR_RE, lambda re: 16 / re,
    "f = 16/Re", "Laminar Friction Factor vs. f = 16/Re Correlation",
    "laminar_friction_comparison.png")

create_regime_comparison_plot(
    friction_results, lambda re: re > TURBULENT_RE, lambda re: 0.079 * re**-0.25,
    "f = 0.079 Re^-0.25 (Blasius)", "Turbulent Friction Factor vs. Blasius Correlation",
    "turbulent_friction_comparison.png")

# both pipes are hydraulically smooth (eps/D ~1e-5), so their Haaland curves are
# visually identical; use the mean diameter for a single representative curve
mean_d_in = statistics.mean(spec["diameter_in"] for spec in PIPE_SPECS.values())
eps_over_d = ROUGHNESS_IN / mean_d_in
create_regime_comparison_plot(
    friction_results, lambda re: re > TURBULENT_RE, lambda re: haaland_fanning(re, eps_over_d),
    "Haaland correlation", "Turbulent Friction Factor vs. Haaland Correlation",
    "turbulent_friction_haaland.png")

create_regime_comparison_plot(
    friction_results, lambda re: True, lambda re: churchill_fanning(re, eps_over_d),
    "Churchill correlation", "Friction Factor vs. Churchill Correlation (All Flow Regimes)",
    "churchill_friction_comparison.png")

create_regime_comparison_plot(
    friction_results, lambda re: re > TURBULENT_RE, lambda re: romeo_royo_monzon_fanning(re, eps_over_d),
    "Romeo, Royo & Monzon correlation", "Turbulent Friction Factor vs. Romeo, Royo & Monzon Correlation",
    "turbulent_friction_romeo_royo_monzon.png")

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
