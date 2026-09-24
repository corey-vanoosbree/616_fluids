import re
import statistics
import math
import matplotlib.pyplot as plt

PSI_TO_PA = 6894.757293168
DIAMETER_PATTERN = re.compile(r'(\d+\.\d+)\s*(?:"|inches)')

pipes = [("pvc", 0.408, 76, 4), ("pvc", 0.282, 70.5, 4), ("pvc", 0.47, 66, 4), ("steel", 0.31, 79.5, 8), ("copper", 0.312, 72.5, 8)]
rho = 997 #kg/m^3
mu = 9.25 * 10**-4 #Pa.s, average of measured values at 20-25C

# nominal diameter as labeled in the logging notes -> measured pipe diameter/length from `pipes`
PIPE_SPECS = {
    0.41: {"diameter_in": 0.408, "length_in": 76},
    0.285: {"diameter_in": 0.282, "length_in": 70.5},
}

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

def averages_with_error(groups):
    """Return diameter, avg_flow, std_flow, avg_delta_p (Pa), std_delta_p (Pa) per group."""
    rows = []
    for (diameter, note), readings in groups:
        flows = [flow for flow, _ in readings]
        pressures_pa = [psid * PSI_TO_PA for _, psid in readings]
        avg_flow = statistics.mean(flows)
        avg_dp = statistics.mean(pressures_pa)
        std_flow = statistics.pstdev(flows)
        std_dp = statistics.pstdev(pressures_pa)
        rows.append((diameter, avg_flow, std_flow, avg_dp, std_dp))
    return sorted(rows)

def reynolds_number(diameter, flow_rate_lpm):
    """Reynolds number for a pipe (nominal diameter in inches) at a given flow rate (LPM)."""
    spec = PIPE_SPECS[diameter]
    d = spec["diameter_in"] * 0.0254
    area = math.pi / 4 * d**2
    q = flow_rate_lpm / 60000  # LPM -> m^3/s
    v = q / area
    return rho * v * d / mu

def friction_factor(diameter, flow_rate_lpm, delta_p):
    """Fanning friction factor for a pipe (nominal diameter in inches), flow rate (LPM),
    and pressure drop (Pa)."""
    spec = PIPE_SPECS[diameter]
    d = spec["diameter_in"] * 0.0254
    length = spec["length_in"] * 0.0254
    area = math.pi / 4 * d**2
    q = flow_rate_lpm / 60000  # LPM -> m^3/s
    v = q / area
    return delta_p * d / (length * 2 * rho * v**2)

def friction_reynolds(rows):
    """Compute Reynolds number and friction factor, with error bars, for each averaged group."""
    results = []
    for diameter, avg_flow, std_flow, avg_dp, std_dp in rows:
        re = reynolds_number(diameter, avg_flow)
        f = friction_factor(diameter, avg_flow, avg_dp)

        rel_flow = std_flow / avg_flow if avg_flow else 0
        rel_dp = std_dp / avg_dp if avg_dp else 0
        std_re = re * rel_flow
        std_f = f * math.sqrt(rel_dp**2 + (2 * rel_flow)**2)

        results.append((diameter, re, std_re, f, std_f))
    return results

def create_plot(rows, filename="flow_vs_pressure.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    diameters = sorted(set(r[0] for r in rows))
    colors = {diameters[0]: "tab:blue", diameters[1]: "tab:orange"} if len(diameters) == 2 \
        else {d: c for d, c in zip(diameters, plt.cm.tab10.colors)}

    for diameter in diameters:
        subset = [r for r in rows if r[0] == diameter]
        avg_flow = [r[1] for r in subset]
        avg_dp = [r[3] for r in subset]
        color = colors[diameter]
        ax.plot(avg_flow, avg_dp, 'o', label=f'{diameter}" pipe', color=color, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow Rate (LPM)")
    ax.set_ylabel("Pressure Drop (Pa)")
    ax.set_title("Pressure Drop vs. Flow Rate by Pipe Diameter")
    ax.legend()
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

LAMINAR_RE = 2300
TURBULENT_RE = 4000

def create_friction_plot(results, filename="friction_factor_vs_reynolds.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    diameters = sorted(set(r[0] for r in results))
    colors = {diameters[0]: "tab:blue", diameters[1]: "tab:orange"} if len(diameters) == 2 \
        else {d: c for d, c in zip(diameters, plt.cm.tab10.colors)}

    re_min = min(r[1] for r in results)
    re_max = max(r[1] for r in results)
    ax.axvspan(re_min, LAMINAR_RE, color="tab:green", alpha=0.08)
    ax.axvspan(LAMINAR_RE, TURBULENT_RE, color="tab:gray", alpha=0.12)
    ax.axvspan(TURBULENT_RE, re_max, color="tab:red", alpha=0.06)
    ax.axvline(LAMINAR_RE, color="gray", linestyle="--", linewidth=1)
    ax.axvline(TURBULENT_RE, color="gray", linestyle="--", linewidth=1)

    for diameter in diameters:
        subset = [r for r in results if r[0] == diameter]
        re = [r[1] for r in subset]
        f = [r[3] for r in subset]
        color = colors[diameter]
        ax.plot(re, f, 'o', label=f'{diameter}" pipe', color=color, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Fanning Friction Factor")
    ax.set_title("Friction Factor vs. Reynolds Number by Pipe Diameter")

    for label, x in [("Laminar", (re_min * LAMINAR_RE)**0.5),
                      ("Transitional", (LAMINAR_RE * TURBULENT_RE)**0.5),
                      ("Turbulent", (TURBULENT_RE * re_max)**0.5)]:
        ax.text(x, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="dimgray")

    ax.legend(loc="lower left")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def create_laminar_comparison_plot(results, filename="laminar_friction_comparison.png"):
    """Compare the laminar-regime (Re < 2300) data to the f = 16/Re correlation
    for the Fanning friction factor."""
    laminar = [r for r in results if r[1] < LAMINAR_RE]
    if not laminar:
        print("No laminar-regime data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    diameters = sorted(set(r[0] for r in laminar))
    colors = {diameters[0]: "tab:blue", diameters[1]: "tab:orange"} if len(diameters) == 2 \
        else {d: c for d, c in zip(diameters, plt.cm.tab10.colors)}

    for diameter in diameters:
        subset = [r for r in laminar if r[0] == diameter]
        re = [r[1] for r in subset]
        f = [r[3] for r in subset]
        color = colors[diameter]
        ax.plot(re, f, 'o', label=f'{diameter}" pipe (measured)', color=color, alpha=0.8)

    re_min = min(r[1] for r in laminar)
    re_max = max(r[1] for r in laminar)
    re_curve = [re_min * (re_max / re_min)**(i / 99) for i in range(100)]
    f_curve = [16 / re for re in re_curve]
    ax.plot(re_curve, f_curve, '--', color="black", label="f = 16/Re")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Fanning Friction Factor")
    ax.set_title("Laminar Friction Factor vs. f = 16/Re Correlation")
    ax.legend()
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {filename}")

slowflow_groups = parse_flow_pressure_groups("FLU-prelab-Slowflow")
highflow_groups = parse_flow_pressure_groups("FLU_Prelab_Highflow")

rows = averages_with_error(slowflow_groups + highflow_groups)

for diameter, avg_flow, std_flow, avg_dp, std_dp in rows:
    print(f'{diameter}" pipe: Q={avg_flow:.2f}+/-{std_flow:.2f} LPM, dP={avg_dp:.1f}+/-{std_dp:.1f} Pa')

create_plot(rows)

friction_results = friction_reynolds(rows)
for diameter, re, std_re, f, std_f in friction_results:
    print(f'{diameter}" pipe: Re={re:.0f}+/-{std_re:.0f}, f={f:.4f}+/-{std_f:.4f}')

create_friction_plot(friction_results)
create_laminar_comparison_plot(friction_results)
