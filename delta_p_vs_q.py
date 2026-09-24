import re
import statistics
import matplotlib.pyplot as plt

PSI_TO_PA = 6894.757293168
DIAMETER_PATTERN = re.compile(r'(\d+\.\d+)\s*(?:"|inches)')

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

def create_plot(rows, filename="flow_vs_pressure.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    diameters = sorted(set(r[0] for r in rows))
    colors = {diameters[0]: "tab:blue", diameters[1]: "tab:orange"} if len(diameters) == 2 \
        else {d: c for d, c in zip(diameters, plt.cm.tab10.colors)}

    for diameter in diameters:
        subset = [r for r in rows if r[0] == diameter]
        avg_flow = [r[1] for r in subset]
        std_flow = [r[2] for r in subset]
        avg_dp = [r[3] for r in subset]
        std_dp = [r[4] for r in subset]
        color = colors[diameter]
        ax.errorbar(avg_flow, avg_dp, xerr=std_flow, yerr=std_dp, fmt='o',
                    capsize=4, label=f'{diameter}" pipe', color=color, ecolor=color, alpha=0.8)

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

slowflow_groups = parse_flow_pressure_groups("FLU-prelab-Slowflow")
highflow_groups = parse_flow_pressure_groups("FLU_Prelab_Highflow")

rows = averages_with_error(slowflow_groups + highflow_groups)

for diameter, avg_flow, std_flow, avg_dp, std_dp in rows:
    print(f'{diameter}" pipe: Q={avg_flow:.2f}+/-{std_flow:.2f} LPM, dP={avg_dp:.1f}+/-{std_dp:.1f} Pa')

create_plot(rows)
