import re
import matplotlib.pyplot as plt

NOTE_PATTERN = re.compile(r'([\d.]+)\s*LPM\s*\(?\s*(\d+\.?\d*)\.?\s*C?\)?')

def parse_flow_temps(filename):
    """Return one (flow_rate, temp_c) pair per distinct logging note, in file order."""
    rows = []
    seen_notes = set()
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines[4:]:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 4:
            continue
        note = fields[3].strip()
        if not note or note in seen_notes:
            continue
        match = NOTE_PATTERN.search(note)
        if not match:
            continue
        seen_notes.add(note)
        flow = float(match.group(1))
        temp = float(match.group(2))
        rows.append((flow, temp))

    return rows

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
    mu = mu20 * 10**log_ratio
    return mu

def build_rows(source, flow_temps):
    rows = []
    for flow, temp in flow_temps:
        rho = density_kell(temp)
        mu = viscosity_kestin(temp)
        rows.append((source, flow, temp, rho, mu))
    return rows

def save_density_table(rows, filename="water_density_kell.png"):
    col_labels = ["Data Set", "Flow Rate (LPM)", "Temp (C)", "Density (kg/m^3)"]
    cell_text = [[source, f"{flow:.2f}", f"{temp:.2f}", f"{rho:.3f}"]
                 for source, flow, temp, rho, mu in rows]

    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.28 * len(rows)))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")

    ax.set_title("Water Density (Kell Equation)", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table to {filename}")

def save_viscosity_table(rows, filename="water_viscosity_kestin.png"):
    col_labels = ["Data Set", "Flow Rate (LPM)", "Temp (C)", "Viscosity (Pa.s)"]
    cell_text = [[source, f"{flow:.2f}", f"{temp:.2f}", f"{mu:.5e}"]
                 for source, flow, temp, rho, mu in rows]

    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.28 * len(rows)))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")

    ax.set_title("Water Viscosity (Kestin Equation)", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table to {filename}")

slowflow = parse_flow_temps("FLU-prelab-Slowflow")
highflow = parse_flow_temps("FLU_Prelab_Highflow")

rows = build_rows("Slowflow", slowflow) + build_rows("Highflow", highflow)

for source, flow, temp, rho, mu in rows:
    print(f"{source}: {flow:.2f} LPM, {temp:.2f} C -> rho={rho:.3f} kg/m^3, mu={mu:.5e} Pa.s")

save_density_table(rows)
save_viscosity_table(rows)
