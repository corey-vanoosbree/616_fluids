import math
import statistics
import matplotlib.pyplot as plt
from plot_style import style_table
from create_plots import (
    parse_flow_pressure_groups, averages, density_kell, viscosity_kestin,
)

# average density and viscosity from the collected flow/pressure data, evaluated
# at the average temperature across all runs (same approach as create_plots.py)
_data_rows = averages(
    parse_flow_pressure_groups("FLU-prelab-Slowflow")
    + parse_flow_pressure_groups("FLU_Prelab_Highflow"))
_avg_temp = statistics.mean(r[4] for r in _data_rows)
rho = density_kell(_avg_temp) #kg/m^3
mu = viscosity_kestin(_avg_temp) #Pa.s
print(f"Average temp across all runs: {_avg_temp:.2f} C -> rho = {rho:.3f} kg/m^3, mu = {mu:.6e} Pa.s")

pipes = [("pvc", 0.408), ("pvc", 0.282), ("pvc", 0.47), ("steel", 0.31), ("copper", 0.312)] #diameter in inches
turbulent = 4000
laminar = 2300

def flow_regime():
    rows = []
    for material, diameter in pipes:
        d = diameter * 0.0254 # convert inches to meters
        Q_turb = turbulent * math.pi / 4 * mu * d / rho * 1000 * 60 #L/min
        Q_lam = laminar * math.pi / 4 * mu * d / rho * 1000 * 60 #L/min
        print(f'{material} diameter: {diameter} inches, Q_turbulent: {Q_turb:.2f} L/min, Q_laminar: {Q_lam:.2f} L/min')
        rows.append((material, diameter, Q_turb, Q_lam))
    return rows

def save_table_png(rows, filename="flowrate_regimes.png"):
    col_labels = ["Material", "Diameter (in)", "Q_turbulent (L/min)", "Q_laminar (L/min)"]
    cell_text = [[material, f"{diameter:.3f}", f"{Q_turb:.2f}", f"{Q_lam:.2f}"]
                 for material, diameter, Q_turb, Q_lam in rows]

    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.35 * len(rows)))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    style_table(ax, table, "Flow Regime Transition Rates")
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table to {filename}")

rows = flow_regime()
save_table_png(rows)