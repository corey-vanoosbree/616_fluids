import math
import matplotlib.pyplot as plt

mu = 1.002 * 10**-3 #Pa.s
rho = 997 #kg/m^3
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

    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.4 * len(rows)))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")

    ax.set_title("Flow Regime Transition Rates", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table to {filename}")

rows = flow_regime()
save_table_png(rows)