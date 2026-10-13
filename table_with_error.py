import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from plot_style import style_table

pipes_raw = [("pvc", [0.410, 0.410, 0.409, 0.409, 0.4085], 76, 4), ("pvc", [0.2855, 0.2855, 0.2855, 0.2855, 0.2845], 70.5, 4), ("pvc", [0.4695, 0.4715, 0.471, 0.469, 0.4735], 66, 4), ("steel", [0.307, 0.306, 0.3055, 0.3055, 0.3075], 79.5, 8), ("copper", [0.311, 0.307, 0.312, 0.3095, 0.3125], 72.5, 6)]
#material, diameter in inches, length in inches, roughness in microinches

def diameter_error(diameters, caliper_acc=0.001, confidence=0.95):
    """Combined uncertainty in mean pipe ID (in): U_C = sqrt(U_A^2 + U_B^2)."""
    d = np.asarray(diameters, dtype=float)
    n = d.size
    t = stats.t.ppf(1 - (1 - confidence) / 2, n - 1)  # t = 2.776 for n = 5
    U_A = t * d.std(ddof=1) / np.sqrt(n)  # random: scatter between angles
    U_B = caliper_acc                     # systematic: same caliper, so no /sqrt(n)
    return np.hypot(U_A, U_B)

pipes = [(material, np.mean(d), diameter_error(d), length, roughness)
         for material, d, length, roughness in pipes_raw]

def save_table_png(rows, filename="Table_2.png"):
    col_labels = ["Material", "Diameter (in)", "Length (in)", "Roughness (microin)"]
    cell_text = [[material, f"{diameter:.4f} ± {error:.4f}", f"{length:.2f}", f"{roughness:.0f}"]
                 for material, diameter, error, length, roughness in rows]

    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.35 * len(rows)))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    style_table(ax, table, "Pipe Data")
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table to {filename}")

if __name__ == "__main__":
    for material, diameter, error, length, roughness in pipes:
        print(f"{material:7s} D = {diameter:.4f} ± {error:.4f} in")
    save_table_png(pipes)
