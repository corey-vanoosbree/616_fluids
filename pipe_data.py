import matplotlib.pyplot as plt
from plot_style import style_table

pipes = [("pvc", 0.408, 76, 4), ("pvc", 0.282, 70.5, 4), ("pvc", 0.47, 66, 4), ("steel", 0.31, 79.5, 8), ("copper", 0.312, 72.5, 8)]
#material, diameter in inches, length in inches, roughness in microinches

def save_table_png(rows, filename="pipe_data.png"):
    col_labels = ["Material", "Diameter (in)", "Length (in)", "Roughness (microin)"]
    cell_text = [[material, f"{diameter:.3f}", f"{length:.2f}", f"{roughness:.0f}"]
                 for material, diameter, length, roughness in rows]

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

save_table_png(pipes)
