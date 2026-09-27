import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

def style_table(ax, table, title):
    """Style a matplotlib table as a black-and-white, three-rule table: no fill
    colors, a bold header row, and only top/header/bottom rules instead of a
    full grid of cell borders."""
    cells = table.get_celld()
    n_rows = max(row for row, _ in cells) + 1

    for (row, _), cell in cells.items():
        cell.set_facecolor("white")
        cell.visible_edges = ""
        cell.set_text_props(ha="left")
        if row == 0:
            cell.set_text_props(weight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    row_height = 1 / n_rows
    for y in (1.0, 1.0 - row_height, 0.0):
        ax.axhline(y, color="black", linewidth=1.1, xmin=0, xmax=1)

    ax.text(0, 1.05, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold")
