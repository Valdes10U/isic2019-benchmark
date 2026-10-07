"""Estilo común de figuras: orden, nombres, colores, marcadores y tramas de cada modelo."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ["resnet50", "efficientnet_b0", "swin_t"]
NAMES = {"resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0", "swin_t": "Swin-T"}
# luminancias L* 30, 45 y 59: se distinguen en escala de grises; las tres con contraste >= 3:1 sobre blanco
COLORS = {"resnet50": "#1A4290", "efficientnet_b0": "#C0392B", "swin_t": "#B8860B"}
MARKERS = {"resnet50": "o", "efficientnet_b0": "s", "swin_t": "^"}
HATCHES = {"resnet50": "", "efficientnet_b0": "////", "swin_t": "...."}
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
ONE_COL, TWO_COL = 3.5, 7.16  # pulgadas, formato IEEE


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
        "lines.markersize": 4, "hatch.linewidth": 0.5, "hatch.color": "white", "axes.grid": True,
        "grid.color": "#dddddd", "grid.linewidth": 0.4, "axes.axisbelow": True,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def save(fig, path):
    """Guarda en PDF y PNG a 300 dpi; path sin extensión."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
