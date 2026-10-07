"""Tablas (CSV y LaTeX) y figuras (PDF y PNG) de la Fase 3, calculadas desde runs/ y results/.

Uso: uv run python scripts/make_figures.py
Requiere haber corrido isic.cost, isic.evaluate --split train, isic.ensemble e isic.gradcam.
Escribe results/tables/, figures/ y results/summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from isic.plotstyle import (CLASSES, COLORS, HATCHES, MARKERS, MODELS, NAMES, ONE_COL, TWO_COL, plt, save,
                            setup)

RUNS, RES, FIG = Path("runs"), Path("results"), Path("figures")
TAB = RES / "tables"
SEEDS = [0, 1, 2]
MINOR = ["DF", "VASC", "SCC", "AK"]
WEIGHTS = {"inverse": ("", "Frecuencia inversa"), "sqrt_inverse": ("_sqrtw", "Raíz cuadrada"),
           "none": ("_now", "Sin pesos")}
SPLIT_NAMES = {"train": "Train", "val": "Val", "test": "Test interno", "test_official": "Test oficial"}
SD_NOTE = "Media ± desviación estándar muestral entre las semillas 0, 1 y 2."


def load(run):
    return json.loads((RUNS / run / "metrics.json").read_text())


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def pct(m, s=None):
    return f"{100 * m:.1f}" if s is None else f"{100 * m:.1f} ± {100 * s:.1f}"


def num(m, s=None, d=1):
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f} ± {s:.{d}f}"


def prec_rec(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    return tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)


def tex(s):
    s = str(s)
    for a, b in [("\\", "\\textbackslash{}"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("&", "\\&")]:
        s = s.replace(a, b)
    return s.replace("±", "$\\pm$").replace("Δ", "$\\Delta$")


def write_table(name, df, caption, notes, groups=None, bold=(), wide=False, colsep=None):
    """CSV con los mismos textos y LaTeX con booktabs; groups = [(etiqueta, n columnas)] sobre el encabezado."""
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    env = "table*" if wide else "table"
    ncol = df.shape[1]
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{tex(caption)}}}", f"\\label{{tab:{name}}}",
             "\\footnotesize"]
    if colsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{colsep}}}")
    lines += [f"\\begin{{tabular}}{{l{'c' * (ncol - 1)}}}", "\\toprule"]
    if groups:
        cells, rules, col = [], [], 1
        for label, span in groups:
            cells.append(f"\\multicolumn{{{span}}}{{c}}{{{tex(label)}}}" if label else
                         " & ".join([""] * span) if span > 1 else "")
            if label:
                rules.append(f"\\cmidrule(lr){{{col + 1}-{col + span}}}")
            col += span
        lines += [" & ".join([""] + cells) + " \\\\", " ".join(rules)]
    lines.append(" & ".join(tex(c) for c in df.columns) + " \\\\")
    lines.append("\\midrule")
    for i, row in enumerate(df.itertuples(index=False)):
        cells = [f"\\textbf{{{tex(v)}}}" if (i, j) in bold else tex(v) for j, v in enumerate(row)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines += ["\\par\\vspace{2pt}", f"\\parbox{{\\linewidth}}{{\\scriptsize {tex(notes)}}}"]
    lines.append(f"\\end{{{env}}}")
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")


def lead(cell):
    return float(str(cell).split(" ")[0])


def best_cells(df, cols):
    """Índices (fila, columna) del mayor valor medio por columna, para marcarlo en negritas."""
    return {(max(range(len(df)), key=lambda i: lead(df.iloc[i, j])), j) for j in cols}


# ---------- datos ----------

def collect():
    S = {"runs": {}, "models": {}}
    for m in MODELS:
        agg = {}
        runs = [load(f"{m}_seed{s}") for s in SEEDS]
        for r in runs:
            S["runs"][r["run"]] = {k: r[k] for k in ["seed", "checkpoint_epoch", "training"]} | {
                sp: {k: v for k, v in r[sp].items() if k != "confusion_matrix"}
                for sp in ["train", "test", "test_official"]}
        for sp in ["train", "test", "test_official"]:
            agg[sp] = {k: ms([r[sp][k] for r in runs])
                       for k in ["accuracy", "f1_macro", "f1_weighted", "balanced_accuracy", "auroc_macro"]}
            agg[sp]["f1_per_class"] = {c: ms([r[sp]["f1_per_class"][c] for r in runs]) for c in CLASSES}
        agg["val"] = {"f1_macro": ms([r["training"]["best_val_f1_macro"] for r in runs])}
        agg["best_epoch"] = ms([r["training"]["best_epoch"] for r in runs])
        S["models"][m] = agg
    return S


# ---------- tablas ----------

def table1(S):
    rows = []
    for m in MODELS:
        t, o = S["models"][m]["test"], S["models"][m]["test_official"]
        rows.append([NAMES[m]] + [pct(*t[k]) for k in ["f1_macro", "f1_weighted", "balanced_accuracy",
                                                        "accuracy", "auroc_macro"]]
                    + [pct(*o[k]) for k in ["f1_macro", "balanced_accuracy"]])
    cols = ["Modelo", "F1 macro", "F1 pond.", "Exact. bal.", "Exactitud", "AUROC", "F1 macro ", "Exact. bal. "]
    df = pd.DataFrame(rows, columns=cols)
    csv_df = df.copy()
    csv_df.columns = ["Modelo"] + [f"{c.strip()} (interno)" for c in cols[1:6]] + \
                     [f"{c.strip()} (oficial)" for c in cols[6:]]
    write_table("tabla1_resultados", df, "Resultados en el test interno y en el test oficial ISIC 2019 (sin UNK), en porcentaje.",
                f"{SD_NOTE} Exact. bal.: exactitud balanceada (recall macro). AUROC macro uno contra el resto. "
                "Test interno: 3,798 imágenes; test oficial: 6,191 imágenes (2,047 UNK excluidas). En negritas, la mejor media por columna.",
                groups=[("Test interno", 5), ("Test oficial", 2)], bold=best_cells(df, range(1, 8)), wide=True)
    csv_df.to_csv(TAB / "tabla1_resultados.csv", index=False)


def table2(S, cost):
    rows = []
    for m in MODELS:
        c = cost.loc[m]
        rows.append([NAMES[m], num(c["params_total"] / 1e6), num(c["gmacs"]), num(c["epoch_s_mean"], c["epoch_s_sd"]),
                     num(c["train_total_s_mean"] / 60, c["train_total_s_sd"] / 60), num(c["peak_vram_mb"] / 1024),
                     f"{c['throughput_img_s']:.0f}", num(*S["models"][m]["best_epoch"])])
    df = pd.DataFrame(rows, columns=["Modelo", "Parámetros (M)", "GMACs", "s/época", "Tiempo total (min)",
                                     "VRAM pico (GiB)", "Inferencia (img/s)", "Época del mejor F1 val"])
    write_table("tabla2_costo", df, "Costo computacional en una RTX 5070 Ti Laptop (12 GB), entrada de 224x224.",
                "GMACs: miles de millones de multiplicaciones-acumulaciones por imagen (fvcore); es lo que la literatura "
                "suele reportar como FLOPs (en FLOPs estrictos el valor es el doble). s/época: media de las épocas 2 en "
                "adelante (la época 1 incluye la calibración de cuDNN), entrenamiento más validación. Tiempo total: suma "
                f"de todas las épocas hasta el early stopping. {SD_NOTE} VRAM pico: máximo de torch.cuda.max_memory_allocated "
                "en entrenamiento con batch 32 y AMP bf16. Inferencia: batch 32 en bf16 con tensores sintéticos "
                "(10 batches de calentamiento y 50 medidos).", wide=True)


def table3(S, support):
    rows = [[c] + [pct(*S["models"][m]["test"]["f1_per_class"][c]) for m in MODELS] + [str(support[c])]
            for c in CLASSES]
    df = pd.DataFrame(rows, columns=["Clase"] + [NAMES[m] for m in MODELS] + ["Soporte"])
    write_table("tabla3_f1_por_clase", df, "F1 por clase en el test interno, en porcentaje.",
                f"{SD_NOTE} En negritas, el mejor modelo por clase. Soporte: imágenes de cada clase en el test interno. MEL: melanoma; NV: nevo melanocítico; "
                "BCC: carcinoma basocelular; AK: queratosis actínica; BKL: queratosis benigna; DF: dermatofibroma; "
                "VASC: lesión vascular; SCC: carcinoma escamocelular.",
                bold={(i, max(range(1, 4), key=lambda j: lead(df.iloc[i, j]))) for i in range(len(df))})


def table4(S, ens):
    rows = []
    for m in ["swin_t", "efficientnet_b0"]:
        t, o = S["models"][m]["test"], S["models"][m]["test_official"]
        rows.append([f"{NAMES[m]} individual (media)", pct(*t["f1_macro"]), pct(*t["balanced_accuracy"]),
                     pct(*o["f1_macro"]), pct(*o["balanced_accuracy"])])
    labels = {"arquitecturas_seed0": "Ensemble de arquitecturas (semilla 0)",
              "semillas_swin_t": "Ensemble de semillas Swin-T",
              "semillas_efficientnet_b0": "Ensemble de semillas EfficientNet-B0"}
    for k, label in labels.items():
        t, o = ens[k]["test"], ens[k]["test_official"]
        rows.append([label, pct(t["f1_macro"]), pct(t["balanced_accuracy"]), pct(o["f1_macro"]),
                     pct(o["balanced_accuracy"])])
    df = pd.DataFrame(rows, columns=["Modelo", "F1 macro", "Exact. bal.", "F1 macro ", "Exact. bal. "])
    write_table("tabla4_ensembles", df, "Ensembles por promedio de probabilidades softmax, en porcentaje.",
                "Ensemble de arquitecturas: ResNet50, EfficientNet-B0 y Swin-T con semilla 0. Ensembles de semillas: "
                "las semillas 0, 1 y 2 de una misma arquitectura (control). Ningún ensemble se entrena: se promedian las "
                "probabilidades guardadas. Las filas individuales son media ± desviación estándar entre semillas.",
                groups=[("Test interno", 2), ("Test oficial", 2)], bold=best_cells(df, range(1, 5)), wide=True)
    df.columns = ["Modelo", "F1 macro (interno)", "Exact. bal. (interno)", "F1 macro (oficial)",
                  "Exact. bal. (oficial)"]
    df.to_csv(TAB / "tabla4_ensembles.csv", index=False)


def convergence():
    out = {}
    for m in MODELS:
        a, b = load(f"{m}_seed0"), load(f"{m}_seed0_e30")
        out[m] = {"15": {"best_epoch": a["training"]["best_epoch"], "epochs_run": a["training"]["epochs_run"],
                         "test_f1": a["test"]["f1_macro"], "official_f1": a["test_official"]["f1_macro"]},
                  "30": {"best_epoch": b["training"]["best_epoch"], "epochs_run": b["training"]["epochs_run"],
                         "test_f1": b["test"]["f1_macro"], "official_f1": b["test_official"]["f1_macro"]}}
    return out


def table5(conv):
    sign = lambda d: f"{100 * d:+.1f}"
    rows = []
    for m in MODELS:
        a, b = conv[m]["15"], conv[m]["30"]
        rows.append([NAMES[m], f"{a['best_epoch']} / {a['epochs_run']}", f"{b['best_epoch']} / {b['epochs_run']}",
                     pct(a["test_f1"]), pct(b["test_f1"]), sign(b["test_f1"] - a["test_f1"]),
                     pct(a["official_f1"]), pct(b["official_f1"]), sign(b["official_f1"] - a["official_f1"])])
    df = pd.DataFrame(rows, columns=["Modelo", "15 ép.", "30 ép.", "15 ép. ", "30 ép. ", "Δ", "15 ép.  ",
                                     "30 ép.  ", "Δ "])
    write_table("tabla5_convergencia", df, "Experimento de convergencia: máximo de 15 contra 30 épocas (semilla 0).",
                "Época: época del mejor F1 macro de validación / épocas corridas hasta el early stopping (paciencia 3). "
                "F1 macro en porcentaje; Δ en puntos porcentuales. La corrida de 30 épocas parte de los pesos de ImageNet "
                "con el cosine ajustado a 30 épocas, por lo que no es una continuación de la de 15. Una sola semilla: "
                "comparar Δ con la desviación entre semillas de la Tabla 1.",
                groups=[("Época", 2), ("F1 test interno", 3), ("F1 test oficial", 3)], wide=True)
    df.columns = ["Modelo", "Época mejor/corridas (15)", "Época mejor/corridas (30)", "F1 interno (15)",
                  "F1 interno (30)", "Δ interno (pp)", "F1 oficial (15)", "F1 oficial (30)", "Δ oficial (pp)"]
    df.to_csv(TAB / "tabla5_convergencia.csv", index=False)


def ablation():
    out = {}
    for m in MODELS:
        out[m] = {}
        for mode, (suffix, _) in WEIGHTS.items():
            r = load(f"{m}_seed0{suffix}")
            p, rc = prec_rec(r["test"]["confusion_matrix"])
            out[m][mode] = {
                "test_f1": r["test"]["f1_macro"], "test_bacc": r["test"]["balanced_accuracy"],
                "test_acc": r["test"]["accuracy"],
                "official_f1": r["test_official"]["f1_macro"], "official_bacc": r["test_official"]["balanced_accuracy"],
                "precision": {c: float(p[CLASSES.index(c)]) for c in CLASSES},
                "recall": {c: float(rc[CLASSES.index(c)]) for c in CLASSES},
                "best_epoch": r["training"]["best_epoch"]}
    return out


def table6(abl):
    rows = []
    for m in MODELS:
        for mode, (_, label) in WEIGHTS.items():
            a = abl[m][mode]
            rows.append([NAMES[m] if mode == "inverse" else "", label, pct(a["test_f1"]), pct(a["test_bacc"]),
                         pct(a["official_f1"]), pct(a["official_bacc"])]
                        + [f"{100 * a['precision'][c]:.1f} / {100 * a['recall'][c]:.1f}" for c in MINOR])
    df = pd.DataFrame(rows, columns=["Modelo", "Ponderación", "F1 macro", "Exact. bal.", "F1 macro ", "Exact. bal. "]
                      + [f"{c} (P / R)" for c in MINOR])
    write_table("tabla6_ablacion_pesos", df,
                "Ablación de la ponderación de clases en la pérdida (semilla 0), en porcentaje.",
                "Frecuencia inversa: protocolo principal (pesos 1/n, normalizados a media 1). Raíz cuadrada: 1/raíz(n). "
                "Sin pesos: cross-entropy estándar. P / R: precisión / recall en el test interno. Todas las variantes "
                "con semilla 0 y el resto del protocolo idéntico.",
                groups=[("", 1), ("Test interno", 2), ("Test oficial", 2), ("Clases minoritarias, test interno", 4)],
                wide=True, colsep="3pt")
    df["Modelo"] = [NAMES[m] for m in MODELS for _ in WEIGHTS]
    df.columns = ["Modelo", "Ponderación", "F1 macro (interno)", "Exact. bal. (interno)", "F1 macro (oficial)",
                  "Exact. bal. (oficial)"] + [f"{c} P / R (interno)" for c in MINOR]
    df.to_csv(TAB / "tabla6_ablacion_pesos.csv", index=False)


# ---------- figuras ----------

def legend_models(fig_or_ax, extra=(), **kw):
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=NAMES[m]) for m in MODELS] + list(extra)
    return fig_or_ax.legend(handles=handles, **kw)


def fig1_curves():
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    curves = {}
    for m in MODELS:
        hs = [pd.read_csv(RUNS / f"{m}_seed{s}" / "history.csv").set_index("epoch")["val_f1_macro"] for s in SEEDS]
        df = pd.concat(hs, axis=1)
        mean, sd, n = df.mean(axis=1), df.std(axis=1, ddof=1), df.count(axis=1)
        band = n >= 2
        ax.plot(mean.index, 100 * mean, color=COLORS[m], marker=MARKERS[m], markersize=2.5)
        ax.fill_between(mean.index[band], 100 * (mean - sd)[band], 100 * (mean + sd)[band], color=COLORS[m],
                        alpha=0.18, lw=0)
        e30 = pd.read_csv(RUNS / f"{m}_seed0_e30" / "history.csv")
        ax.plot(e30["epoch"], 100 * e30["val_f1_macro"], color=COLORS[m], ls=(0, (2.5, 1.5)), lw=1.0)
        curves[m] = {"epoch": mean.index.tolist(), "mean": mean.round(5).tolist(),
                     "sd": sd.round(5).where(band).tolist(), "n_seeds": n.tolist(),
                     "e30": e30["val_f1_macro"].tolist()}
    ax.set_xlabel("Época")
    ax.set_ylabel("F1 macro en validación (%)")
    ax.set_xlim(0.5, 23.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.axvline(15, color="#999999", lw=0.6, ls=":")
    legend_models(ax, extra=[Line2D([], [], color="#444444", ls=(0, (2.5, 1.5)), lw=1.0,
                                    label="30 épocas (semilla 0)")], loc="lower right", ncol=1)
    save(fig, FIG / "fig1_curvas_val")
    return curves


def bars(ax, groups, values, errors, width=0.26, labels=True, fmt="{:.1f}", fontsize=5.5):
    x = np.arange(len(groups))
    for k, m in enumerate(MODELS):
        pos = x + (k - 1) * width
        ax.bar(pos, values[m], width * 0.92, color=COLORS[m], hatch=HATCHES[m], edgecolor="white", lw=0.4,
               label=NAMES[m], zorder=2)
        ax.errorbar(pos, values[m], yerr=errors[m], fmt="none", ecolor="#222222", elinewidth=0.6, capsize=1.5,
                    zorder=3)
        if labels:
            for p, v, e in zip(pos, values[m], errors[m]):
                ax.text(p, v + e + 0.8, fmt.format(v), ha="center", va="bottom", fontsize=fontsize)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


def fig2_bars(S):
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.2))
    vals = {m: [100 * S["models"][m][sp]["f1_macro"][0] for sp in ["test", "test_official"]] for m in MODELS}
    errs = {m: [100 * S["models"][m][sp]["f1_macro"][1] for sp in ["test", "test_official"]] for m in MODELS}
    bars(ax, ["Test interno", "Test oficial"], vals, errs, fontsize=6)
    ax.set_ylabel("F1 macro (%)")
    ax.set_ylim(0, 75)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), handlelength=1.2, columnspacing=1.0)
    save(fig, FIG / "fig2_f1_interno_oficial")


def fig3_confusion():
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.55), layout="constrained")
    out = {}
    for ax, m in zip(axes, MODELS):
        cm = np.asarray(load(f"{m}_seed0")["test"]["confusion_matrix"], dtype=float)
        norm = 100 * cm / cm.sum(1, keepdims=True)
        out[m] = norm.round(2).tolist()
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{norm[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if norm[i, j] > 55 else "#222222")
        ax.set_xticks(range(8), CLASSES, rotation=90)
        ax.set_yticks(range(8), CLASSES)
        ax.set_xlabel("Clase predicha")
        ax.set_title(NAMES[m], fontsize=8)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Clase real")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01, aspect=25)
    cb.set_label("Porcentaje de la clase real (%)", fontsize=7)
    cb.outline.set_visible(False)
    save(fig, FIG / "fig3_matrices_confusion")
    return out


def fig4_per_class(S):
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.1))
    vals = {m: [100 * S["models"][m]["test"]["f1_per_class"][c][0] for c in CLASSES] for m in MODELS}
    errs = {m: [100 * S["models"][m]["test"]["f1_per_class"][c][1] for c in CLASSES] for m in MODELS}
    bars(ax, CLASSES, vals, errs, labels=False)
    ax.set_ylabel("F1 en test interno (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", ncol=3, handlelength=1.2)
    save(fig, FIG / "fig4_f1_por_clase")


def fig5_cost(S, cost):
    fig, axes = plt.subplots(1, 2, figsize=(ONE_COL, 1.9), sharey=True, layout="constrained")
    specs = [("params_total", 1e-6, "Parámetros (millones)", True), ("epoch_s_mean", 1, "Segundos por época", False)]
    # desplazamiento de cada etiqueta en puntos, para que no se encime con el punto ni con su barra de error
    offsets = [{"resnet50": (-5, -9, "right"), "efficientnet_b0": (-4, 8, "left"), "swin_t": (-5, 4, "right")},
               {"resnet50": (-5, -9, "right"), "efficientnet_b0": (0, 8, "center"), "swin_t": (-5, 4, "right")}]
    for ax, (col, scale, label, log), off in zip(axes, specs, offsets):
        for m in MODELS:
            x = cost.loc[m, col] * scale
            y, e = (100 * v for v in S["models"][m]["test"]["f1_macro"])
            ax.errorbar(x, y, yerr=e, fmt=MARKERS[m], color=COLORS[m], markersize=5, elinewidth=0.8, capsize=2)
            dx, dy, ha = off[m]
            ax.annotate(NAMES[m], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=6.5)
        ax.set_xlabel(label)
        if log:
            ax.set_xscale("log")
            ax.set_xlim(2.5, 45)
            ax.set_xticks([3, 10, 30], ["3", "10", "30"])
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xlim(15, 52)
    axes[0].set_ylabel("F1 macro, test interno (%)")
    axes[0].set_ylim(48, 63)
    axes[0].set_yticks([50, 55, 60])
    save(fig, FIG / "fig5_f1_vs_costo")


def fig6_gap(S):
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.2))
    splits = ["train", "val", "test", "test_official"]
    out = {}
    for k, m in enumerate(MODELS):
        x = np.arange(4) + (k - 1) * 0.09
        mean = [100 * S["models"][m][sp]["f1_macro"][0] for sp in splits]
        sd = [100 * S["models"][m][sp]["f1_macro"][1] for sp in splits]
        out[m] = {sp: S["models"][m][sp]["f1_macro"] for sp in splits}
        ax.errorbar(x, mean, yerr=sd, color=COLORS[m], marker=MARKERS[m], markersize=4, capsize=2, elinewidth=0.7,
                    label=NAMES[m])
    ax.set_xticks(range(4), [SPLIT_NAMES[s] for s in splits])
    ax.set_ylabel("F1 macro (%)")
    ax.set_ylim(45, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, FIG / "fig6_brecha_generalizacion")
    return out


def fig7_ablation(abl):
    fig, axes = plt.subplots(1, 2, figsize=(ONE_COL, 2.0), sharey=True, layout="constrained")
    xt = ["Frec.\ninversa", "Raíz\ncuadrada", "Sin\npesos"]
    for ax, (key, title) in zip(axes, [("test_f1", "F1 macro"), ("test_bacc", "Exactitud balanceada")]):
        for k, m in enumerate(MODELS):
            # desplazamiento horizontal leve para que no se tapen puntos con valores casi iguales
            ax.plot(np.arange(3) + (k - 1) * 0.07, [100 * abl[m][mode][key] for mode in WEIGHTS], color=COLORS[m],
                    marker=MARKERS[m], markersize=4)
        ax.set_xticks(range(3), xt)
        ax.set_xlim(-0.3, 2.3)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test interno, semilla 0 (%)")
    legend_models(fig, loc="outside upper center", ncol=3, handlelength=1.5, columnspacing=1.0)
    save(fig, FIG / "fig7_ablacion_pesos")


def main():
    setup()
    FIG.mkdir(exist_ok=True)
    S = collect()
    cost = pd.read_csv(RES / "cost.csv").set_index("model")
    ens = json.loads((RES / "ensembles.json").read_text())
    support = pd.read_csv(RES / "splits" / "class_counts.csv").set_index("class")["test"].to_dict()
    conv, abl = convergence(), ablation()

    table1(S)
    table2(S, cost)
    table3(S, support)
    table4(S, ens)
    table5(conv)
    table6(abl)
    S["curves_val"] = fig1_curves()
    fig2_bars(S)
    S["confusion_seed0_test_pct"] = fig3_confusion()
    fig4_per_class(S)
    fig5_cost(S, cost)
    S["generalization_f1_macro"] = fig6_gap(S)
    fig7_ablation(abl)

    S["cost"] = cost.reset_index().to_dict(orient="records")
    S["ensembles"] = {k: {kk: (vv if kk == "runs" else {x: y for x, y in vv.items() if x != "confusion_matrix"})
                          for kk, vv in v.items()} for k, v in ens.items()}
    S["convergence"], S["ablation"] = conv, abl
    S["support_test"] = {c: int(support[c]) for c in CLASSES}
    S["gradcam_images"] = pd.read_csv(RES / "gradcam_images.csv").to_dict(orient="records")
    S["notes"] = {"aggregate": "pares [media, desviación estándar muestral] entre semillas 0, 1 y 2; proporciones en [0, 1]",
                  "macs": "gmacs es la salida de fvcore (multiplicaciones-acumulaciones); gflops = 2 x gmacs",
                  "epoch_s": "media de las épocas 2 en adelante; el tiempo total incluye todas"}
    (RES / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
    print("Tablas en results/tables/, figuras en figures/, resumen en results/summary.json")


if __name__ == "__main__":
    main()
