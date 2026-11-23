"""Figures of the manuscript and Supplement that are produced by code."""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from . import config as C  # noqa: E402

BASE_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True}
PAL = {
    "neutral": dict(edge="#607d8b", fill="#f3f7fa", title="#263238"),
    "grey": dict(edge="#90a4ae", fill="#f3f7fa", title="#455a64"),
    "blue": dict(edge="#5c88a5", fill="#edf5fa", title="#2f5f80"),
    "green": dict(edge="#2e7d32", fill="#f5faf6", title="#2e7d32"),
    "orange": dict(edge="#d84315", fill="#fff8f5", title="#d84315"),
    "purple": dict(edge="#6a4c93", fill="#f7f4fb", title="#6a4c93"),
    "amber": dict(edge="#e09a45", fill="#fff8ee", title="#a65d00"),
}


def save(fig, stem):
    pdf = C.FIGURES_DIR / f"{stem}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    fig.savefig(C.FIGURES_DIR / f"{stem}_preview.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- Figure 1
def roc_auc_per_split(auc: dict):
    """auc: (feature set, model) -> five split-level ROC-AUC values. Points = splits, diamonds = means."""
    display = {"Diet + Taxa": "Diet + taxa", "Confounders + Diet + Taxa": "Confounders +\ndiet + taxa"}
    col = {"LR": "#3c78b6", "SVM": "#d6604d", "BRF": "#61b641"}
    dark = {"LR": "#1a5189", "SVM": "#ab4c3d", "BRF": "#4e9a35"}
    with plt.rc_context({**BASE_RC, "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        width, offs = 0.24, {"LR": -0.27, "SVM": 0.0, "BRF": 0.27}
        jit = np.array([-0.06, -0.03, 0.0, 0.03, 0.06])
        for gi, fs in enumerate(C.FEATURE_SETS):
            for m in C.MODELS:
                x, v = gi + offs[m], np.asarray(auc[(fs, m)], float)
                bp = ax.boxplot(v, positions=[x], widths=width, patch_artist=True, showfliers=False, whis=1.5,
                                medianprops=dict(color="black", lw=1.4), whiskerprops=dict(color="black", lw=0.8),
                                capprops=dict(color="black", lw=1.0), boxprops=dict(lw=0))
                bp["boxes"][0].set_facecolor(col[m])
                bp["boxes"][0].set_alpha(0.4)
                ax.scatter(x + jit, v, s=16, color=dark[m], zorder=3, lw=0)
                ax.scatter([x], [v.mean()], marker="D", s=52, facecolor="white", edgecolor=dark[m], lw=1.6, zorder=4)
                if (fs, m) == C.MAIN_CONFIG:      # outline around the main configuration only
                    wy = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
                    y_lo, y_hi = min(wy.min(), v.min()) - 0.035, max(wy.max(), v.max()) + 0.035
                    x_lo, x_hi = x - width / 2 - 0.012, x + width / 2 + 0.012
                    ax.add_patch(FancyBboxPatch((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                                                boxstyle="round,pad=0,rounding_size=0.03", fill=False, ec="#e09a45",
                                                lw=1.3, ls=(0, (4, 2)), zorder=1, mutation_aspect=1 / 6))
        ax.axhline(0.5, color="grey", ls="--", lw=0.9, zorder=0)
        for gi in range(1, len(C.FEATURE_SETS)):
            ax.axvline(gi - 0.5, color="#e0e0e0", lw=0.6, zorder=0)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(C.FEATURE_SETS) - 0.5)
        ax.set_xticks(range(len(C.FEATURE_SETS)))
        ax.set_xticklabels([display.get(fs, fs) for fs in C.FEATURE_SETS], fontsize=9.5)
        ax.set_ylabel("ROC-AUC", fontsize=10.5)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        handles = [Patch(facecolor=col[m], alpha=0.4, label={"LR": "LR", "SVM": "Linear SVM", "BRF": "Balanced RF"}[m])
                   for m in C.MODELS]
        handles += [Line2D([], [], marker="D", ls="", mfc="white", mec="#455a64", mew=1.4, ms=6.5, label="Mean of 5 splits"),
                    Patch(fill=False, ec="#e09a45", ls=(0, (4, 2)), lw=1.3, label="Main configuration")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=5, frameon=False, fontsize=8.6,
                  handlelength=1.4, columnspacing=1.2)
        fig.tight_layout()
        save(fig, "figure1_roc_auc_per_split")


# ----------------------------------------------------------------------------- Figure 2
def roc_curve_and_confusion(pred):
    """pred: held-out predictions of the main configuration (5 splits x 18). Returns the pooled counts."""
    grid, tprs = np.linspace(0, 1, 201), []
    for _, g in pred.groupby("Split"):
        fpr, tpr, _ = roc_curve(g["y_true"], g["y_score"], drop_intermediate=False)
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    from sklearn.metrics import roc_auc_score
    mean_auc = float(np.mean([roc_auc_score(g["y_true"], g["y_score"]) for _, g in pred.groupby("Split")]))
    with plt.rc_context({**BASE_RC, "font.family": "Arial"}):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for t in tprs:
            ax.plot(grid, t, color="#f28e2b", alpha=0.28, lw=1.0, drawstyle="default")
        ax.plot(grid, mean_tpr, color="#f28e2b", lw=2.6, label="Mean ROC curve (5 test sets)")
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1.0)
        ax.plot([], [], color="#f28e2b", alpha=0.35, lw=1.0, label="Individual test sets")
        ax.text(0.97, 0.05, f"Mean ROC-AUC = {mean_auc:.3f}", transform=ax.transAxes, ha="right", fontsize=11)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ticks = [0, 0.25, 0.5, 0.75, 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{t:.2f}" for t in ticks])
        ax.set_yticklabels([f"{t:.2f}" for t in ticks])
        ax.set_xlabel("False positive rate", fontsize=12)
        ax.set_ylabel("True positive rate", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=10)
        fig.tight_layout()
        save(fig, "figure2a_roc_curve")

        y, yp = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
        tp, fn = int(((y == 1) & (yp == 1)).sum()), int(((y == 1) & (yp == 0)).sum())
        tn, fp = int(((y == 0) & (yp == 0)).sum()), int(((y == 0) & (yp == 1)).sum())
        cm = np.array([[fn, tp], [tn, fp]])        # rows: depressive, control; columns: predicted control, depressive
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f7fbff", "#4a77aa"])
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{100 * cm[i, j] / cm[i].sum():.1f}%", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="black", linespacing=1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Depressive"], fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Depressive", "Control"], fontsize=12)
        ax.set_xlabel("Predicted label", fontsize=13)
        ax.set_ylabel("True label", fontsize=13)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.05)
        cb.ax.tick_params(labelsize=10)
        cb.outline.set_visible(False)
        fig.tight_layout()
        fig.subplots_adjust(left=0.24)
        save(fig, "figure2b_confusion_matrix")
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "mean_auc": mean_auc}


# ----------------------------------------------------------------------------- Figure 3
def coefficients_across_splits(prio, labels: dict):
    """prio: prioritised dietary and microbial features (median-rule table) in display order."""
    color = {"Diet": PAL["orange"]["edge"], "Taxa": PAL["green"]["edge"], "Confounder": PAL["purple"]["edge"]}
    coef = prio[[f"Coefficient split {s}" for s in range(1, 6)]].to_numpy(float)
    kept = prio[[f"Retained split {s}" for s in range(1, 6)]].to_numpy(bool)
    groups = prio["Feature group"].tolist()
    n = len(prio)
    with plt.rc_context({**BASE_RC, "font.family": "Arial"}):
        fig, ax = plt.subplots(figsize=(7.2, 5.3))
        off = {s: (s - 3) * 0.18 for s in range(1, 6)}
        for xi in range(n):
            col = color[groups[xi]]
            present = [(s, coef[xi, s - 1]) for s in range(1, 6) if not np.isnan(coef[xi, s - 1])]
            vals = np.array([v for _, v in present])
            ax.bar(xi, vals.max() - vals.min() if len(vals) > 1 else 0.02,
                   bottom=vals.min() if len(vals) > 1 else vals[0] - 0.01, width=0.72, color=col, alpha=0.13, lw=0,
                   zorder=1)
            for s, c in present:
                filled = bool(kept[xi, s - 1])
                ax.scatter(xi + off[s], c, marker="s", s=40, facecolor=col if filled else "white", edgecolor=col,
                           lw=1.2, zorder=3)
            ax.text(xi, 1.0, f"{int(prio['Splits retained'].iloc[xi])}/5", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.6, color="#455a64")
        ax.axhline(0, color="#555555", ls="--", lw=1.0, zorder=0)
        ax.set_xticks(range(n))
        ax.set_xticklabels([labels[f] for f in prio["Feature ID"]], rotation=45, ha="right", rotation_mode="anchor",
                           fontsize=8.6)
        for tl, g in zip(ax.get_xticklabels(), groups):
            if g == "Taxa":
                tl.set_fontstyle("italic")
        ax.set_xlim(-0.6, n - 0.4)
        ylo, yhi = np.nanmin(coef), np.nanmax(coef)
        pad = 0.12 * (yhi - ylo)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_ylabel("LR coefficient (standardized features)", fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate("toward depressive class", xy=(1.005, 0.98), xycoords="axes fraction", ha="left", va="top",
                    fontsize=7.8, color="#455a64", rotation=90)
        ax.annotate("toward control class", xy=(1.005, 0.02), xycoords="axes fraction", ha="left", va="bottom",
                    fontsize=7.8, color="#455a64", rotation=90)
        ax.text(-0.6, 1.0, "Splits retained:", transform=ax.get_xaxis_transform(), ha="right", va="bottom",
                fontsize=7.6, color="#455a64")
        names = {"Diet": "Dietary variable", "Taxa": "Taxon", "Confounder": "Confounder"}
        handles = [Line2D([], [], marker="s", ls="", mfc=color[g], mec=color[g], ms=7, label=names[g])
                   for g in dict.fromkeys(groups)]
        handles += [Line2D([], [], marker="s", ls="", mfc="#78909c", mec="#78909c", ms=7,
                           label="|coefficient| ≥ split median"),
                    Line2D([], [], marker="s", ls="", mfc="white", mec="#78909c", mew=1.2, ms=7,
                           label="in model, below split median")]
        fig.subplots_adjust(left=0.115, right=0.955, top=0.87, bottom=0.30)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=len(handles), frameon=False,
                   fontsize=7.8, handletextpad=0.3, columnspacing=1.0)
        save(fig, "figure3_lr_coefficients")


# ----------------------------------------------------------------------------- Figure 5
def workflow():
    """Schematic of the study workflow (no data). Every box has a title and at most one short subtitle."""
    W, H = 1026.0, 947.0
    TF, SF = 21.0, 19.5
    G, H2, H1 = 17.0, 56.0, 44.0
    with plt.rc_context({**BASE_RC, "font.family": "Arial"}):
        fig = plt.figure(figsize=(W / 72, H / 72))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        boxes, texts = {}, []

        def box(name, x, y, w, style, title, sub=None, h=None):
            h = h or (H2 if sub else H1)
            c = PAL[style]
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9", lw=1.3, ec=c["edge"],
                                        fc=c["fill"], zorder=2))
            boxes[name] = (x, y, w, h)
            if sub:
                texts.append((ax.text(x + w / 2, y + 7, title, ha="center", va="top", fontsize=TF, fontweight="bold",
                                      color=c["title"], zorder=3), name))
                texts.append((ax.text(x + w / 2, y + 31, sub, ha="center", va="top", fontsize=SF, fontweight="bold",
                                      color="#37474f", zorder=3), name))
            else:
                texts.append((ax.text(x + w / 2, y + h / 2 + 1, title, ha="center", va="center", fontsize=TF,
                                      fontweight="bold", color=c["title"], zorder=3), name))

        def arrow(p0, p1):
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#455a64",
                                         shrinkA=0, shrinkB=0.5, zorder=1))

        def bottom(n, fx=0.5):
            x, y, w, h = boxes[n]
            return (x + w * fx, y + h)

        def top(n, fx=0.5):
            x, y, w, h = boxes[n]
            return (x + w * fx, y)

        def centered(w):
            return (W - w) / 2

        y = 10.0
        box("data", centered(830), y, 830, "neutral", "Dataset",
            "Microbiome taxonomy table • Dietary data • Confounders • Depression status")
        y += H2 + G
        box("pre", centered(300), y, 300, "neutral", "Preprocessing")
        y += H1 + G + 6
        xm, wm = 12.0, 380.0
        xd, wd = xm + wm + 16, 320.0
        xc, wc = xd + wd + 16, W - 12 - (xd + wd + 16)
        rows = [y + i * (H2 + G) for i in range(4)]
        box("m1", xm + 20, rows[0] + (H2 - H1) / 2, wm - 40, "green", "Microbiome Data")
        box("m2", xm, rows[1], wm, "green", "Filtering & Transformation", "Prevalence, minimum count, CLR")
        box("m3", xm, rows[2], wm, "green", "Taxa Feature Selection", "ANOVA F-test + MI + RF importance")
        box("m4", xm + 20, rows[3], wm - 40, "green", "Consensus Selection", "Selected by ≥ 2 methods")
        box("d1", xd + 20, rows[0] + (H2 - H1) / 2, wd - 40, "orange", "Dietary Data")
        box("d2", xd + 20, rows[1] + (H2 - H1) / 2, wd - 40, "orange", "Energy Adjustment")
        box("d3", xd, rows[2] + (H2 - H1) / 2, wd, "orange", "Spearman Correlation Filter")
        box("c1", xc, rows[0], wc, "purple", "Confounders", "Age • BMI • SQ_all • PA")
        y = rows[3] + H2 + G + 4
        box("fs", centered(800), y, 800, "blue", "Five Feature Sets",
            "Confounders | Diet | Taxa | Diet + Taxa | Confounders + Diet + Taxa")
        y += H2 + G
        box("clf", centered(560), y, 560, "grey", "Three Classifiers", "Logistic Regression • SVM • BRF")
        y += H2 + G
        box("eval", centered(700), y, 700, "grey", "Performance Evaluation",
            "Five repeated test splits • Bootstrap CI • Permutation test")
        y += H2 + G
        box("main", centered(560), y, 560, "amber", "Main Configuration", "LR with Confounders + Diet + Taxa")
        y += H2 + G
        box("prio", centered(640), y, 640, "grey", "Candidate-Feature Prioritisation",
            "LR coefficients, adjusted Firth regression")
        y += H2 + G
        box("out_t", 150, y, 320, "green", "Microbiome (Taxa)")
        box("out_d", W - 150 - 320, y, 320, "orange", "Dietary Variables")
        y += H1 + G
        box("func", 310 - 260, y, 520, "grey", "Functional Analysis", "PICRUSt2-predicted KEGG Orthology profiles")

        arrow(bottom("data"), top("pre"))
        for n in ("m1", "d1", "c1"):
            x_, y_ = top(n)
            xp = min(max(x_, boxes["pre"][0] + 30), boxes["pre"][0] + boxes["pre"][2] - 30)
            arrow((xp, boxes["pre"][1] + boxes["pre"][3]), (x_, y_))
        for a, b in (("m1", "m2"), ("m2", "m3"), ("m3", "m4"), ("d1", "d2"), ("d2", "d3")):
            arrow(bottom(a), top(b))
        fs_x, fs_y, fs_w, _ = boxes["fs"]
        for n in ("m4", "d3", "c1"):
            x_ = min(max(bottom(n)[0], fs_x + 40), fs_x + fs_w - 40)
            arrow((x_, bottom(n)[1]), (x_, fs_y))
        for a, b in (("fs", "clf"), ("clf", "eval"), ("eval", "main"), ("main", "prio")):
            arrow(bottom(a), top(b))
        arrow(bottom("prio", 0.30), top("out_t", 0.62))
        arrow(bottom("prio", 0.70), top("out_d", 0.38))
        arrow(bottom("out_t"), top("func"))

        # layout checks: no text outside its box, no overlapping boxes
        fig.canvas.draw()
        renderer, inv = fig.canvas.get_renderer(), ax.transData.inverted()
        for t, name in texts:
            bb = t.get_window_extent(renderer)
            (x0, y1), (x1, y0) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            x, y_, w, h = boxes[name]
            assert x0 >= x + 6 and x1 <= x + w - 6 and y0 >= y_ + 2 and y1 <= y_ + h - 2, name
        names = list(boxes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ax0, ay0, aw, ah = boxes[a]
                bx0, by0, bw, bh = boxes[b]
                assert not (ax0 < bx0 + bw + 8 and bx0 < ax0 + aw + 8 and ay0 < by0 + bh + 8 and by0 < ay0 + ah + 8)
        assert not [t for t, _ in texts if re.search(r"\d", t.get_text().replace("PICRUSt2", "").replace("≥ 2", ""))]
        save(fig, "figure5_workflow")


# ----------------------------------------------------------------------------- Supplementary Figure S1
def diet_correlation_heatmap(corr):
    """Spearman correlations among the eight energy-adjusted dietary variables (all participants)."""
    names = list(corr.columns)
    values = corr.to_numpy(float)
    with plt.rc_context({**BASE_RC, "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(8.54, 7.38))
        im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(values[i, j]) > 0.6 else "black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1)
        ax.tick_params(which="minor", length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Spearman rho")
        ax.set_title("Spearman correlation among energy-adjusted dietary variables", fontsize=11, pad=12)
        fig.tight_layout()
        save(fig, "figureS1_diet_spearman_heatmap")
